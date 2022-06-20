import numpy as np
import tensorflow as tf
from contextlib import contextmanager
from tensorflow import keras
from sklearn.cluster import KMeans

from lib.metrics import ordered_cmat
from tf_lib.backbones import create_backbone
from tf_lib.loss import Loss
from tf_lib.kernel import cdist
from tf_lib.tf_helpers import optimizer_from_config


class TrainingPhaseMonitor(keras.callbacks.Callback):
    def __init__(self, model_cfg):
        super(TrainingPhaseMonitor, self).__init__()
        self.target_dist_update_interval = model_cfg.target_dist_update_interval

    def on_epoch_end(self, epoch, logs=None):
        if (epoch % self.target_dist_update_interval) == 0:
            self.model.update_target_dist()


class AE(keras.Model):
    def __init__(self, encoder, bottleneck, decoder):
        super(AE, self).__init__()

        self.encoder = encoder
        self.bottleneck = bottleneck
        self.decoder = decoder
        self.flatten = keras.layers.Flatten()
        self.mse_loss = keras.losses.MeanSquaredError()
        self.loss_values = None

    def call(self, inputs):
        if isinstance(inputs, dict):
            inputs = inputs["data"]
        reconstruction = self.decoder(self.bottleneck(self.encoder(inputs))[1])
        loss = self.mse_loss(self.flatten(inputs), reconstruction)
        self.add_loss(loss)
        self.add_metric(loss, name="reconstruct")
        self.loss_values = {"reconstruct": loss}
        return reconstruction


class ClusteringModule(keras.layers.Layer):
    def __init__(self, cfg, input_size):
        super(ClusteringModule, self).__init__()
        self.cfg = cfg
        self.centroids = self.add_weight(shape=(self.cfg.n_clusters, input_size[0]), initializer="random_normal",
                                         trainable=True)
        self.pow = -1 * (self.cfg.alpha + 1) / 2

    def call(self, inputs):
        dists = cdist(inputs, self.centroids)
        sim = (1 + dists / self.cfg.alpha) ** self.pow
        norm_fac = tf.reduce_sum(sim, axis=1, keepdims=True)
        sim = sim / norm_fac
        return sim

    def init_centroids(self, data, labels=None):
        km = KMeans(n_clusters=self.cfg.n_clusters, n_init=20)
        pred = km.fit_predict(data)
        self.set_weights([km.cluster_centers_])

        if labels is not None:
            # Only use labels to evaluate k-means accuracy.
            acc, cmat = ordered_cmat(labels, pred)
            print("k-means accuracy:", acc)
            print("k-means confusion matrix:\n", cmat)


class TargetDistribution(keras.layers.Layer):
    def __init__(self, shape):
        super(TargetDistribution, self).__init__()

        self._shape = shape
        initial = np.random.normal(size=shape)
        initial = initial / initial.sum(axis=1, keepdims=True)
        self._dist = self.add_weight(shape=shape, initializer=lambda *args, **kwargs: initial,
                                     trainable=False, name="target_dist")

    def update_dist(self, q):
        un_normed = (q ** 2) / q.sum(axis=0, keepdims=True)
        normed = un_normed / un_normed.sum(axis=1, keepdims=True)
        filler = np.full((self._shape[0] - q.shape[0], self._shape[1]), fill_value=np.nan)
        new_dist = np.concatenate((normed, filler), axis=0)
        self.set_weights([new_dist])

    def call(self, idx):
        return tf.gather(self._dist, idx)


class Bottleneck(keras.layers.Layer):
    def __init__(self, bottleneck_units, input_size, activation=(None, None)):
        super(Bottleneck, self).__init__()
        
        self.input_size = input_size
        self.flatten = keras.layers.Flatten()
        self.layer_1 = keras.layers.Dense(units=bottleneck_units, activation=activation[0])
        self.layer_2 = keras.layers.Dense(units=np.prod(input_size), activation=activation[1])

    def call(self, x):
        hidden = self.layer_1(self.flatten(x))
        out = tf.reshape(self.layer_2(hidden), (-1, *self.input_size))
        return hidden, out


class DummyBottleneck(keras.layers.Layer):
    def call(self, x):
        return x, x


class TFDEC(keras.Model):
    def __init__(self, cfg):
        super(TFDEC, self).__init__()

        self.cfg = cfg
        self.backbone = create_backbone(cfg.backbone_config, flatten_output=False)
        self.decoder = create_backbone(cfg.decoder_config, input_size=self.backbone.output_size)
        self.flatten = keras.layers.Flatten()

        if cfg.bottleneck_units is None:
            self.bottleneck = DummyBottleneck()
            hidden_size = np.prod(self.backbone.output_size)
        else:
            self.bottleneck = Bottleneck(cfg.bottleneck_units, self.backbone.output_size, activation=(None, "relu"))
            hidden_size = cfg.bottleneck_units

        self.clustering_module = ClusteringModule(cfg.cm_config, input_size=[hidden_size])

        assert "kl" not in cfg.loss_config.funcs, "'kl' should not be specified as a loss_func in TFDEC!"
        self.loss_layer = Loss(cfg.loss_config)
        self.target_dist = TargetDistribution(shape=(int(1e5), cfg.cm_config.n_clusters))

        self.reconstruct_in_fine_tune = "reconstruct" in cfg.loss_config.funcs

        self.inputs = self.targets = self.backbone_output = self.hidden = self.decoder_input = self.outputs =\
            self.reconstruction = None
        self.train_data = None
        self.training = False

    def call(self, inputs, training=False):
        self.training = training
        self.inputs = inputs["data"]
        self.targets = self.target_dist(inputs["idx"])
        self.backbone_output = self.backbone(self.inputs)
        self.hidden, self.decoder_input = self.bottleneck(self.backbone_output)

        # if self.reconstruct_in_fine_tune:
        #     self.reconstruction = self.decoder(self.decoder_input)
        # else:
        #     self.reconstruction = None
        self.reconstruction = self.decoder(self.decoder_input)

        self.outputs = self.clustering_module(self.flatten(self.hidden))

        loss_values = self.loss_layer(self)
        self._update_losses_and_metrics(loss_values)
        return self.outputs

    def test_step(self, data):
        inputs, _ = data
        targets = self.target_dist(inputs["idx"])
        pred = self(inputs, training=False)
        self.compiled_loss(targets, pred, regularization_losses=self.losses)
        self.compiled_metrics.update_state(targets, pred)
        return {m.name: m.result() for m in self.metrics}

    def _update_losses_and_metrics(self, loss_values):
        for key, value in loss_values.items():
            self.add_loss(value)
            self.add_metric(value, name=key)

    def pre_train(self, train_loader, callbacks):
        self._print_msg("Pre-training autoencoder")
        autoencoder = AE(encoder=self.backbone, bottleneck=self.bottleneck, decoder=self.decoder)
        autoencoder.compile(optimizer=optimizer_from_config(self.cfg.pre_train_optimizer_config))
        autoencoder.fit(train_loader, epochs=self.cfg.n_pre_train_epochs, callbacks=callbacks, verbose=False)

    def init_training(self):
        self._print_msg("Pre-training done. Initializing centroids.")

        data, idx, labels = self.train_data
        hidden = []
        batch_size = 256
        for i in range(int(np.ceil(data.shape[0] / batch_size))):
            batch = data[(i * batch_size): ((i+1) * batch_size)]
            hidden.append(self.bottleneck(self.backbone(batch))[0].numpy())

        hidden = np.concatenate(hidden, axis=0)
        self.clustering_module.init_centroids(hidden, labels)
        self.reconstruction = None
        # self.update_target_dist()

    def update_target_dist(self):
        self._print_msg("Updating target distribution")

        data, idx, _ = self.train_data
        predictions = self.predict({"data": data, "idx": idx.astype(int)}, batch_size=256)
        self.target_dist.update_dist(predictions)

    @staticmethod
    def _print_msg(msg, sep=(100 * "=")):
        print(f"{sep}\n{msg}\n{sep}")


    @contextmanager
    def disable_loss_calculation(self):
        self._do_calc_losses = False
        try:
            yield
        finally:
            self._do_calc_losses = True
