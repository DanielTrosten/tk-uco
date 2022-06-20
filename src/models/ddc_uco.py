import tensorflow as tf
from tensorflow import keras

from tf_lib import backbones, loss
from lib.metrics import calc_metrics


class DDCClusteringModule(keras.layers.Layer):
    def __init__(self, cfg):
        super(DDCClusteringModule, self).__init__()

        self.hidden_layers = [keras.layers.Dense(units=cfg.n_hidden, activation="relu")]
        if cfg.use_bn:
            trainable = getattr(cfg, "bn_trainable_params", True)
            self.hidden_layers.append(keras.layers.BatchNormalization(center=trainable, scale=trainable))

        self.output_layer = keras.layers.Dense(units=cfg.n_clusters, activation="softmax")

    def call(self, x):
        for layer in self.hidden_layers:
            x = layer(x)
        output = self.output_layer(x)
        return output, x


class DDCUCO(keras.Model):
    def __init__(self, cfg):
        super(DDCUCO, self).__init__()

        self.cfg = cfg
        self.loss_values = self.outputs = self.hidden = self.backbone_output = None
        self.training = False

        self.backbone = backbones.create_backbone(cfg.backbone_config)
        self.clustering_module = DDCClusteringModule(cfg.cm_config)
        self.loss_layer = loss.Loss(cfg.loss_config)

    def call(self, inputs, training=False):
        if isinstance(inputs, dict):
            inputs = inputs["data"]

        self.training = training
        self.backbone_output = self.backbone(inputs)
        self.outputs, self.hidden = self.clustering_module(self.backbone_output)
        self.loss_values = self.loss_layer(self)
        self._update_losses_and_metrics()
        return self.outputs

    def _update_losses_and_metrics(self):
        for key, value in self.loss_values.items():
            self.add_loss(value)
            self.add_metric(value, name=key)

        for loss_term in self.loss_layer.terms:
            if isinstance(loss_term, loss.DDC1):
                self.add_metric(loss_term.sigma, name="sigma_hidden")
            elif isinstance(loss_term, loss.UCO):
                # for i, s in enumerate(loss_term.sigmas):
                for i in range(loss_term.n_outputs):
                    self.add_metric(loss_term.sigmas[i], name=f"sigma_uco_{i}")
