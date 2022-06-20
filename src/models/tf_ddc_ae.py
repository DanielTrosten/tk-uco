import tensorflow as tf
from tensorflow import keras

from tf_lib import backbones, loss
from models.ddc_uco import DDCClusteringModule


class TFDDCAE(keras.Model):
    def __init__(self, cfg):
        super(TFDDCAE, self).__init__()

        self.cfg = cfg
        self.inputs = self.loss_values = self.outputs = self.hidden = self.backbone_output = self.reconstruction = None
        self.training = False

        self.backbone = backbones.create_backbone(cfg.backbone_config, flatten_output=False)
        self.clustering_module = DDCClusteringModule(cfg.cm_config)
        self.flatten = keras.layers.Flatten()
        self.decoder = backbones.create_backbone(cfg.decoder_config, input_size=self.backbone.output_size)
        self.loss_layer = loss.Loss(cfg.loss_config)

    def call(self, inputs, training=False):
        if isinstance(inputs, dict):
            inputs = inputs["data"]
        self.training = training

        self.inputs = inputs
        self.backbone_output = self.backbone(inputs)
        self.outputs, self.hidden = self.clustering_module(self.flatten(self.backbone_output))
        self.reconstruction = self.decoder(self.backbone_output)
        self.loss_values = self.loss_layer(self)
        self._update_losses_and_metrics()
        return self.outputs

    def _update_losses_and_metrics(self):
        for key, value in self.loss_values.items():
            self.add_loss(value)
            self.add_metric(value, name=key)

    # def test_step(self, data):
    #     inputs, labels = data
    #
    #     outputs = self(inputs, training=False)
    #
    #     logs = {m.name: m.result() for m in self.metrics}
    #     # mtc = calc_metrics(labels, outputs.argmax(axis=1))
    #     # logs.update(**mtc)
    #     # print(logs)
    #     return logs
