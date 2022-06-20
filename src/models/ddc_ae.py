import numpy as np

import helpers
from lib.loss import Loss
from lib.backbones import create_backbone
from models.model_base import ModelBase
from models.clustering_module import DDC


class DDCAE(ModelBase):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.cfg = cfg
        self.input = self.backbone_output = self.output = self.hidden = self.reconstruction = None
        self.backbone = create_backbone(cfg.backbone_config)

        self.decoder_input_size = self.backbone.output_size_before_flatten
        self.decoder = create_backbone(cfg.decoder_config, input_size=self.decoder_input_size, flatten_output=False)

        self.ddc_input_size = np.prod(self.backbone.output_size)
        self.ddc = DDC([self.ddc_input_size], cfg.cm_config)

        self.loss = Loss(cfg.loss_config)

        # Initialize weights.
        self.apply(helpers.he_init_weights)

    def forward(self, x, idx=None):
        self.input = x
        self.backbone_output = self.backbone(x).view(-1, self.ddc_input_size)

        decoder_input = self.backbone_output.view(-1, *self.decoder_input_size)
        self.reconstruction = self.decoder(decoder_input)

        self.output, self.hidden = self.ddc(self.backbone_output)
        return self.output
