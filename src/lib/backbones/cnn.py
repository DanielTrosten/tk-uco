import torch.nn as nn
import numpy as np
from copy import deepcopy

import helpers
from lib.backbones.backbone import Backbone


class CNN(Backbone):
    def __init__(self, cfg, input_size=None, flatten_output=True, **_):
        """
        CNN backbone

        :param cfg: CNN config
        :type cfg: config.defaults.CNN
        :param flatten_output: Flatten the backbone output?
        :type flatten_output: bool
        :param _:
        :type _:
        """
        super().__init__()

        if input_size is not None:
            self.output_size = deepcopy(input_size)
        else:
            self.output_size = list(cfg.input_size)

        for layer_type, *layer_params in cfg.layers:
            layer_func = getattr(self, f"_{layer_type}", None)
            if layer_func is None:
                raise RuntimeError(f"Unknown layer type: {layer_type}")
            layer_func(cfg, layer_params)

        self.output_size_before_flatten = self.output_size
        if flatten_output:
            self.layers.append(nn.Flatten())
            self.output_size = [np.prod(self.output_size)]

    def _conv(self, cfg, layer_params):
        pad = self.get_padding(cfg.padding, layer_params[:2])
        self.layers.append(nn.Conv2d(in_channels=self.output_size[0], out_channels=layer_params[2],
                                     kernel_size=layer_params[:2], padding=pad))
        # Update output size
        self.output_size[0] = layer_params[2]
        self.output_size[1:] = helpers.conv2d_output_shape(self.output_size[1:], kernel_size=layer_params[:2],
                                                           pad=pad)
        # Add activation
        if layer_params[3] == "relu":
            self.layers.append(nn.ReLU())

    def _tconv(self, cfg, layer_params):
        pad = self.get_padding(cfg.padding, layer_params[:2])
        self.layers.append(nn.ConvTranspose2d(in_channels=self.output_size[0], out_channels=layer_params[2],
                                              kernel_size=layer_params[:2], padding=pad))
        # Update output size
        self.output_size[0] = layer_params[2]
        self.output_size[1:] = helpers.conv_transpose_2d_output_shape(self.output_size[1:],
                                                                      kernel_size=layer_params[:2], pad=pad)
        # Add activation
        if layer_params[3] == "relu":
            self.layers.append(nn.ReLU())

    def _pool(self, cfg, layer_params):
        self.layers.append(nn.MaxPool2d(kernel_size=layer_params))
        # Update output size
        self.output_size[1:] = helpers.conv2d_output_shape(self.output_size[1:], kernel_size=layer_params,
                                                           stride=layer_params)

    def _upsample(self, cfg, layer_params):
        self.layers.append(nn.Upsample(scale_factor=layer_params[0]))
        self.output_size[1] *= layer_params[0]
        self.output_size[2] *= layer_params[0]

    def _relu(self, cfg, layer_params):
        self.layers.append(nn.ReLU())

    def _lrelu(self, cfg, layer_params):
        self.layers.append(nn.LeakyReLU(layer_params[0]))

    def _bn(self, cfg, layer_params):
        if len(self.output_size) > 1:
            self.layers.append(nn.BatchNorm2d(num_features=self.output_size[0]))
        else:
            self.layers.append(nn.BatchNorm1d(num_features=self.output_size[0]))

    def _fc(self, cfg, layer_params):
        self.layers.append(nn.Flatten())
        self.output_size = [np.prod(self.output_size)]
        self.layers.append(nn.Linear(self.output_size[0], layer_params[0], bias=True))
        self.output_size = [layer_params[0]]

    def _out(self, cfg, layer_params):
        self.layers.append(self.intermediate_outputs_container)

    @staticmethod
    def get_padding(pad_mode, ksize):
        if pad_mode == "valid":
            return 0, 0
        if pad_mode == "same":
            return ksize[0] // 2, ksize[1] // 2
        raise RuntimeError(f"Unknown padding mode: {pad_mode}.")