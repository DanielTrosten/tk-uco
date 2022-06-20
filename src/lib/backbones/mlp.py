import numpy as np
import torch.nn as nn

import helpers
from lib.backbones.backbone import Backbone


class MLP(Backbone):
    def __init__(self, cfg, input_size=None, **_):
        super().__init__()
        self.output_size = self.create_linear_layers(cfg, self.layers, input_size=input_size,
                                                     intermediate_outputs_container=self.intermediate_outputs_container)

    @staticmethod
    def get_activation_module(a):
        if a == "relu":
            return nn.ReLU()
        elif a == "sigmoid":
            return nn.Sigmoid()
        elif a == "tanh":
            return nn.Tanh()
        elif a == "softmax":
            return nn.Softmax(dim=1)
        elif a.startswith("leaky_relu"):
            neg_slope = float(a.split(":")[1])
            return nn.LeakyReLU(neg_slope)
        else:
            raise RuntimeError(f"Invalid MLP activation: {a}.")

    @classmethod
    def create_linear_layers(cls, cfg, layer_container, input_size=None, intermediate_outputs_container=None):
        # `input_size` takes priority over `cfg.input_size`
        if input_size is not None:
            output_size = list(input_size)
        else:
            output_size = list(cfg.input_size)

        if len(output_size) > 1:
            layer_container.append(nn.Flatten())
            output_size = [np.prod(output_size)]

        n_layers = len(cfg.layers)
        activations = helpers.ensure_iterable(cfg.activation, expected_length=n_layers)
        use_bias = helpers.ensure_iterable(cfg.use_bias, expected_length=n_layers)
        use_bn = helpers.ensure_iterable(cfg.use_bn, expected_length=n_layers)

        for n_units, act, _use_bias, _use_bn in zip(cfg.layers, activations, use_bias, use_bn):
            # If we get n_units = -1, then the number of units should be the same as the previous number of units, or
            # the input dim.
            if n_units == -1:
                n_units = output_size[0]
            if n_units == "out" and intermediate_outputs_container is not None:
                layer_container.append(intermediate_outputs_container)
            else:
                layer_container.append(nn.Linear(in_features=output_size[0], out_features=n_units, bias=_use_bias))
                if _use_bn:
                    # Add BN before activation
                    layer_container.append(nn.BatchNorm1d(num_features=n_units))
                if act is not None:
                    # Add activation
                    layer_container.append(cls.get_activation_module(act))
                output_size[0] = n_units

        return output_size
