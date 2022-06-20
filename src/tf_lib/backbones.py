import numpy as np
from tensorflow import keras
from copy import deepcopy

import helpers


class IntermediateOutputsContainer(keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.outputs = []

    def flush(self):
        self.outputs = []

    def call(self, x):
        self.outputs.append(x)
        return x


class Backbone(keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.intermediate_outputs_container = IntermediateOutputsContainer()
        self.layers_ = []

    @property
    def intermediate_outputs(self):
        return self.intermediate_outputs_container.outputs

    def call(self, x):
        self.intermediate_outputs_container.flush()
        for layer in self.layers_:
            x = layer(x)
        return x


class CNN(Backbone):
    def __init__(self, cfg, input_size=None, flatten_output=True, **_):
        super().__init__()

        if input_size is not None:
            self.output_size = deepcopy(input_size)
        else:
            s = list(cfg.input_size)
            self.output_size = [*(s[1:]), s[0]]

        self.layer_is_first = True
        for i, (layer_type, *layer_params) in enumerate(cfg.layers):
            layer_func = getattr(self, f"_{layer_type}", None)
            if layer_func is None:
                raise RuntimeError(f"Unknown layer type: {layer_type}")
            layer_func(cfg, layer_params)
            self.layer_is_first = False

        self.output_size_before_flatten = self.output_size
        if flatten_output:
            self.layers_.append(keras.layers.Flatten())
            self.output_size = [np.prod(self.output_size)]

    def _conv(self, cfg, layer_params):
        kwargs = dict(filters=layer_params[2], kernel_size=layer_params[:2])
        if self.layer_is_first:
            kwargs["input_shape"] = list(self.output_size)
        self.layers_.append(keras.layers.Conv2D(**kwargs))

        # Update output size
        self.output_size[2] = layer_params[2]
        self.output_size[:2] = helpers.conv2d_output_shape(self.output_size[:2], kernel_size=layer_params[:2])
        # Add activation
        if layer_params[3] == "relu":
            self.layers_.append(keras.layers.ReLU())

    def _pool(self, cfg, layer_params):
        self.layers_.append(keras.layers.MaxPool2D(pool_size=layer_params))
        # Update output size
        self.output_size[:2] = helpers.conv2d_output_shape(self.output_size[:2], kernel_size=layer_params,
                                                           stride=layer_params)

    def _relu(self, cfg, layer_params):
        self.layers_.append(keras.layers.ReLU())

    def _lrelu(self, cfg, layer_params):
        self.layers_.append(keras.layers.LeakyReLU(layer_params[0]))

    def _bn(self, cfg, layer_params):
        self.layers_.append(keras.layers.BatchNormalization(axis=-1))

    def _fc(self, cfg, layer_params):
        self.layers_.append(keras.layers.Flatten())
        self.output_size = [np.prod(self.output_size)]
        self.layers_.append(keras.layers.Dense(units=layer_params[0]))
        self.output_size = [layer_params[0]]

    def _upsample(self, cfg, layer_params):
        self.layers_.append(keras.layers.UpSampling2D(size=(layer_params[0], layer_params[0])))

    def _tconv(self, cfg, layer_params):
        *ksize, filters, activation = layer_params
        self.layers_.append(keras.layers.Conv2DTranspose(filters=filters, kernel_size=ksize, activation=activation))

    def _out(self, cfg, layer_params):
        self.layers_.append(self.intermediate_outputs_container)


class MLP(Backbone):
    def __init__(self, cfg, input_size=None, **_):
        super(MLP, self).__init__()

        self.output_size = self.create_linear_layers(cfg, self.layers_, input_size=input_size,
                                                     intermediate_outputs_container=self.intermediate_outputs_container)

    @classmethod
    def create_linear_layers(cls, cfg, layer_container, input_size=None, intermediate_outputs_container=None):
        # `input_size` takes priority over `cfg.input_size`
        if input_size is not None:
            output_size = list(input_size)
        else:
            output_size = list(cfg.input_size)

        if len(output_size) > 1:
            layer_container.append(keras.layers.Flatten())
            print(output_size)
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
                layer_container.append(keras.layers.Dense(units=n_units, activation=act, use_bias=_use_bias))
                if _use_bn:
                    # Add BN before activation
                    layer_container.append(keras.layers.BatchNormalization())
                output_size[0] = n_units
        return output_size


BACKBONE_CONSTRUCTORS = {
        "CNN": CNN,
        "MLP": MLP,
}


def create_backbone(cfg, input_size=None, flatten_output=True):
    if cfg.class_name not in BACKBONE_CONSTRUCTORS:
        raise RuntimeError(f"Invalid backbone: '{cfg.class_name}'")
    return BACKBONE_CONSTRUCTORS[cfg.class_name](cfg, input_size=input_size, flatten_output=flatten_output)
