import numpy as np
import tensorflow as tf
from tensorflow import keras

import config
from tf_lib import kernel

EPSILON = 1E-9
DEBUG_MODE = False


def triu(X, name=None):
    # Sum of strictly upper triangular part
    return tf.reduce_sum(tf.linalg.band_part(X, 0, -1) - tf.linalg.band_part(X, 0, 0), name=name)


def _atleast_epsilon(X, eps=EPSILON):
    """
    Ensure that all elements are >= `eps`.

    :param X: Input elements
    :type X: tf.Tensor
    :param eps: epsilon
    :type eps: float
    :return: New version of X where elements smaller than `eps` have been replaced with `eps`.
    :rtype: tf.Tensor
    """
    return tf.where(X < eps, eps, X)


def d_cs(A, K, n_clusters):
    nom = tf.transpose(A) @ K @ A
    dnom_squared = tf.expand_dims(tf.linalg.diag_part(nom), -1) @ tf.expand_dims(tf.linalg.diag_part(nom), 0)

    nom = _atleast_epsilon(nom)
    dnom_squared = _atleast_epsilon(dnom_squared, eps=EPSILON ** 2)

    d = 2 / (n_clusters * (n_clusters - 1)) * triu(nom / tf.sqrt(dnom_squared))
    return d


def kernel_from_distances(dists, sigma):
    return tf.exp((-1 * dists) / (2 * sigma))


# ======================================================================================================================
# Loss terms
# ======================================================================================================================

class LossTerm(keras.layers.Layer):
    # Names of tensors required for the loss computation
    required_tensors = []

    def __init__(self, *args, **kwargs):
        """
        Base class for a term in the loss function.

        :param args:
        :type args:
        :param kwargs:
        :type kwargs:
        """
        super(LossTerm, self).__init__()

    def call(self, net, cfg, extra):
        raise NotImplementedError()


class DDC1(LossTerm):
    """
    L_1 loss from DDC
    """
    required_tensors = ["hidden_distances"]

    def __init__(self, cfg):
        super(DDC1, self).__init__()

        self.sigma = tf.Variable(initial_value=0.1, trainable=False, name="hidden_sigma")
        self.rel_sigma = cfg.rel_sigma
        self.sigma_mode = cfg.sigma_mode
        self.sigma_momentum = tf.constant(cfg.sigma_momentum)

    def update_sigma(self, dists, training):
        new_sigma = self.rel_sigma * tf.stop_gradient(kernel.median(dists))

        if self.sigma_mode == "otf":
            self.sigma.assign(new_sigma)
        elif self.sigma_mode == "running_mean":
            if training:
                self.sigma.assign((1 - self.sigma_momentum) * new_sigma + self.sigma_momentum * self.sigma)
        else:
            raise ValueError(f"Invalid sigma_mode: {self.sigma_mode}")

    def call(self, net, cfg, extra):
        dists = extra["hidden_distances"]
        self.update_sigma(dists, training=net.training)
        ker = kernel_from_distances(dists, self.sigma)
        # Store the kernel in the `extra`-dict since it is required by DDC3.
        extra["hidden_kernel"] = ker
        return d_cs(net.outputs, ker, cfg.n_clusters)


class DDC2(LossTerm):
    """
    L_2 loss from DDC
    """
    def call(self, net, cfg, extra):
        n = net.outputs.shape[0]
        return 2 / (n * (n - 1)) * triu(net.outputs @ tf.transpose(net.outputs))


class DDC3(LossTerm):
    """
    L_3 loss from DDC
    """
    def call(self, net, cfg, extra):
        eye = tf.eye(cfg.n_clusters)
        m = tf.exp(-kernel.cdist(net.outputs, eye))
        return d_cs(m, extra["hidden_kernel"], cfg.n_clusters)


class UCO(LossTerm):
    required_tensors = ["intermediate_distances"]

    def __init__(self, cfg):
        super(UCO, self).__init__()
        self.rel_sigma = cfg.rel_sigma
        # self.sigmas = 0.1 * tf.ones(shape=[cfg.n_outputs], dtype=tf.float32)
        self.sigmas = tf.Variable(initial_value=0.1 * np.ones(cfg.n_outputs), trainable=False, dtype=tf.float32,
                                  name="uco_sigmas")
        self.sigma_mode = cfg.sigma_mode
        self.sigma_momentum = cfg.sigma_momentum
        self.uco_weights = self.get_uco_weights(cfg.uco_weighting_method, cfg.n_outputs)
        self.n_outputs = cfg.n_outputs

    @staticmethod
    def get_uco_weights(weighting_method, n_kernels):
        if weighting_method == "constant":
            w = np.ones(n_kernels)
        elif weighting_method == "linear":
            w = np.arange(1, n_kernels + 1) / n_kernels
        elif weighting_method == "exp":
            w = 10 ** np.arange(n_kernels)
            w = w / np.max(w)
        else:
            raise RuntimeError(f"Unknown UCO weighting method: {weighting_method}.")
        return w

    def update_sigmas(self, dists_list, training):
        # new_sigmas = []
        # for dists in dists_list:
        #     new_sigmas.append(self.rel_sigma * tf.stop_gradient(kernel.median(dists)))
        # self.sigmas = new_sigmas
        # self.set_weights([np.array(new_sigmas)])

        # new_sigmas = tf.zeros(shape=[self.n_outputs])
        # new_sigmas = np.zeros(self.n_outputs)
        new_sigmas = []
        for i, dists in enumerate(dists_list):
            new_sigmas.append(self.rel_sigma * tf.stop_gradient(kernel.median(dists)))
        new_sigmas = tf.stack(new_sigmas, axis=0)

        if self.sigma_mode == "otf":
            self.sigmas.assign(new_sigmas)
        elif self.sigma_mode == "running_mean":
            if training:
                self.sigmas.assign((1 - self.sigma_momentum) * new_sigmas + self.sigma_momentum * self.sigmas)
        else:
            raise ValueError(f"Invalid sigma_mode: {self.sigma_mode}")

    def call(self, net, cfg, extra):
        self.update_sigmas(extra["intermediate_distances"], net.training)

        outputs = net.outputs
        if not cfg.uco_assignment_gradient:
            outputs = tf.stop_gradient(outputs)

        losses = {}
        for i, (w, dists) in enumerate(zip(self.uco_weights, extra["intermediate_distances"])):
            ker = kernel_from_distances(dists, sigma=self.sigmas[i])
            losses[str(i)] = cfg.uco_lambda * w * d_cs(outputs, ker, cfg.n_clusters)
        return losses


class KLLoss(LossTerm):
    def call(self, net, cfg, extra):
        loss = tf.reduce_mean(keras.losses.KLD(tf.stop_gradient(net.targets), net.outputs))
        return loss


class ReconstructionLoss(LossTerm):
    def __init__(self, cfg):
        super(ReconstructionLoss, self).__init__()
        self.mse_loss = keras.losses.MeanSquaredError()
        self.flatten = keras.layers.Flatten()

    def call(self, net, cfg, extra):
        return self.mse_loss(self.flatten(net.inputs), net.reconstruction)


# ======================================================================================================================
# Extra functions
# ======================================================================================================================
def hidden_distances(net, cfg):
    return kernel.pdist(net.hidden)


def intermediate_distances(net, cfg):
    dists = []
    for x in net.backbone.intermediate_outputs:
        if cfg.uco_kernel_type == "naive":
            if tf.rank(x) > 2:
                x = keras.layers.Flatten()(x)
            ker = kernel.pdist(x)
        elif cfg.uco_kernel_type == "tensor":
            ker = kernel.batch_tensor_dist_matrix(x)
        else:
            raise RuntimeError(f"Unknown kernel type: {cfg.kernel_type}")

        dists.append(ker)
    return dists


# ======================================================================================================================
# Loss class
# ======================================================================================================================

class Loss(keras.layers.Layer):
    # Possible terms to include in the loss
    TERM_CLASSES = {
        "ddc_1": DDC1,
        "ddc_2": DDC2,
        "ddc_3": DDC3,
        "uco": UCO,
        "kl": KLLoss,
        "reconstruct": ReconstructionLoss,
    }
    # Functions to compute the required tensors for the terms.
    EXTRA_FUNCS = {
        "hidden_distances": hidden_distances,
        "intermediate_distances": intermediate_distances
    }

    def __init__(self, cfg):
        """
        Implementation of a general loss function

        :param cfg: Loss function config
        :type cfg: config.defaults.Loss
        """
        super().__init__()
        self.cfg = cfg

        self.names = cfg.funcs.split("|")
        self.weights_ = cfg.weights if cfg.weights is not None else len(self.names) * [1]

        self.terms = []
        for term_name in self.names:
            if term_name:
                self.terms.append(self.TERM_CLASSES[term_name](cfg))

        self.required_extras_names = list(set(sum([t.required_tensors for t in self.terms], [])))

    def call(self, net):
        extra = {name: self.EXTRA_FUNCS[name](net, self.cfg) for name in self.required_extras_names}
        losses = {}
        for name, term, weight in zip(self.names, self.terms, self.weights_):
            value = term(net, self.cfg, extra)
            # If we got a dict, add each term from the dict with "name/" as the scope.
            if isinstance(value, dict):
                for key, _value in value.items():
                    losses[f"{name}/{key}"] = weight * _value
            # Otherwise, just add the value to the dict directly
            else:
                losses[name] = weight * value

        return losses