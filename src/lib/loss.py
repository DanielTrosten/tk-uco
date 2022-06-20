import numpy as np
import torch as th
import torch.nn as nn

import config
from lib import kernel

EPSILON = 1E-9
DEBUG_MODE = False


def triu(X):
    # Sum of strictly upper triangular part
    return th.sum(th.triu(X, diagonal=1))


def _atleast_epsilon(X, eps=EPSILON):
    """
    Ensure that all elements are >= `eps`.

    :param X: Input elements
    :type X: th.Tensor
    :param eps: epsilon
    :type eps: float
    :return: New version of X where elements smaller than `eps` have been replaced with `eps`.
    :rtype: th.Tensor
    """
    return th.where(X < eps, X.new_tensor(eps), X)


def d_cs(A, K, n_clusters):
    """
    Cauchy-Schwarz divergence.

    :param A: Cluster assignment matrix
    :type A:  th.Tensor
    :param K: Kernel matrix
    :type K: th.Tensor
    :param n_clusters: Number of clusters
    :type n_clusters: int
    :return: CS-divergence
    :rtype: th.Tensor
    """
    nom = th.t(A) @ K @ A
    dnom_squared = th.unsqueeze(th.diagonal(nom), -1) @ th.unsqueeze(th.diagonal(nom), 0)

    nom = _atleast_epsilon(nom)
    dnom_squared = _atleast_epsilon(dnom_squared, eps=EPSILON**2)

    d = 2 / (n_clusters * (n_clusters - 1)) * triu(nom / th.sqrt(dnom_squared))
    return d


# ======================================================================================================================
# Loss terms
# ======================================================================================================================

class LossTerm(nn.Module):
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

    def forward(self, net, cfg, extra):
        raise NotImplementedError()


class DDC1(LossTerm):
    """
    L_1 loss from DDC
    """
    required_tensors = ["hidden_kernel"]

    def forward(self, net, cfg, extra):
        return d_cs(net.output, extra["hidden_kernel"], cfg.n_clusters)


class DDC2(LossTerm):
    """
    L_2 loss from DDC
    """
    def forward(self, net, cfg, extra):
        n = net.output.size(0)
        return 2 / (n * (n - 1)) * triu(net.output @ th.t(net.output))


class DDC2Flipped(LossTerm):
    """
    Flipped version of the L_2 loss from DDC. Used by EAMC
    """

    def forward(self, net, cfg, extra):
        return 2 / (cfg.n_clusters * (cfg.n_clusters - 1)) * triu(th.t(net.output) @ net.output)


class DDC3(LossTerm):
    """
    L_3 loss from DDC
    """
    required_tensors = ["hidden_kernel"]

    # def __init__(self, cfg):
    #     super().__init__()
        # self.eye = th.eye(cfg.n_clusters, device=config.DEVICE)

    def forward(self, net, cfg, extra):
        eye = th.eye(cfg.n_clusters).type_as(net.output)
        m = th.exp(-kernel.cdist(net.output, eye))
        return d_cs(m, extra["hidden_kernel"], cfg.n_clusters)


class UCO(LossTerm):
    required_tensors = ["intermediate_kernels"]

    @staticmethod
    def get_weights(weighting_method, n_kernels):
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

    def forward(self, net, cfg, extra):
        weights = self.get_weights(cfg.uco_weighting_method, len(extra["intermediate_kernels"]))

        assignments = net.output
        if not cfg.uco_assignment_gradient:
            assignments = assignments.detach()

        losses = {}
        for i, (w, ker) in enumerate(zip(weights, extra["intermediate_kernels"])):
            losses[str(i)] = cfg.uco_lambda * w * d_cs(assignments, ker, cfg.n_clusters)
        return losses


class Reconstruction(LossTerm):
    def forward(self, net, cfg, extra):
        return nn.functional.mse_loss(input=net.reconstruction, target=net.input)


class Contrastive(LossTerm):
    large_num = 1e9

    @staticmethod
    def _norm(mat):
        return th.nn.functional.normalize(mat, p=2, dim=1)

    @classmethod
    def _normalized_projections(cls, net):
        n = net.projections.size(0) // 2
        h1, h2 = net.projections[:n], net.projections[n:]
        h2 = cls._norm(h2)
        h1 = cls._norm(h1)
        return n, h1, h2

    def forward(self, net, cfg, extra):
        # Adapted from: https://github.com/google-research/simclr/blob/master/objective.py
        n, h1, h2 = self._normalized_projections(net)

        labels = th.arange(0, n, device=config.DEVICE, dtype=th.long)
        masks = th.eye(n, device=config.DEVICE)

        logits_aa = ((h1 @ h1.t()) / cfg.tau) - masks * self.large_num
        logits_bb = ((h2 @ h2.t()) / cfg.tau) - masks * self.large_num

        logits_ab = (h1 @ h2.t()) / cfg.tau
        logits_ba = (h2 @ h1.t()) / cfg.tau

        loss_a = th.nn.functional.cross_entropy(th.cat((logits_ab, logits_aa), dim=1), labels)
        loss_b = th.nn.functional.cross_entropy(th.cat((logits_ba, logits_bb), dim=1), labels)
        loss = (loss_a + loss_b)
        return cfg.delta * loss


# ======================================================================================================================
# Extra functions
# ======================================================================================================================

def hidden_kernel(net, cfg):
    return kernel.vector_kernel(net.hidden, cfg.rel_sigma)


def intermediate_kernels(net, cfg):
    kernels = []
    for x in net.backbone.intermediate_outputs:
        if cfg.uco_kernel_type == "naive":
            x = th.flatten(x, start_dim=1)
            ker = kernel.vector_kernel(x, rel_sigma=cfg.rel_sigma)
        elif cfg.uco_kernel_type == "tensor":
            ker = kernel.tensor_kernel(x, rel_sigma=cfg.rel_sigma)
        else:
            raise RuntimeError(f"Unknown kernel type: {cfg.kernel_type}")

        kernels.append(ker)
    return kernels


# ======================================================================================================================
# Loss class
# ======================================================================================================================

class Loss(nn.Module):
    # Possible terms to include in the loss
    TERM_CLASSES = {
        "ddc_1": DDC1,
        "ddc_2": DDC2,
        "ddc_2_flipped": DDC2Flipped,
        "ddc_3": DDC3,
        "uco": UCO,
        "reconstruct": Reconstruction,
        "contrast": Contrastive
    }
    # Functions to compute the required tensors for the terms.
    EXTRA_FUNCS = {
        "hidden_kernel": hidden_kernel,
        "intermediate_kernels": intermediate_kernels
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
        self.weights = cfg.weights if cfg.weights is not None else len(self.names) * [1]

        self.terms = nn.ModuleList()
        for term_name in self.names:
            self.terms.append(self.TERM_CLASSES[term_name](cfg))

        self.required_extras_names = list(set(sum([t.required_tensors for t in self.terms], [])))

    def forward(self, net, ignore_in_total=tuple()):
        extra = {name: self.EXTRA_FUNCS[name](net, self.cfg) for name in self.required_extras_names}
        loss_values = {}
        for name, term, weight in zip(self.names, self.terms, self.weights):
            value = term(net, self.cfg, extra)
            # If we got a dict, add each term from the dict with "name/" as the scope.
            if isinstance(value, dict):
                for key, _value in value.items():
                    loss_values[f"{name}/{key}"] = weight * _value
            # Otherwise, just add the value to the dict directly
            else:
                loss_values[name] = weight * value

        loss_values["tot"] = sum([loss_values[k] for k in loss_values.keys() if k not in ignore_in_total])
        return loss_values

