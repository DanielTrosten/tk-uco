import os
import numpy as np
import numba as nb
import torch as th

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
import tensorflow as tf

import helpers


def batch_fill_diagonal_(a, v):
    assert a.ndim == 3
    assert v.ndim == 2

    n, d1, d2 = a.size()
    assert d1 == d2
    assert v.size() == (n, d1)

    idx = range(d1)
    a[:, idx, idx] = v


def batch_from_diagonal(d):
    assert d.ndim == 2
    b = th.eye(d.size(1)).type_as(d)
    c = d.unsqueeze(2).expand(*d.size(), d.size(1))
    result = c * b
    return result


@nb.guvectorize([(nb.float32[:, :], nb.float32[:], nb.float32[:, :])], "(n, n)->(n),(n,n)", nopython=True)
def batched_eigh(a, vals, vecs):
    vals[:], vecs[:, :] = np.linalg.eigh(a)


def tf_eig(a):
    t = tf.constant(helpers.npy(a))
    vals, vecs = tf.linalg.eigh(t)
    return th.tensor(vals.numpy()).type_as(a), th.tensor(vecs.numpy()).type_as(a)


class SymEigCg(th.autograd.Function):
    """
    Symmetric eigendecomposition with custom gradient.

    From: /usr/local/Cellar/python/3.6.2/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages/tensorflow/python/ops/linalg_grad.py
    """
    @staticmethod
    def forward(ctx, a):
        vals, vecs = th.symeig(a, eigenvectors=True, upper=False)
        # vals, vecs = tf_eig(a)
        ctx.save_for_backward(vals, vecs)
        return vals, vecs

    @staticmethod
    def backward(ctx, grad_e, grad_v):
        e, v = ctx.saved_tensors

        eps = 1e-6
        denom = e[..., None, :] - e[..., :, None]
        f = th.where(th.abs(denom) >= eps, 1 / denom, th.zeros_like(denom).type_as(denom))
        batch_fill_diagonal_(f, th.zeros(f.size()[:2]).type_as(f))

        vT = v.permute(0, 2, 1)
        grad_a = v @ (batch_from_diagonal(grad_e) + f * (vT @ grad_v)) @ vT
        grad_a = (grad_a + grad_a.permute(0, 2, 1)) / 2
        return grad_a


symeig_cg = SymEigCg.apply


if __name__ == '__main__':

    ta = np.array([[
        [14, 10, 10],
        [10, 25,  7],
        [10,  7, 25]
    ]]).astype(np.float32)

    t = th.Tensor(ta)
    t.requires_grad = True

    vals, vecs = symeig_cg(t)
    # vals, vecs = th.symeig(t, eigenvectors=True, upper=False)
    l = th.sum(vals) + th.sum(vecs)
    l.backward()
    print(t.grad)
