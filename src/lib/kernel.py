import numpy as np
import torch as th
from torch.nn.functional import relu

import config
import helpers
from lib.eig import symeig_cg

EPSILON = 1E-9


def kernel_from_distance_matrix(dist, rel_sigma, min_sigma=EPSILON):
    # `dist` can sometimes contain negative values due to floating point errors, so just set these to zero.
    dist = relu(dist)
    sigma2 = rel_sigma * th.median(dist)
    # Disable gradient for sigma
    sigma2 = sigma2.detach()
    sigma2 = th.where(sigma2 < min_sigma, sigma2.new_tensor(min_sigma), sigma2)
    k = th.exp(- dist / (2 * sigma2))
    return k


def vector_kernel(x, rel_sigma=0.15):
    return kernel_from_distance_matrix(cdist(x, x), rel_sigma)


def tensor_kernel(x, rel_sigma=0.15):
    return kernel_from_distance_matrix(batch_tensor_dist_matrix(x), rel_sigma)


def auto_kernel(x, rel_sigma=0.15):
    return vector_kernel(x, rel_sigma) if x.ndim == 2 else tensor_kernel(x, rel_sigma)


def cdist(X, Y):
    xyT = X @ th.t(Y)
    x2 = th.sum(X**2, dim=1, keepdim=True)
    y2 = th.sum(Y**2, dim=1, keepdim=True)
    d = x2 - 2 * xyT + th.t(y2)
    return d


def batch_matrizize(X, i):
    """
    Matricize a batch of tensors along dimension i. Matricization along the batch-axis (0) is not allowed.
    :param X: Batch of input tensors. Shape (batch_size, D_1, ..., D_r).
    :type X: tf.Tensor
    :param i: Matricization dimension
    :type i: int
    :return: Batch of matricizized tensors
    :rtype: tf.Tensor
    """
    assert i != 0, "Attempting to batch-matrizize along batch-axis. Are you sure about this?"
    # x = th.cat(th.unbind(X, dim=i), dim=1)
    if i != 1:
        permute_axes = list(range(X.dim()))
        permute_axes[i] = 1
        permute_axes[1] = i
        X = X.permute(permute_axes)

    size = X.size()
    X = X.reshape(size[0], size[1], int(np.prod(size[2:])))

    if X.size(1) > X.size(2):
        X = X.permute(0, 2, 1)

    return X


def projection_dist(v):
    """
    Pairwise projection distance between the elements specified by the first axis of 'v'. Each element of 'v' (v[0],
    v[1], ...) is assumed to be a matrix with orthonormal rows.

    :param v: Input matricies
    :type v: tf.Tensor
    :return: Projection-distance matrix between the elements of 'v'.
    :rtype: tf.Tensor
    """
    z = th.einsum("ali,blj->abij", v, v)
    zTz = th.matmul(z.permute(0, 1, 3, 2), z)
    D_i = 2 * v.size()[2] - th.einsum("abii->ab", zTz)
    return D_i


def orthogonalize_eig(x):
    """
    Orthogonalize the matricizations in x using eigendecomposition.

    :param x: Input matrices
    :type x: tf.Tensor
    :return: Orthonormal matrices
    :rtype: tf.Tensor
    """
    xT = x.permute(0, 2, 1)
    xxT = x @ xT

    # vals, vecs = th.symeig(xxT, eigenvectors=True)
    vals, vecs = symeig_cg(xxT)

    # xxT = xxT.cpu()
    # vals, vecs = symeig_cg(xxT)
    # vals = vals.type_as(x)
    # vecs = vecs.type_as(x)

    vecs = xT @ vecs
    norms = th.sqrt(relu(vals[:, None, :]) + EPSILON)
    vecs = vecs / norms
    return vecs


def orthogonalize_svd(x):
    # *_, vT = th.svd(x, some=True, compute_uv=True)
    *_, vT = th.svd_lowrank(x, q=x.size(1))
    v = vT.permute(0, 2, 1)
    return v


def orthogonalize_lobpcg(x):
    xT = x.permute(0, 2, 1)
    xTx = xT @ x
    vals, vecs = th.lobpcg(xTx, k=x.size(1), method="ortho")

    vecs = vecs.permute(0, 2, 1)
    # print(x.size())
    # print(vecs.size())
    return vecs


def batch_tensor_dist_matrix(X):
    """
    Compute the projection distance matrix for a batch of input tensors.

    :param X: Input tensors
    :type X: tf.Tensor
    :return: Distance matrix
    :rtype: tf.Tensor
    """
    input_size = X.size()
    D = th.zeros((input_size[0], input_size[0])).type_as(X)
    for i in range(1, len(input_size)):
        x = batch_matrizize(X, i)

        v = orthogonalize_eig(x)
        # v = orthogonalize_svd(x)
        # v = orthogonalize_lobpcg(x)

        D += relu(projection_dist(v))
    return D
