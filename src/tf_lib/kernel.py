"""
Kernel matrix computations
"""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from tf_lib.eig import _eig_cg


@tf.custom_gradient
def pdist(X):
    """
    Pairwise squared-euclidean distance matrix between the rows of X

    :param X: Input tensor
    :type X: tf.Tensor
    :return: Distance matrix
    :rtype: tf.Tensor
    """
    xxT = tf.matmul(X, X, transpose_b=True)
    x2 = tf.reduce_sum(X**2, axis=1, keepdims=True)
    d = x2 - 2 * xxT + tf.transpose(x2)

    def grad(dy):
        g1 = X * tf.reduce_sum(dy, axis=1, keepdims=True)
        g2 = tf.matmul(dy, X)
        return 4 * (g1 - g2)

    return d, grad


@tf.custom_gradient
def cdist(X, Y):
    """
    Pairwise squared-euclidean distance matrix between the rows of X and the rows of Y

    :param X: Input tensor 1
    :type X: tf.Tensor
    :param Y: Input tensor 2
    :type Y: tf.Tensor
    :return: Distance matrix
    :rtype: tf.Tensor
    """
    xyT = tf.matmul(X, Y, transpose_b=True)
    x2 = tf.reduce_sum(X**2, axis=1, keepdims=True)
    y2 = tf.reduce_sum(Y**2, axis=1, keepdims=True)
    d = x2 - 2 * xyT + tf.transpose(y2)

    def grad(dy):
        gx1 = X * tf.reduce_sum(dy, axis=1, keepdims=True)
        gx2 = tf.matmul(dy, Y)
        gx = 2 * (gx1 - gx2)

        gy1 = Y * tf.transpose(tf.reduce_sum(dy, axis=0, keepdims=True))
        gy2 = tf.matmul(dy, X, transpose_a=True)
        gy = 2 * (gy1 - gy2)
        return [gx, gy]

    return d, grad


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
    x = tf.stack(tf.unstack(X, axis=i), axis=1)
    x = tf.reshape(x, (tf.shape(x)[0], x.get_shape()[1], np.prod(x.get_shape()[2:])))
    return x


def projection_dist(v):
    """
    Pairwise projection distance between the elements specified by the first axis of 'v'. Each element of 'v' (v[0],
    v[1], ...) is assumed to be a matrix with orthonormal rows.

    :param v: Input matricies
    :type v: tf.Tensor
    :return: Projection-distance matrix between the elements of 'v'.
    :rtype: tf.Tensor
    """
    z = tf.einsum("ali,blj->abij", v, v)
    zTz = tf.matmul(tf.transpose(z, (0, 1, 3, 2)), z)
    D_i = 2 * (tf.cast(v.get_shape()[2], tf.float32) - tf.linalg.trace(zTz))
    return D_i


def orthogonalize_eig(x):
    """
    Orthogonalize the matricizations in x using eigendecomposition.

    :param x: Input matrices
    :type x: tf.Tensor
    :return: Orthonormal matrices
    :rtype: tf.Tensor
    """
    xxT = tf.matmul(x, x, transpose_b=True)

    vals, vecs = _eig_cg(xxT)
    eps = 1e-6
    vecs = tf.matmul(x, vecs, transpose_a=True)
    norms = tf.sqrt(tf.nn.relu(vals[:, None, :]) + eps)
    vecs = vecs/norms
    return vecs


def batch_tensor_dist_matrix(X):
    """
    Compute the projection distance matrix for a batch of input tensors.

    :param X: Input tensors
    :type X: tf.Tensor
    :return: Distance matrix
    :rtype: tf.Tensor
    """
    ndim = tf.shape(X).shape[0]
    ds = []
    for i in range(1, ndim):
        x = batch_matrizize(X, i)
        v = orthogonalize_eig(x)
        ds.append(projection_dist(v))

    ds = tf.stack(ds, axis=0)
    ds = tf.nn.relu(ds)
    D = tf.reduce_sum(ds, axis=0, name="D")
    return D


def median(x):
    return tfp.stats.percentile(tf.reshape(x, (-1,)), 50.0, interpolation='midpoint')


def _kernel_from_dist(D, rel_sigma):
    """
    Compute the Gaussian kernel matrix from a distance matrix.

    :param D: Distance matrix
    :type D: tf.Tensor
    :param rel_sigma: Scaling factor for the sigma hyperparameter.
    :type rel_sigma: float
    :return: Kernel matrix
    :rtype: tf.Tensor
    """
    sigma_squared = rel_sigma * tf.stop_gradient(median(D))
    K = tf.exp(-D/sigma_squared)
    return K


def tensor_kernel(X, rel_sigma):
    """
    Get the tensor kernel for a batch of tensors.

    :param X: Input tensors
    :type X: tf.Tensor
    :param rel_sigma: Scaling factor for the sigma hyperparameter.
    :type rel_sigma: float
    :return: Tensor kernel matrix
    :rtype: tf.Tensor
    """
    D = batch_tensor_dist_matrix(X)
    return _kernel_from_dist(D, rel_sigma)


def vector_kernel(X, rel_sigma):
    """
    Get the vector kernel for a batch of vectors.

    :param X: Input vectors
    :type X: tf.Tensor
    :param rel_sigma: Scaling factor for the sigma hyperparameter.
    :type rel_sigma: float
    :return: Vector kernel matrix
    :rtype: tf.Tensor
    """
    D = pdist(X)
    return _kernel_from_dist(D, rel_sigma)
