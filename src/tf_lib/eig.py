"""
Custom eigensolver for companion objective computations. Modified version of TensorFlow's source code.
"""
import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.linalg import linalg_impl as _linalg


@tf.custom_gradient
def _eig_cg(x):
    with tf.device("cpu"):
        vals, vecs = tf.linalg.eigh(x)

    # Gradient of eigendecomposition
    def grad(grad_e, grad_v):
        eps = 1e-8
        e, v = vals, vecs
        # Construct the matrix f(i,j) = (i != j ? 1 / (e_i - e_j) : 0).
        # Notice that because of the term involving f, the gradient becomes
        # infinite (or NaN in practice) when eigenvalues are not unique.
        # Mathematically this should not be surprising, since for (k-fold)
        # degenerate eigenvalues, the corresponding eigenvectors are only defined
        # up to arbitrary rotation in a (k-dimensional) subspace.

        denom = array_ops.expand_dims(e, -2) - array_ops.expand_dims(e, -1)
        f = tf.where(tf.abs(denom) >= eps, 1/denom, tf.zeros_like(denom))
        f = array_ops.matrix_set_diag(f, tf.zeros_like(e))

        grad_a = math_ops.matmul(
            v,
            math_ops.matmul(
                array_ops.matrix_diag(grad_e) +
                f * math_ops.matmul(v, grad_v, adjoint_a=True),
                v,
                adjoint_b=True))

        # The forward op only depends on the lower triangular part of a, so here we
        # symmetrize and take the lower triangle
        grad_a = array_ops.matrix_band_part(grad_a + _linalg.adjoint(grad_a), -1, 0)
        grad_a = array_ops.matrix_set_diag(grad_a,
                                           0.5 * array_ops.matrix_diag_part(grad_a))
        return grad_a

    return (vals, vecs), grad
