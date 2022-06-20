import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.special import comb
from sklearn.metrics import confusion_matrix, normalized_mutual_info_score


def ordered_cmat(labels, pred):
    """
    Compute the confusion matrix and accuracy corresponding to the best cluster-to-class assignment.

    :param labels: Label array
    :type labels: np.array
    :param pred: Predictions array
    :type pred: np.array
    :return: Accuracy and confusion matrix
    :rtype: Tuple[float, np.array]
    """
    cmat = confusion_matrix(labels, pred)
    ri, ci = linear_sum_assignment(-cmat)
    ordered = cmat[np.ix_(ri, ci)]
    acc = np.sum(np.diag(ordered))/np.sum(ordered)
    return acc, ordered


def cmat_to_dict(cmat, prefix=""):
    return {prefix + f"{i}_{j}": cmat[i, j] for i in range(cmat.shape[0]) for j in range(cmat.shape[1])}


def cmat_from_dict(dct, prefix="", del_elements=False):
    n_clusters = 0
    while prefix + f"{n_clusters}_{0}" in dct.keys():
        n_clusters += 1

    out = np.empty((n_clusters, n_clusters))
    for i in range(n_clusters):
        for j in range(n_clusters):
            key = prefix + f"{i}_{j}"
            out[i, j] = int(dct[key])

            if del_elements:
                del dct[key]

    return out.astype(int)


def rand_index(classes, clusters):
    classes = classes.astype(np.int32)
    clusters = clusters.astype(np.int32)
    tp_plus_fp = comb(np.bincount(clusters), 2).sum()
    tp_plus_fn = comb(np.bincount(classes), 2).sum()
    A = np.c_[(clusters, classes)]
    tp = sum(comb(np.bincount(A[A[:, 0] == i, 1]), 2).sum()
             for i in set(clusters))
    fp = tp_plus_fp - tp
    fn = tp_plus_fn - tp
    tn = comb(len(A), 2) - tp - fp - fn
    return (tp + tn) / (tp + fp + fn + tn)


def calc_metrics(labels, pred, flatten_cmat=True):
    """
    Compute metrics.

    :param labels: Label tensor
    :type labels: th.Tensor
    :param pred: Predictions tensor
    :type pred: th.Tensor
    :return: Dictionary containing calculated metrics
    :rtype: dict
    """
    acc, cmat = ordered_cmat(labels, pred)
    metrics = {
        "acc": acc,
        "nmi": normalized_mutual_info_score(labels, pred, average_method="geometric"),
        "rand": rand_index(labels, pred),
    }
    if flatten_cmat:
        metrics.update(cmat_to_dict(cmat, prefix="cmat/"))
    else:
        metrics["cmat"] = cmat
    return metrics
