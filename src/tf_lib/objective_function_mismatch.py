import numpy as np
import tensorflow as tf

from dec.tf_dec import TFDEC


def _flat_cat_norm(grad):
    flat = np.concatenate([t.numpy().ravel() for t in grad], axis=0)
    return flat / np.linalg.norm(flat)


def get_grads_dec(model, batch):
    with tf.GradientTape(persistent=True) as tape:
        pred = model(batch, training=True)
        targets = model.target_dist(batch["idx"])
        l_cluster = model.compiled_loss(targets, pred, regularization_losses=[])
        l_companion = sum(model.losses)

    params = model.backbone.trainable_variables
    g_cluster = tape.gradient(l_cluster, params)
    g_companion = tape.gradient(l_companion, params)
    return g_cluster, g_companion


def get_grads_ddc(model, batch):
    l_cluster = tf.constant(0.0)
    l_companion = tf.constant(0.0)
    with tf.GradientTape(persistent=True) as tape:
        _ = model(batch, training=True)

        for name, loss in model.loss_values.items():
            if ("uco" in name) or ("reconstruct" in name):
                l_companion += loss
            else:
                l_cluster += loss

    params = model.backbone.trainable_variables
    g_cluster = tape.gradient(l_cluster, params)
    g_companion = tape.gradient(l_companion, params)
    return g_cluster, g_companion


def calc_ofm_simple(model, batch):
    if isinstance(model, TFDEC):
        g_cluster, g_companion = get_grads_dec(model, batch)
    else:
        g_cluster, g_companion = get_grads_ddc(model, batch)

    g_cluster = _flat_cat_norm(g_cluster)
    g_companion = _flat_cat_norm(g_companion)
    ofm = np.dot(g_cluster, g_companion)
    return ofm


def calc_ofm_pairwise(model, batch):
    with tf.GradientTape(persistent=True) as tape:
        _ = model(batch, training=True)

    params = model.backbone.trainable_variables
    loss_names = list(model.loss_values.keys())
    grads = [_flat_cat_norm(tape.gradient(model.loss_values[name], params)) for name in loss_names]
    grads = np.stack(grads, axis=0)
    ofm_arr = grads @ grads.T

    ofm = {}
    for i in range(len(grads)):
        for j in range(i+1, len(grads)):
            ofm[f"{loss_names[i]}_{loss_names[j]}"] = ofm_arr[i, j]
    return ofm


def calc_ofm(model, batch, mode="simple"):
    if mode == "simple":
        return calc_ofm_simple(model, batch)
    elif mode == "pairwise":
        return calc_ofm_pairwise(model, batch)
    raise RuntimeError(f"Invalid OFM mode: {mode}")
