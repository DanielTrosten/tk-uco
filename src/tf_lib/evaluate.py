import numpy as np

import helpers
from lib.metrics import calc_metrics
from tf_lib import objective_function_mismatch


def evaluate_model(model, dataset, batch_size, calc_ofm=False, set_type="val", ofm_mode="simple"):
    losses, labels, predictions, ofm = [], [], [], []
    for batch, targets in dataset:
        pred = model.predict_on_batch(batch).argmax(axis=1)
        predictions.append(pred)
        labels.append(targets["output_1"].numpy())
        if batch["data"].shape[0] == batch_size:
            loss = model.test_on_batch(x=batch, y=targets, return_dict=True)
            loss["tot"] = loss["loss"]
            del loss["loss"]
            losses.append(loss)

            if calc_ofm:
                ofm.append(objective_function_mismatch.calc_ofm(model, batch, mode=ofm_mode))

    labels = np.concatenate(labels, axis=0)
    predictions = np.concatenate(predictions, axis=0)
    metrics = helpers.add_prefix(calc_metrics(labels, predictions, flatten_cmat=False), f"{set_type}_metrics")
    losses = helpers.add_prefix(helpers.dict_means(losses), f"{set_type}_loss")

    ofm_key = f"{set_type}_metrics/ofm"
    if ofm:
        if isinstance(ofm[0], dict):
            for key, value in helpers.dict_means(ofm).items():
                metrics[ofm_key + "/" + key.replace("/", "_")] = value
        else:
            metrics[ofm_key] = np.mean(ofm)
    else:
        metrics[ofm_key] = np.nan

    # Remove sigma-stuff from losses.
    loss_keys = list(losses.keys())
    for k in loss_keys:
        if "sigma" in k:
            del losses[k]

    return losses, metrics
