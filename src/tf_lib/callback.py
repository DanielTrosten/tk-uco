import os
import time
import wandb
import numpy as np
import plotly.figure_factory as ff
from copy import deepcopy
from tensorflow import keras
from tabulate import tabulate
import tensorflow as tf

import config
import helpers
from tf_lib.loss import DDC1, UCO
from tf_lib.evaluate import evaluate_model


class CalculateMetrics(keras.callbacks.Callback):
    def __init__(self, eval_freq, dataset, batch_size, calc_ofm, ofm_mode="simple"):
        self.eval_freq = eval_freq
        self.dataset = dataset
        self.batch_size = batch_size
        self.calc_ofm = calc_ofm
        self.ofm_mode = ofm_mode

    def on_epoch_end(self, epoch, logs=None):
        logs["tot"] = logs["loss"]
        del logs["loss"]
        helpers.add_prefix(logs, "train_loss", inplace=True)

        logs["epoch"] = epoch

        if ((epoch + 1) % self.eval_freq) == 0:
            losses, metrics = evaluate_model(self.model, self.dataset, self.batch_size, calc_ofm=self.calc_ofm,
                                             ofm_mode=self.ofm_mode)
            logs.update(**metrics)
            logs.update(**losses)

        log_keys = list(logs.keys())
        for k in log_keys:
            if "sigma" in k:
                del logs[k]


class Printer(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = deepcopy(logs)
        
        if "val_loss/tot" in logs:
            # Remove the train losses to unclutter the table
            for key in [k for k in logs.keys() if k.startswith("train")]:
                del logs[key]

        headers = list(logs.keys())
        values = list(logs.values())
        if "epoch" in headers:
            helpers.move_elem_to_idx(headers, elem="epoch", idx=0, twins=(values,))
        if "time_delta" in headers:
            helpers.move_elem_to_idx(headers, elem="time_delta", idx=1, twins=(values,))
        print(tabulate([values], headers=headers), "\n")


class WandB(keras.callbacks.Callback):
    def __init__(self, name, tag, run, cfg, net):
        super(WandB, self).__init__()

        self.save_dir = helpers.get_save_dir(name, tag, run)
        os.makedirs(self.save_dir, exist_ok=True)

        self.group = f"{name}-{tag}"
        full_name = f"{self.group}-run-{run}"
        run_kwargs = dict(
            group=self.group,
            name=full_name,
            id=full_name,
            config=config.hparams_dict(cfg),
            dir=str(self.save_dir),
            reinit=True
        )
        if not cfg.is_sweep:
            run_kwargs["project"] = config.WANDB_PROJECT
        self.wanbd_run = wandb.init(**run_kwargs)

        self.log_sigmas = getattr(cfg, "log_sigmas", False)
        self.prev_keys = None
        self.prev_means = 0

    @staticmethod
    def _cmat_to_heatmap(cmat):
        ax_labels = list(range(cmat.shape[0]))
        fig = ff.create_annotated_heatmap(x=ax_labels, y=ax_labels, z=cmat, showscale=True, colorscale="Inferno")
        return wandb.data_types.Plotly(fig)

    def on_epoch_end(self, epoch, logs=None):
        log_metrics = deepcopy(logs)

        if "val_metrics/cmat" in log_metrics:
            # del log_metrics["val_metrics/cmat"]
            log_metrics["val_metrics/cmat"] = self._cmat_to_heatmap(log_metrics["val_metrics/cmat"])

        self.wanbd_run.log(log_metrics, step=log_metrics.get("epoch", epoch))

    def on_train_batch_end(self, batch, logs=None):
        if not self.log_sigmas:
            return

        keys, means = [], []
        for key, value in logs.items():
            if "sigma" in key:
                keys.append(key)
                means.append(value)
        means = np.array(means)

        if self.prev_keys is not None:
            assert self.prev_keys == keys

        values = (batch + 1) * means - batch * self.prev_means
        self.prev_keys = keys
        self.prev_means = means
        # dct = dict(zip(keys, values))
        # dct["batch"] = batch
        # self.wanbd_run.log(dct, commit=False)
        with open(self.save_dir / "sigmas.csv", "a+") as f:
            f.write(", ".join([str(v) for v in values]) + "\n")

    def log_summary(self, val_results, test_results):
        for set_type, logs in zip(["val", "test"], [val_results, test_results]):
            logs = deepcopy(logs)
            for key, value in logs.items():
                self.wanbd_run.summary[f"summary/{key}"] = value


class ModelCheckpoint(keras.callbacks.Callback):
    def __init__(self, save_dir, freq, best_loss_term):
        super(ModelCheckpoint, self).__init__()

        self.save_dir = save_dir
        self.freq = freq
        self.best_loss_term = best_loss_term
        self.best_loss = np.inf
        self.last_model_files = {}

    def _save_model(self, epoch, prefix="checkpoint", include_epoch=True):
        if include_epoch:
            model_path = self.save_dir / "{prefix}_{epoch:04d}.h5".format(prefix=prefix, epoch=epoch)
        else:
            model_path = self.save_dir / f"{prefix}.h5"

        self.model.save_weights(model_path)
        self.last_model_files[prefix] = model_path
        print("Successfully saved model weights to:", model_path)

    def on_epoch_end(self, epoch, logs=None):
        if ((epoch + 1) % self.freq) == 0:
            self._save_model(epoch, "checkpoint", include_epoch=True)

        if self.best_loss_term in logs and logs[self.best_loss_term] < self.best_loss:
            self._save_model(epoch, "best", include_epoch=False)
            self.best_loss = logs[self.best_loss_term]
