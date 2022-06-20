import time
import numpy as np
import torch as th
import pytorch_lightning as pl

import helpers
from lib.metrics import calc_metrics
from lib.objective_function_mismatch import calc_ofm


class ModelBase(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.loss = None
        self._test_prefix = "test"
        self.enable_automatic_optimization = True
        self.train_loader = None
        self.current_time = time.time()

    @property
    def test_prefix(self):
        return self._test_prefix

    @test_prefix.setter
    def test_prefix(self, new):
        assert new in ("test", "val"), f"Invalid test prefix: {new}"
        self._test_prefix = new

    def _optimizer_from_cfg(self, cfg, params):
        if cfg.opt_type == "adam":
            optimizer = th.optim.Adam(params, lr=cfg.learning_rate)
        elif cfg.opt_type == "sgd":
            optimizer = th.optim.SGD(params, lr=cfg.learning_rate, momentum=cfg.sgd_momentum)
        else:
            raise RuntimeError()
        return optimizer
    
    def configure_optimizers(self):
        return self._optimizer_from_cfg(self.cfg.optimizer_config, self.parameters())

    def _log_dict(self, dct, prefix, sep="/", ignore_keys=tuple()):
        if prefix:
            prefix += sep
        for key, value in dct.items():
            if key not in ignore_keys:
                self.log(prefix + key, value)

    def get_loss(self):
        return self.loss(self)

    def get_ofm(self, batch):
        return calc_ofm(batch, self)

    def training_step(self, batch, idx):
        _ = self(batch[0])
        losses = self.get_loss()

        now = time.time()
        time_delta = now - self.current_time
        self.current_time = now

        self._log_dict({"epoch": self.current_epoch, "time_delta": time_delta}, prefix="")
        self._log_dict(losses, prefix="train_loss")
        return losses["tot"]

    def _val_test_step(self, batch, idx, prefix):
        data, labels, data_idx = batch
        pred = self(data, data_idx)

        # Only evaluate losses on full batches
        if data.size(0) == self.cfg.batch_size:
            losses = self.get_loss()
            self._log_dict(losses, prefix=f"{prefix}_loss")

            # Calculate OFM?
            if self.cfg.calc_ofm:
                self.log(f"{prefix}_metrics/ofm", self.get_ofm(batch))

        return np.stack((helpers.npy(labels), helpers.npy(pred).argmax(axis=1)), axis=0)

    def _val_test_epoch_end(self, step_outputs, prefix):
        if not isinstance(step_outputs, list):
            step_outputs = [step_outputs]

        labels_pred = np.concatenate(step_outputs, axis=1)
        mtc = calc_metrics(labels=labels_pred[0], pred=labels_pred[1])
        self._log_dict(mtc, prefix=f"{prefix}_metrics")

    def validation_step(self, batch, idx):
        return self._val_test_step(batch, idx, "val")

    def validation_epoch_end(self, step_outputs):
        return self._val_test_epoch_end(step_outputs, "val")

    def test_step(self, batch, idx):
        return self._val_test_step(batch, idx, self.test_prefix)

    def test_epoch_end(self, step_outputs):
        return self._val_test_epoch_end(step_outputs, self.test_prefix)
