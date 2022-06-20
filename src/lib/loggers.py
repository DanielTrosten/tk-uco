import wandb
import argparse
import numpy as np
import plotly.figure_factory as ff
from contextlib import contextmanager
from copy import deepcopy
from tabulate import tabulate
from typing import Union, Dict, Optional, Any
from pytorch_lightning.loggers.base import LightningLoggerBase

import helpers
import config
from lib.metrics import cmat_from_dict


def fix_cmat(metrics, set_type, cmat_hook=lambda x: x):
    if f"{set_type}_metrics/cmat/0_0" not in metrics:
        return

    cmat = cmat_from_dict(metrics, prefix=f"{set_type}_metrics/cmat/", del_elements=True)
    metrics[f"{set_type}_metrics/cmat"] = cmat_hook(cmat)


class ConsoleLogger(LightningLoggerBase):
    def __init__(self, ename, ignore_keys=tuple(), print_cmat=True):
        super(ConsoleLogger, self).__init__()
        self.ignore_keys = ignore_keys
        self.print_cmat = print_cmat
        self._ename = ename
        self._version = "0"

    @property
    def experiment(self):
        return None

    def log_metrics(self, metrics, step=None):
        print_logs = deepcopy(metrics)
        for key in metrics.keys():
            if any([key.startswith(ik) for ik in self.ignore_keys]):
                del print_logs[key]

        if self.print_cmat:
            fix_cmat(print_logs, "val")
            fix_cmat(print_logs, "test")

        headers = list(print_logs.keys())
        values = list(print_logs.values())

        if "epoch" in headers:
            helpers.move_elem_to_idx(headers, elem="epoch", idx=0, twins=(values,))
        if "time_delta" in headers:
            helpers.move_elem_to_idx(headers, elem="time_delta", idx=1, twins=(values,))

        print(tabulate([values], headers=headers), "\n")

    def log_hyperparams(self, params):
        pass

    @property
    def name(self):
        return self._ename

    @property
    def version(self):
        return self._version


class WeightsAndBiasesLogger(LightningLoggerBase):
    def __init__(self, name, tag, run, cfg, net):
        super(WeightsAndBiasesLogger, self).__init__()

        self.group = f"{name}-{tag}"
        self.wanbd_run = wandb.init(
            project=config.WANDB_PROJECT,
            group=self.group,
            name=f"{self.group}-run-{run}",
            config=config.hparams_dict(cfg),
            dir=str(helpers.get_save_dir(name, tag, run)),
            reinit=True
        )
        wandb.watch(net)
        self._version = "0"
        self.logging_enabled = True

    @contextmanager
    def disable_logging(self):
        self.logging_enabled = False
        try:
            yield None
        finally:
            self.logging_enabled = True

    @property
    def experiment(self) -> Any:
        return self.wanbd_run

    @staticmethod
    def _cmat_to_heatmap(cmat):
        ax_labels = list(range(cmat.shape[0]))
        fig = ff.create_annotated_heatmap(x=ax_labels, y=ax_labels, z=cmat, showscale=True, colorscale="Inferno")
        return wandb.data_types.Plotly(fig)

    def log_metrics(self, metrics, step=None):
        if not self.logging_enabled:
            return

        log_metrics = deepcopy(metrics)
        fix_cmat(log_metrics, set_type="val", cmat_hook=self._cmat_to_heatmap)
        fix_cmat(log_metrics, set_type="test", cmat_hook=self._cmat_to_heatmap)
        self.wanbd_run.log(log_metrics, step=log_metrics["epoch"])

    def log_hyperparams(self, params):
        pass

    @property
    def name(self):
        return self.group

    @property
    def version(self):
        return self._version

    def log_summary(self, val_results, test_results):
        if not self.logging_enabled:
            return

        for set_type, logs in zip(["val", "test"], [val_results, test_results]):
            logs = deepcopy(logs)
            fix_cmat(logs, set_type=set_type)
            for key, value in logs.items():
                self.wanbd_run.summary[f"summary/{key}"] = value

