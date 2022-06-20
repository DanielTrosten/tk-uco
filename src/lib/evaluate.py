import os
import wandb
import pytorch_lightning as pl
from copy import deepcopy

import config
import helpers
from lib.loggers import fix_cmat


def evaluate(net, ckpt_path, loader, logger):
    # Define a separate trainer here, so we don't get unwanted val_* stuff in the results.
    eval_trainer = pl.Trainer(logger=logger, progress_bar_refresh_rate=0, gpus=config.GPUS)
    results = eval_trainer.test(model=net, test_dataloaders=loader, ckpt_path=ckpt_path, verbose=False)
    assert len(results) == 1
    return results[0]


def log_best_run(val_logs_list, test_logs_list, cfg, experiment_name, group_id):
    group = f"{experiment_name}-{group_id}"
    wandb_dir = str(helpers.get_save_dir(experiment_name, group_id, "best"))
    os.makedirs(wandb_dir, exist_ok=True)

    wanbd_run = wandb.init(
        project=config.WANDB_PROJECT,
        group=group,
        name=f"{group}-best",
        config=config.hparams_dict(cfg),
        dir=wandb_dir,
        reinit=True
    )

    best_run = None
    best_loss = float("inf")
    for run, logs in enumerate(val_logs_list):
        tot_loss = logs[f"val_loss/{cfg.best_loss_term}"]
        if tot_loss < best_loss:
            best_run = run
            best_loss = tot_loss

    for set_type, logs in zip(["val", "test"], [val_logs_list, test_logs_list]):
        best_logs = deepcopy(logs[best_run])
        best_logs["is_best"] = True
        best_logs["best_run"] = best_run

        if f"{set_type}_metrics/cmat" not in best_logs:
            fix_cmat(best_logs, set_type=set_type)

        for key, value in best_logs.items():
            wanbd_run.summary[f"summary/{key}"] = value
    wandb.join()
