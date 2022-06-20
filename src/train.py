import os
import sys
import wandb
import pytorch_lightning as pl
from copy import deepcopy

import config
import helpers
from data.data_module import DataModule
from models.build_model import build_model
from lib.loggers import ConsoleLogger, WeightsAndBiasesLogger
from lib.evaluate import evaluate, log_best_run

import torch as th


def train(cfg, ename, tag, run, net, data_module):
    save_dir = helpers.get_save_dir(ename, tag, run)
    os.makedirs(save_dir, exist_ok=True)
    cfg.to_pickle(save_dir / "config.pkl")

    best_callback = pl.callbacks.ModelCheckpoint(dirpath=save_dir, filename="best_{epoch:04d}",
                                                 monitor="train_loss/tot", mode="min", period=cfg.eval_interval,
                                                 save_top_k=1)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=save_dir, filename="checkpoint_{epoch:04d}",
                                                       save_top_k=-1, period=cfg.checkpoint_interval)

    wandb_logger = WeightsAndBiasesLogger(ename, tag, run, cfg, net)
    console_logger = ConsoleLogger(ename)

    # ==== Train ====
    try:
        gradient_clip_val = cfg.model_config.optimizer_config.clip_norm
    except AttributeError:
        gradient_clip_val = 0

    trainer = pl.Trainer(
        callbacks=[best_callback, checkpoint_callback],
        logger=[wandb_logger, console_logger],
        log_every_n_steps=data_module.n_batches,
        check_val_every_n_epoch=cfg.eval_interval,
        progress_bar_refresh_rate=0,
        max_epochs=cfg.n_epochs,
        terminate_on_nan=True,
        gradient_clip_val=gradient_clip_val,
        automatic_optimization=net.enable_automatic_optimization,
        gpus=cfg.gpus,
    )
    trainer.fit(net, datamodule=data_module)

    # ==== Evaluate ====
    # Validation set
    net.test_prefix = "val"
    val_results = evaluate(net, best_callback.best_model_path, data_module.val_dataloader(), console_logger)
    # Test set
    net.test_prefix = "test"
    test_results = evaluate(net, best_callback.best_model_path, data_module.test_dataloader(), console_logger)
    # Log evaluation results
    wandb_logger.log_summary(val_results, test_results)
    wandb.join()

    return val_results, test_results


def main(ename, cfg, tag):
    data_module = DataModule(cfg.dataset_config)

    val_logs, test_logs = [], []
    for run in range(cfg.n_runs):
        net = build_model(cfg.model_config)
        net.train_loader = data_module.train_dataloader()
        print(net)

        val, test = train(cfg, ename, tag, run, net, data_module)
        val_logs.append(val)
        test_logs.append(test)

    log_best_run(val_logs, test_logs, cfg, ename, tag)


if __name__ == '__main__':
    ename, cfg = config.get_experiment_config()
    tag = os.environ.get("EXPERIMENT_ID", None)
    if tag is None:
        tag = wandb.util.generate_id()
        print(f"Could not find EXPERIMENT_ID in environment variables. Using generated tag '{tag}'.")

    if "is_tensorflow_model" in cfg.model_config.dict():
        from tf_lib.tf_train import main as tf_main
        tf_main(ename, cfg, tag)

    else:
        main(ename, cfg, tag)

