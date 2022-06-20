import os
import wandb
import tensorflow as tf

import helpers
from lib.evaluate import log_best_run
from tf_lib import callback
from tf_lib.evaluate import evaluate_model
from tf_lib.tf_data_module import TFDataModule
from tf_lib.tf_helpers import optimizer_from_config
from models.build_model import build_model


def train(cfg, ename, tag, run, net, data_module):
    save_dir = helpers.get_save_dir(ename, tag, run)
    os.makedirs(save_dir, exist_ok=True)
    cfg.to_pickle(save_dir / "config.pkl")
    ofm_mode = getattr(cfg, "ofm_mode", "simple")

    wandb_callback = callback.WandB(ename, tag=tag, run=run, cfg=cfg, net=net)
    checkpoint_callback = callback.ModelCheckpoint(save_dir=save_dir, freq=cfg.checkpoint_interval,
                                                   best_loss_term=f"val_loss/{cfg.best_loss_term}")
    callbacks = [
        callback.CalculateMetrics(cfg.eval_interval, data_module.batched_val_dataset, cfg.batch_size, cfg.calc_ofm,
                                  ofm_mode=ofm_mode),
        callback.Printer(),
        checkpoint_callback,
        wandb_callback,
    ]

    opt = optimizer_from_config(cfg.model_config.optimizer_config)

    if "DEC" in cfg.model_config.class_name:
        net.pre_train(data_module.batched_train_dataset, callbacks=[callbacks[1], wandb_callback])
        net.train_data = data_module.as_numpy_arrays(data_module.train_dataset, assert_ordered=True)
        net.compile(optimizer=opt, loss="kld")
        net.init_training()

        data, idx, labels = data_module.as_numpy_arrays(data_module.train_dataset)
        idx = idx.astype(int)
        update_interval = cfg.model_config.target_dist_update_interval
        for e in range(-(-(cfg.n_epochs - cfg.model_config.n_pre_train_epochs) // update_interval)):
            net.update_target_dist()
            targets = net.target_dist(idx)
            step = cfg.model_config.n_pre_train_epochs + e * update_interval + 1
            net.fit(
                x={"data": data, "idx": idx},
                y=targets,
                epochs=min(step + update_interval, cfg.n_epochs),
                batch_size=cfg.batch_size,
                callbacks=callbacks,
                initial_epoch=step,
                verbose=True
            )

    else:
        net.compile(optimizer=opt)
        net.fit(
            data_module.batched_train_dataset,
            epochs=cfg.n_epochs,
            callbacks=callbacks,
            initial_epoch=getattr(cfg.model_config, "n_pre_train_epochs", 0),
            verbose=False
        )

    # Restore weights from best model
    net.load_weights(checkpoint_callback.last_model_files["best"])
    # Run evaluation
    val_losses, val_metrics = evaluate_model(net, data_module.batched_val_dataset, cfg.batch_size, set_type="val",
                                             calc_ofm=cfg.calc_ofm, ofm_mode=ofm_mode)
    test_losses, test_metrics = evaluate_model(net, data_module.batched_test_dataset, cfg.batch_size, set_type="test",
                                               calc_ofm=cfg.calc_ofm, ofm_mode=ofm_mode)
    # Log results
    val_logs = dict(**val_losses, **val_metrics)
    test_logs = dict(**test_losses, **test_metrics)
    wandb_callback.log_summary(val_logs, test_logs)
    wandb_callback.wanbd_run.finish()

    return val_logs, test_logs


def reset_wandb_env():
    exclude = {
        "WANDB_PROJECT",
        "WANDB_ENTITY",
        "WANDB_API_KEY",
    }
    for k, v in os.environ.items():
        if k.startswith("WANDB_") and k not in exclude:
            del os.environ[k]


def main(ename, cfg, tag):
    if cfg.random_seed is not None:
        tf.random.set_seed(cfg.random_seed)

    data_module = TFDataModule(cfg.dataset_config)

    val_logs, test_logs = [], []
    for run in range(cfg.n_runs):
        net = build_model(cfg.model_config, run=run)

        val, test = train(cfg, ename, tag, run, net, data_module)
        val_logs.append(val)
        test_logs.append(test)
        wandb.join()

    log_best_run(val_logs, test_logs, cfg, ename, tag)
