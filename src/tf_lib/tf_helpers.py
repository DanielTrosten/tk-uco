from tensorflow import keras


def optimizer_from_config(cfg):
    ot = cfg.opt_type.lower()
    if ot == "adam":
        opt = keras.optimizers.Adam(lr=cfg.learning_rate, clipnorm=cfg.clip_norm)
    elif ot == "sgd":
        opt = keras.optimizers.SGD(lr=cfg.learning_rate, momentum=cfg.sgd_momentum, clipnorm=cfg.clip_norm)
    else:
        raise ValueError(f"Invalid optimizer type: {ot}.")
    return opt
