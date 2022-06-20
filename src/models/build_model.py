import numpy as np
import torch as th

import config
import helpers
from models.ddc import DDCModel
from models.ddc_ae import DDCAE
from models.ddc_uco import DDCUCO
from models.tf_ddc_ae import TFDDCAE
from data.data_module import load_dataset
from dec.dec import DEC
from dec.tf_dec import TFDEC


MODEL_CONSTRUCTORS = {
    "DDCModel": DDCModel,
    "DDCAE": DDCAE,
    "DEC": DEC,
    "TFDEC": TFDEC,
    "DTKC": DDCUCO,
    "TFDDCAE": TFDDCAE,
}


def build_model(model_cfg, skip_load_weights=False, run=None):
    """
    Build the model specified by `model_cfg`.
    """
    if model_cfg.class_name not in MODEL_CONSTRUCTORS:
        raise ValueError(f"Invalid model type: {model_cfg.type}")
    model = MODEL_CONSTRUCTORS[model_cfg.class_name](model_cfg)

    if (getattr(model_cfg, "initial_weights", None) is not None) and (not skip_load_weights):
        size = model_cfg.backbone_config.input_size
        if len(size) > 1:
            data = np.random.random(size=(100, size[1], size[2], size[0]))
        else:
            data = np.random.random(size=(100, size[0]))

        _ = model({
            "data": data,
            "idx": np.arange(100),
        })
        weights_file = config.INITIAL_WEIGHTS_DIR / model_cfg.initial_weights / f"run-{run}.h5"
        model.load_weights(weights_file, by_name=True, skip_mismatch=True)
        print(f"Successfully loaded initial weights from {weights_file}")

    return model


def from_file(experiment_name=None, tag=None, run=None, ckpt="best", return_data=False, return_config=False, **kwargs):
    """
    Load a trained from disc

    :param experiment_name: Name of the experiment (name of the config)
    :type experiment_name: str
    :param tag: 8-character experiment identifier
    :type tag: str
    :param run: Training run to load
    :type run: int
    :param ckpt: Checkpoint to load. Specify a valid checkpoint, or "best" to load the best model.
    :type ckpt: Union[int, str]
    :param return_data: Return the dataset?
    :type return_data: bool
    :param return_config: Return the experiment config?
    :type return_config: bool
    :param kwargs:
    :type kwargs:
    :return: Loaded model, dataset (if return_data == True), config (if return_config == True)
    :rtype:
    """
    try:
        cfg = config.get_config_from_file(name=experiment_name, tag=tag)
    except FileNotFoundError:
        print("WARNING: Could not get pickled config.")
        cfg = config.get_config_by_name(experiment_name)

    model_dir = helpers.get_save_dir(experiment_name, identifier=tag, run=run)
    if ckpt == "best":
        model_file = "best.pt"
    else:
        model_file = f"checkpoint_{str(ckpt).zfill(4)}.pt"

    model_path = model_dir / model_file
    net = build_model(cfg.model_config)
    print(f"Loading model from {model_path}")
    net.load_state_dict(th.load(model_path, map_location=config.DEVICE))
    net.eval()

    out = [net]

    if return_data:
        dataset_kwargs = cfg.dataset_config.dict()
        for key, value in kwargs.items():
            dataset_kwargs[key] = value
        data, labels = load_dataset(to_dataset=False, **dataset_kwargs)
        out = [net, data, labels]

    if return_config:
        out.append(cfg)

    if len(out) == 1:
        out = out[0]

    return out
