import os
import numpy as np
import wandb
from datetime import datetime
import yaml

import config
from models.build_model import build_model


def get_weights_file(ename, tag, run):
    return config.INITIAL_WEIGHTS_DIR / f"{ename}-{tag}" / f"run-{run}.h5"


def write_log_line(ename, tag):
    fname = config.INITIAL_WEIGHTS_DIR / "info.yml"
    
    if os.path.exists(fname):
        with open(fname, "r") as f:
            info = yaml.safe_load(f)
    else:
        info = []
        
    info.append({
        "ename": ename,
        "tag": tag,
        "timestamp": datetime.now().strftime(config.DATETIME_FMT) 
    })
    
    with open(fname, "w") as f:
        yaml.safe_dump(info, f)


if __name__ == '__main__':
    ename, cfg = config.get_experiment_config()
    tag = wandb.util.generate_id()

    for run in range(cfg.n_runs):
        model = build_model(cfg.model_config, skip_load_weights=True)
        size = cfg.model_config.backbone_config.input_size

        if len(size) > 1:
            data = np.random.random(size=(100, size[1], size[2], size[0]))
        else:
            data = np.random.random(size=(100, size[0]))

        if "dec" in ename:
            _ = model({"data": data, "idx": np.arange(100)})
        else:
            _ = model(data)

        weights_file = get_weights_file(ename, tag, run)
        os.makedirs(weights_file.parents[0], exist_ok=True)
        model.save_weights(weights_file)
        print(f"Successfully saved initialized weights to: {weights_file}")
    
    write_log_line(ename, tag)
