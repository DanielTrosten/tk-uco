from .cnn import CNN
from .mlp import MLP

BACKBONE_CONSTRUCTORS = {
        "CNN": CNN,
        "MLP": MLP,
}


def create_backbone(cfg, input_size=None, flatten_output=True):
    if cfg.class_name not in BACKBONE_CONSTRUCTORS:
        raise RuntimeError(f"Invalid backbone: '{cfg.class_name}'")
    return BACKBONE_CONSTRUCTORS[cfg.class_name](cfg, input_size=input_size, flatten_output=flatten_output)