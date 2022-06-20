from typing import Tuple, List, Union, Any, Optional, Callable
from typing_extensions import Literal

from config import Config, layers, constants


# ======================================================================================================================
# Dataset
# ======================================================================================================================

class Dataset(Config):
    # Name of the dataset. Must correspond to a filename in data/processed/
    name: str
    # Number of samples to load. Set to None to load all samples
    n_train_samples: int = None
    n_val_samples: int = None
    n_test_samples: int = None
    # Subset of views to load. Set to None to load all views
    select_views: Tuple[int, ...] = None
    # Subset of labels (classes) to load. Set to None to load all classes
    select_labels: Tuple[int, ...] = None
    # Number of samples to load for each class. Set to None to load all samples
    train_label_counts: Tuple[int, ...] = None
    val_label_counts: Tuple[int, ...] = None
    test_label_counts: Tuple[int, ...] = None
    random_seed: int = None
    # Batch size (This is a placeholder. Set the batch size in Experiment).
    batch_size: int = None
    drop_last: bool = True
    shuffle_first: bool = True
    initial_shuffle: bool = False


# ======================================================================================================================
# Backbones
# ======================================================================================================================

class Backbone(Config):
    input_size: Optional[Tuple[int, ...]]
    layers: Tuple[Any, ...]


class CNN(Backbone):
    # Network layers
    layers: Union[Tuple[Tuple[Union[int, str, None], ...], ...], Callable] = layers.cnn_small
    padding: str = "valid"


class CNN1D(Backbone):
    layers: Tuple[Union[str, dict], ...] = layers.cnn_1d_medium


class MLP(Backbone):
    # Units in the network layers
    layers: Tuple[Union[int, str], ...] = (512, 512, 256)
    # Activation function. Can be a single string specifying the activation function for all layers, or a list/tuple of
    # string specifying the activation function for each layer.
    activation: Union[str, None, List[Union[None, str]], Tuple[Union[None, str], ...]] = "relu"
    # Include bias parameters? A single bool for all layers, or a list/tuple of booleans for individual layers.
    use_bias: Union[bool, Tuple[bool, ...]] = True
    # Include batch norm after layers? A single bool for all layers, or a list/tuple of booleans for individual layers.
    use_bn: Union[bool, Tuple[bool, ...]] = False


class GRU(Backbone):
    hidden_size: int
    n_layers: int = 2
    bidirectional: bool = True


# ======================================================================================================================
# Clustering modules
# ======================================================================================================================

class DDC(Config):
    # Number of clusters
    n_clusters: int = None
    # Number of units in the first fully connected layer
    n_hidden: int = 100
    # Use batch norm after the first fully connected layer?
    use_bn: bool = True
    bn_trainable_params: bool = True


# ======================================================================================================================
# Loss
# ======================================================================================================================

class Loss(Config):
    # Number of clusters
    n_clusters: int = None
    # Terms to use in the loss, separated by '|'. E.g. "ddc_1|ddc_2|ddc_3|" for the DDC clustering loss
    funcs: str = "ddc_1|ddc_2|ddc_3"
    # Optional weights for the loss terms. Set to None to have all weights equal to 1.
    weights: Tuple[Union[float, int], ...] = None
    # Multiplication factor for the sigma hyperparameter
    rel_sigma: float = 0.15

    sigma_mode: Literal["running_mean", "otf"] = "otf"
    sigma_momentum: float = 0.9


class UCOLoss(Loss):
    funcs: str = "ddc_1|ddc_2|ddc_3|uco"
    n_outputs: int = 2
    uco_weighting_method: Literal["constant", "linear", "exp"] = "constant"
    uco_kernel_type: Literal["naive", "tensor"] = "tensor"
    uco_lambda: float = 0.01
    uco_assignment_gradient: bool = True


class ContrastiveLoss(Loss):
    funcs: str = "ddc_1|ddc_2|ddc_3|contrast"
    tau: str = 0.1
    delta: str = 1.0


# ======================================================================================================================
# Optimizer
# ======================================================================================================================


class Optimizer(Config):
    opt_type: Literal["adam", "sgd"] = "adam"
    # Base learning rate
    learning_rate: float = 1e-3
    # SGD momentum
    sgd_momentum: float = 0.0
    # Max gradient norm for gradient clipping.
    clip_norm: Optional[float] = 10.0
    # Step size for the learning rate scheduler. None disables the scheduler.
    scheduler_step_size: int = None
    # Multiplication factor for the learning rate scheduler
    scheduler_gamma: float = None


# ======================================================================================================================
# Models
# ======================================================================================================================

class BaseModel(Config):
    # Encoder network config
    backbone_config: Union[GRU, MLP, CNN1D, CNN]
    # Clustering module config
    cm_config: Union[DDC]
    # Loss function config
    loss_config: Loss
    # Optimizer config
    optimizer_config: Optimizer = Optimizer()
    batch_size: int = None
    calc_ofm: bool = False


class DDCModel(BaseModel):
    pass


class DDCUCO(BaseModel):
    is_tensorflow_model: bool = True
    initial_weights: str = None


class DDCAE(BaseModel):
    backbone_config: CNN
    decoder_config: CNN


class TFDDCAE(BaseModel):
    is_tensorflow_model: bool = True
    initial_weights: str = None
    decoder_config: CNN


# ======================================================================================================================
# Experiment
# ======================================================================================================================

class Experiment(Config):
    _glob_vars: Tuple[str, ...] = ("n_clusters", "batch_size")

    # Dataset config
    dataset_config: Dataset

    # Number of clusters
    n_clusters: int
    # Batch size
    batch_size: int = 120

    # Model config
    model_config: Union[DDCAE, DDCModel, DDCUCO, TFDDCAE]
    # Number of training runs
    n_runs: int = 20
    # Number of training epochs
    n_epochs: int = 200
    # Number of epochs between model evaluation.
    eval_interval: int = 4
    # Number of epochs between model checkpoints.
    checkpoint_interval: int = 50
    # Patience for early stopping.
    patience: int = 50000
    # Number of samples to use for evaluation. Set to None to use all samples in the dataset.
    n_eval_samples: int = None
    # Term in loss function to use for model selection. Set to "tot" to use the sum of all terms.
    best_loss_term: str = "tot"
    is_sweep: bool = False
    calc_ofm: bool = False
    ofm_mode: Literal["simple", "pairwise"] = "simple"
    gpus: int = constants.GPUS
    random_seed: Optional[int] = 7
    log_sigmas: bool = False
