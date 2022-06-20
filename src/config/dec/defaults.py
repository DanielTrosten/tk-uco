from typing import Union, Tuple, Optional
from typing_extensions import Literal

from config import constants
from config.config import Config
from config.defaults import MLP, CNN, Optimizer, Loss, UCOLoss, Dataset as _Dataset


class Dataset(_Dataset):
    drop_last: bool = False
    shuffle_first: bool = True
    initial_shuffle: bool = False


# class Loss(Config):
#     funcs: str
#     n_clusters: int = None
#     weights: Tuple[float, ...] = None
#
#
# class UCOLoss(Loss):
#     rel_sigma: float = 0.15
#     uco_weighting_method: Literal["constant", "linear", "exp"] = "constant"
#     uco_kernel_type: Literal["naive", "tensor"] = "tensor"
#     uco_lambda: float = 0.01
#     uco_assignment_gradient: bool = True


class ClusteringModule(Config):
    n_clusters: int = None
    alpha: float = 1.0


class DEC(Config):
    # improved: bool = False
    backbone_config: Union[MLP, CNN]
    cm_config: ClusteringModule = ClusteringModule()
    decoder_config: Union[MLP, CNN]

    pre_train_loss_config: Loss = Loss(funcs="reconstruct")
    fine_tune_loss_config: Loss = Loss(funcs="kl")

    n_pre_train_epochs: int = 150
    dropout_prob: float = 0.0
    target_dist_update_interval: int = 140

    # pre_train_optimizer_config: Optimizer = Optimizer(opt_type="sgd", learning_rate=0.1, sgd_momentum=0.9)
    pre_train_optimizer_config: Optimizer = Optimizer(opt_type="adam", learning_rate=1e-3)
    fine_tune_optimizer_config: Optimizer = Optimizer(opt_type="adam", learning_rate=1e-3)

    batch_size: int = None
    calc_ofm: bool = False


class TFDEC(Config):
    is_tensorflow_model: bool = True

    backbone_config: Union[MLP, CNN]
    cm_config: ClusteringModule = ClusteringModule()
    decoder_config: Union[MLP, CNN]
    bottleneck_units: int = 10

    loss_config: Loss = Loss(funcs="")

    n_pre_train_epochs: int = 200
    target_dist_update_interval: int = 140

    pre_train_optimizer_config: Optimizer = Optimizer(opt_type="adam", learning_rate=1e-3)
    optimizer_config: Optimizer = Optimizer(opt_type="adam", learning_rate=1e-3)

    batch_size: int = None
    calc_ofm: bool = False

    initial_weights: str = None


class Experiment(Config):
    _glob_vars: Tuple[str, ...] = ("n_clusters", "batch_size")

    # Dataset config
    dataset_config: Dataset

    # Number of clusters
    n_clusters: int
    # Batch size
    batch_size: int = 256

    # Model config
    model_config: Union[DEC, TFDEC]
    # Number of training runs
    n_runs = 20
    # Number of training epochs
    n_epochs = 600
    # Number of epochs between model evaluation.
    eval_interval: int = 5
    # Number of epochs between model checkpoints.
    checkpoint_interval = 100
    # Patience for early stopping.
    patience = 50000
    # Number of samples to use for evaluation. Set to None to use all samples in the dataset.
    n_eval_samples: int = None
    # Term in loss function to use for model selection. Set to "tot" to use the sum of all terms.
    best_loss_term = "tot"
    is_sweep: bool = False
    calc_ofm: bool = False
    gpus: int = constants.GPUS
    random_seed: Optional[int] = 7
