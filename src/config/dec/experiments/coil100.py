import numpy as np
from config import layers
from config.defaults import MLP, Optimizer, CNN
from config.dec.defaults import Experiment, DEC, Loss, ClusteringModule, Dataset, UCOLoss, TFDEC, Optimizer


DATASET_NAME = "coil100"
N_CLUSTERS = 100
INPUT_SIZE = (3, 64, 64)

ENCODER_LAYERS = (500, 500, "out", 2000, "out")
ENCODER_ACTIVATION = ("relu", "relu", None, "relu", None)
DECODER_LAYERS = (500, 500, np.prod(INPUT_SIZE))
DECODER_ACTIVATION = ("relu", "relu", None)


dec_coil100 = Experiment(
    dataset_config=Dataset(name=DATASET_NAME),
    n_clusters=N_CLUSTERS,
    model_config=TFDEC(
        backbone_config=MLP(
            layers=ENCODER_LAYERS,
            activation=ENCODER_ACTIVATION,
            input_size=INPUT_SIZE,
        ),
        decoder_config=MLP(
            layers=DECODER_LAYERS,
            activation=DECODER_ACTIVATION,
        ),
        initial_weights="idec_coil100-35j1y4d8"
    ),
)

idec_coil100 = Experiment(
    dataset_config=Dataset(name=DATASET_NAME),
    n_clusters=N_CLUSTERS,
    model_config=TFDEC(
        backbone_config=MLP(
            layers=ENCODER_LAYERS,
            activation=ENCODER_ACTIVATION,
            input_size=INPUT_SIZE,
        ),
        decoder_config=MLP(
            layers=DECODER_LAYERS,
            activation=DECODER_ACTIVATION,
        ),
        loss_config=Loss(
            funcs="reconstruct",
            weights=(10.0,)
        ),
        initial_weights="idec_coil100-35j1y4d8"
    ),
    calc_ofm=True,
    
)

dec_uco_coil100 = Experiment(
    dataset_config=Dataset(name=DATASET_NAME),
    n_clusters=N_CLUSTERS,
    model_config=TFDEC(
        loss_config=UCOLoss(
            funcs="uco",
            uco_kernel_type="naive"
        ),
        backbone_config=MLP(
            layers=ENCODER_LAYERS,
            activation=ENCODER_ACTIVATION,
            input_size=INPUT_SIZE,
        ),
        decoder_config=MLP(
            layers=DECODER_LAYERS,
            activation=DECODER_ACTIVATION,
        ),
        initial_weights="idec_coil100-35j1y4d8",
        optimizer_config=Optimizer(learning_rate=1e-4),
    ),
    calc_ofm=True,
)


cdec_coil100 = Experiment(
    dataset_config=Dataset(name=DATASET_NAME),
    n_clusters=N_CLUSTERS,
    model_config=TFDEC(
        backbone_config=CNN(
            input_size=INPUT_SIZE,
        ),
        decoder_config=CNN(
            layers=layers.cnn_small_transposed(out_channels=INPUT_SIZE[0])
        ),
        initial_weights="icdec_coil100-2qalvjco"
    ),
)

icdec_coil100 = Experiment(
    dataset_config=Dataset(name=DATASET_NAME),
    n_clusters=N_CLUSTERS,
    model_config=TFDEC(
        backbone_config=CNN(
            input_size=INPUT_SIZE,
        ),
        decoder_config=CNN(
            layers=layers.cnn_small_transposed(out_channels=INPUT_SIZE[0])
        ),
        loss_config=Loss(
            funcs="reconstruct",
            weights=(10.0,)
        ),
        initial_weights="icdec_coil100-2qalvjco"
    ),
    calc_ofm=True,
)

cdec_uco_coil100 = Experiment(
    dataset_config=Dataset(name=DATASET_NAME),
    n_clusters=N_CLUSTERS,
    model_config=TFDEC(
        backbone_config=CNN(
            input_size=INPUT_SIZE,
        ),
        decoder_config=CNN(
            layers=layers.cnn_small_transposed(out_channels=INPUT_SIZE[0])
        ),
        loss_config=UCOLoss(
            funcs="uco",
            uco_kernel_type="tensor"
        ),
        initial_weights="icdec_coil100-2qalvjco",
        optimizer_config=Optimizer(learning_rate=1e-4),
    ),
    batch_size=200,
    calc_ofm=True,
)
