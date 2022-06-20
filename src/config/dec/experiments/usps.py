import numpy as np
from config import layers
from config.defaults import MLP, Optimizer, CNN
from config.dec.defaults import Experiment, DEC, Loss, ClusteringModule, Dataset, UCOLoss, TFDEC, Optimizer


DATASET_NAME = "usps"
N_CLUSTERS = 10
INPUT_SIZE = (1, 32, 32)

ENCODER_LAYERS = (500, 500, "out", 2000, "out")
ENCODER_ACTIVATION = ("relu", "relu", None, "relu", None)
DECODER_LAYERS = (500, 500, np.prod(INPUT_SIZE))
DECODER_ACTIVATION = ("relu", "relu", None)


dec_usps = Experiment(
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
        initial_weights="idec_usps-3ky15dnj"
    ),
)

idec_usps = Experiment(
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
        initial_weights="idec_usps-3ky15dnj"
    ),
    calc_ofm=True,
    
)

dec_uco_usps = Experiment(
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
        initial_weights="idec_usps-3ky15dnj",
        optimizer_config=Optimizer(learning_rate=1e-4),
    ),
    calc_ofm=True,
)


cdec_usps = Experiment(
    dataset_config=Dataset(name=DATASET_NAME),
    n_clusters=N_CLUSTERS,
    model_config=TFDEC(
        backbone_config=CNN(
            input_size=INPUT_SIZE,
        ),
        decoder_config=CNN(
            layers=layers.cnn_small_transposed(out_channels=INPUT_SIZE[0])
        ),
        initial_weights="icdec_usps-h47d9bjr"
    ),
)

icdec_usps = Experiment(
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
        initial_weights="icdec_usps-h47d9bjr"
    ),
    calc_ofm=True,
)

cdec_uco_usps = Experiment(
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
        initial_weights="icdec_usps-h47d9bjr",
        optimizer_config=Optimizer(learning_rate=1e-4),
    ),
    calc_ofm=True,
)
