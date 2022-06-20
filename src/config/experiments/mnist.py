from config import layers
from config.defaults import DDCModel, Experiment, Loss, UCOLoss, Dataset, MLP, DDC, CNN, DDCUCO, DDCAE, TFDDCAE, Optimizer

N_CLUSTERS = 10
INPUT_SIZE = (1, 28, 28)
DATASET_NAME = "mnist"


ddc_mnist = Experiment(
    dataset_config=Dataset(name=DATASET_NAME),
    n_clusters=N_CLUSTERS,
    model_config=DDCUCO(
        backbone_config=CNN(
            input_size=INPUT_SIZE,
        ),
        cm_config=DDC(),
        loss_config=Loss(),
        initial_weights="ddc_ae_mnist-3rv1c8qe"
    ),
)

ddc_uco_mnist = Experiment(
    dataset_config=Dataset(name=DATASET_NAME),
    n_clusters=N_CLUSTERS,
    model_config=DDCUCO(
        backbone_config=CNN(
            input_size=INPUT_SIZE,
        ),
        cm_config=DDC(),
        loss_config=UCOLoss(),
        initial_weights="ddc_ae_mnist-3rv1c8qe",
        optimizer_config=Optimizer(learning_rate=1e-4),
    ),
    calc_ofm=True,
)

ddc_ae_mnist = Experiment(
    dataset_config=Dataset(name=DATASET_NAME),
    n_clusters=N_CLUSTERS,
    model_config=TFDDCAE(
        backbone_config=CNN(
            input_size=INPUT_SIZE,
        ),
        decoder_config=CNN(
            layers=layers.cnn_small_transposed(out_channels=INPUT_SIZE[0]),
        ),
        cm_config=DDC(),
        loss_config=Loss(
            funcs="ddc_1|ddc_2|ddc_3|reconstruct",
            weights=(1, 1, 1, 0.1)
        ),
        initial_weights="ddc_ae_mnist-3rv1c8qe",
    ),
    calc_ofm=True,
)
