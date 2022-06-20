import warnings
import numpy as np
import tensorflow as tf

from data.data_module import load_dataset


def create_tf_dataset(cfg, set_type):
    cfg_dict = cfg.dict()
    dataset = load_dataset(
        name=cfg.name,
        set_type=set_type,
        n_samples=cfg_dict[f"n_{set_type}_samples"],
        select_labels=cfg.select_labels,
        label_counts=cfg_dict[f"{set_type}_label_counts"],
        to_dataset=False
    )

    if dataset is None:
        return None

    data, labels = dataset

    if cfg.initial_shuffle:
        inds = np.random.permutation(data.shape[0])
        data, labels = data[inds], labels[inds]

    if data.ndim == 4:
        # Convert to "channels last" format
        data = np.transpose(data, axes=(0, 2, 3, 1))

    idx = np.arange(data.shape[0])
    dataset = tf.data.Dataset.from_tensor_slices((
        {"data": data, "idx": idx},
        {"output_1": labels}
    ))
    return dataset


class TFDataModule:
    def __init__(self, cfg):
        self.cfg = cfg
        self.n_batches = None

        self.train_dataset = create_tf_dataset(cfg, "train")
        self.val_dataset = create_tf_dataset(cfg, "val") or self.train_dataset
        self.test_dataset = create_tf_dataset(cfg, "test") or self.train_dataset

        self.n_samples = self.train_dataset.cardinality().numpy()
        self.n_batches = self.n_samples // cfg.batch_size

        if not cfg.shuffle_first:
            warnings.warn("Dataset config has shuffle_first=False. This will result in the same batching at each "
                          "training epoch")

        self.batched_train_dataset = self.batch_and_shuffle(self.train_dataset, shuffle_first=cfg.shuffle_first,
                                                            drop_remainder=cfg.drop_last)
        self.batched_val_dataset = self.batch_and_shuffle(self.val_dataset, shuffle_first=cfg.shuffle_first,
                                                          drop_remainder=False)
        self.batched_test_dataset = self.batch_and_shuffle(self.test_dataset, shuffle_first=cfg.shuffle_first,
                                                           drop_remainder=False)

    def batch_and_shuffle(self, ds, shuffle_first=True, drop_remainder=False):
        if shuffle_first:
            out = ds.shuffle(self.n_samples).batch(self.cfg.batch_size, drop_remainder=drop_remainder)
        else:
            out = ds.batch(self.cfg.batch_size, drop_remainder=drop_remainder).shuffle(self.n_samples)
        return out

    def as_numpy_arrays(self, ds, assert_ordered=True):
        data = np.empty((self.n_samples, *tuple(self.train_dataset.element_spec[0]["data"].shape)))
        labels = np.empty(self.n_samples)
        idx = np.empty(self.n_samples)
        for i, (inputs, targets) in enumerate(ds):
            data[i] = inputs["data"]
            idx[i] = inputs["idx"]
            labels[i] = targets["output_1"]

        if assert_ordered:
            assert (idx == np.arange(self.n_samples)).all()

        return data, idx, labels

