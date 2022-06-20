import os
import numpy as np
import torch as th
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset

import config


def _load_npz(name, set_type):
    fname = config.DATA_DIR / "processed" / f"{name}_{set_type}.npz"
    if os.path.exists(fname):
        return np.load(fname)
    return None


def _fix_labels(l):
    uniq = np.unique(l)[None, :]
    new = (l[:, None] == uniq).argmax(axis=1)
    return new


def load_dataset(name, set_type, n_samples=None, select_labels=None, label_counts=None, to_dataset=True,
                 random_seed=None, **kwargs):

    if random_seed is not None:
        np.random.seed(random_seed)

    npz = _load_npz(name, set_type)
    if npz is None:
        return

    labels = npz["labels"]
    data = npz["data"]

    if select_labels is not None:
        mask = np.isin(labels, select_labels)
        labels = labels[mask]
        data = data[mask]
        labels = _fix_labels(labels)

    if label_counts is not None:
        idx = []
        unique_labels = np.unique(labels)
        assert len(unique_labels) == len(label_counts)
        for l, n in zip(unique_labels, label_counts):
            _idx = np.random.choice(np.where(labels == l)[0], size=n, replace=False)
            idx.append(_idx)

        idx = np.concatenate(idx, axis=0)
        labels = labels[idx]
        data = data[idx]

    if n_samples is not None:
        idx = np.random.choice(labels.shape[0], size=min(labels.shape[0], int(n_samples)), replace=False)
        labels = labels[idx]
        data = data[idx]

    data = data.astype(np.float32)
    if to_dataset:
        dataset = TensorDataset(th.Tensor(data), th.Tensor(labels), th.tensor(np.arange(data.shape[0])))
    else:
        dataset = (data, labels)
    return dataset


class DataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super(DataModule, self).__init__()
        self.cfg = cfg
        self.n_batches = None
        self.train_dataset = self.val_dataset = self.test_dataset = None

        self.train_dataset = load_dataset(name=cfg.name, set_type="train", n_samples=cfg.n_train_samples,
                                          select_labels=cfg.select_labels, label_counts=cfg.train_label_counts,
                                          to_dataset=True)
        self.val_dataset = load_dataset(name=cfg.name, set_type="val", n_samples=cfg.n_val_samples,
                                        select_labels=cfg.select_labels, label_counts=cfg.val_label_counts,
                                        to_dataset=True)
        self.val_dataset = self.val_dataset or self.train_dataset
        self.test_dataset = load_dataset(name=cfg.name, set_type="test", n_samples=cfg.n_test_samples,
                                         select_labels=cfg.select_labels, label_counts=cfg.test_label_counts,
                                         to_dataset=True)
        self.test_dataset = self.test_dataset or self.train_dataset
        self.n_batches = len(self.train_dataset) // cfg.batch_size

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.cfg.batch_size, shuffle=True,
                          drop_last=self.cfg.drop_last, num_workers=config.DATALODER_WORKERS, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.cfg.batch_size, shuffle=True, drop_last=False,
                          num_workers=config.DATALODER_WORKERS)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.cfg.batch_size, shuffle=True, drop_last=False,
                          num_workers=config.DATALODER_WORKERS)
