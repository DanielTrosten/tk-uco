import os
import numpy as np
import torch as th
import torchvision
import torchvision.transforms as transforms
from skimage.io import imread
from skimage.transform import resize

import config

RAW_DIR = config.DATA_DIR / "raw"


def _torchvision_dataset(dataset_class, means, stds, splits=None):
    img_transforms = [transforms.ToTensor(), transforms.Normalize(means, stds)]
    transform = transforms.Compose(img_transforms)

    if splits is None:
        datasets = [
            dataset_class(root=config.DATA_DIR / "raw", train=True, download=True, transform=transform),
            dataset_class(root=config.DATA_DIR / "raw", train=False, download=True, transform=transform),
        ]
    else:
        datasets = [dataset_class(root=config.DATA_DIR / "raw", split=split, download=True, transform=transform)
                    for split in splits]

    dataset = th.utils.data.ConcatDataset(datasets)
    loader = th.utils.data.DataLoader(dataset, batch_size=len(dataset))
    data, labels = list(loader)[0]
    return data, labels


def mnist():
    return _torchvision_dataset(torchvision.datasets.MNIST, means=(0.5,), stds=(0.5,))


def fmnist():
    return _torchvision_dataset(torchvision.datasets.FashionMNIST, means=(0.5,), stds=(0.5,))


def usps():
    data, labels = _torchvision_dataset(torchvision.datasets.USPS, means=(0.5,), stds=(0.5,))
    data = np.stack([resize(img[0], output_shape=(32, 32))[None, :, :] for img in data], axis=0)
    return data, labels


def coil100():
    full_dir = RAW_DIR / "coil100"
    data, labels = [], []

    for fname in os.listdir(full_dir):
        if ".png" not in fname:
            continue

        label = int(fname.split("__")[0].strip("obj")) - 1
        img = imread(os.path.join(full_dir, fname))
        img = resize(img, output_shape=(64, 64, 3))
        labels.append(label)
        data.append(img)

    data = np.stack(data, axis=0)
    data = np.transpose(data, (0, 3, 1, 2))
    labels = np.array(labels)
    return data, labels
