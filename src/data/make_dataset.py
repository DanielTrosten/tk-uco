import os
import sys
import numpy as np
from tabulate import tabulate
from sklearn.datasets import make_blobs

import config
from data import image_exporters

RAW_DIR = config.DATA_DIR / "raw"


def print_summary(name, data, labels, file_path):
    uniq, count = np.unique(labels, return_counts=True)
    rows = [
        ["name", name],
        ["file_path", file_path],
        ["data.shape", data.shape],
        ["labels.shape", labels.shape],
        ["n_clusters", len(uniq)],
        ["unique labels", " ".join([str(u) for u in uniq])],
        ["label counts", " ".join([str(c) for c in count])]
    ]
    print(tabulate(rows), end="\n\n")


def export_dataset(name, data, labels):
    assert labels.shape[0] == data.shape[0]
    assert labels.ndim == 1
    assert data.ndim in (2, 3, 4)
    if data.ndim == 4:
        assert data.shape[1] in (1, 3)

    processed_dir = config.DATA_DIR / "processed"
    os.makedirs(processed_dir, exist_ok=True)
    file_path = processed_dir / f"{name}.npz"
    np.savez(file_path, labels=labels, data=data)
    print_summary(name, data, labels, file_path)


def blobs():
    nc = 1000
    ndim = 2
    k = 3
    data, labels = make_blobs(n_samples=k * [nc], n_features=ndim, cluster_std=1.0, shuffle=False)
    return data, labels


LOADERS = {
    "blobs": blobs,
    "mnist": image_exporters.mnist,
    "fmnist": image_exporters.fmnist,
    "usps": image_exporters.usps,
    "coil100": image_exporters.coil100,
}


def main():
    export_sets = sys.argv[1:] if len(sys.argv) > 1 else LOADERS.keys()

    for name in export_sets:
        print(f"Exporting dataset '{name}'")
        data, labels = LOADERS[name]()
        export_dataset(name=name + "_train", data=data, labels=labels)


if __name__ == '__main__':
    main()
