from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile

import numpy as np
import torch
from torch.utils.data import TensorDataset

urls = {
    "arcene": "https://archive.ics.uci.edu/static/public/167/arcene.zip",
    "dexter": "https://archive.ics.uci.edu/static/public/168/dexter.zip",
    "dorothea": "https://archive.ics.uci.edu/static/public/169/dorothea.zip",
    "gisette": "https://archive.ics.uci.edu/static/public/170/gisette.zip",
    "madelon": "https://archive.ics.uci.edu/static/public/171/madelon.zip",
}


class Arcene(TensorDataset):

    def __init__(self, root, split="train", download=False):
        x, y = map(torch.from_numpy, fetch_arcene(root, split, download))
        super().__init__(x, y.unsqueeze(1))


class Dexter(TensorDataset):

    def __init__(self, root, split="train", download=False):
        x, y = map(torch.from_numpy, fetch_dexter(root, split, download))
        super().__init__(x, y.unsqueeze(1))


def download_zip(url, path):
    zip_path, _ = urlretrieve(url)
    with ZipFile(zip_path, "r") as f:
        f.extractall(path)


def load_sparse_format(path, size, features, binary=False):
    X = np.zeros((size, features))
    with open(path) as f:
        for i, line in enumerate(f.readlines()):
            for entry in line.strip().split(" "):
                if binary:
                    feature = int(entry.strip())
                    value = 1
                else:
                    feature, value = map(int, entry.strip().split(":"))
                X[i, feature - 1] = value
    return X


def fetch_nips2003(root, dataset, split="train", download=False, load="dense"):
    path = Path(root, dataset).expanduser()
    if not path.exists() and download:
        download_zip(urls[dataset], path)

    if split == "train":
        X = np.loadtxt(path.joinpath(f"{dataset.upper()}/{dataset}_train.data"))
        y = np.loadtxt(path.joinpath(f"{dataset.upper()}/{dataset}_train.labels"))
    elif split == "valid":
        X = np.loadtxt(path.joinpath(f"{dataset.upper()}/{dataset}_valid.data"))
        y = np.loadtxt(path.joinpath(f"{dataset}_valid.labels"))
    elif split == "test":
        X = np.loadtxt(path.joinpath(f"{dataset.upper()}/{dataset}_test.data"))
        y = np.ones_like(X) * np.inf
    else:
        raise ValueError("split must be one of 'train', 'valid', or 'test'")
    y = np.maximum(0, y)  # -1 -> 0
    return X.astype(np.float32), y.astype(np.float32)


def fetch_arcene(root, split="train", download=False):
    return fetch_nips2003(root, "arcene", split, download)


def fetch_dexter(root, split="train", download=False):
    path = Path(root, "dexter").expanduser()
    if not path.exists() and download:
        download_zip(urls["dexter"], path)

    if split == "train":
        X = load_sparse_format(path.joinpath(f"DEXTER/dexter_train.data"), 300, 20_000)
        y = np.loadtxt(path.joinpath(f"DEXTER/dexter_train.labels"))
    elif split == "valid":
        X = load_sparse_format(path.joinpath(f"DEXTER/dexter_valid.data"), 300, 20_000)
        y = np.loadtxt(path.joinpath(f"dexter_valid.labels"))
    elif split == "test":
        X = load_sparse_format(path.joinpath(f"DEXTER/dexter_test.data"), 2000, 20_000)
        y = np.ones_like(X) * np.inf
    else:
        raise ValueError("split must be one of 'train', 'valid', or 'test'")
    y = np.maximum(0, y)  # -1 -> 0
    return X.astype(np.float32), y.astype(np.float32)


def fetch_dorothea(root, split="train", download=False):
    path = Path(root, "dorothea").expanduser()
    if not path.exists() and download:
        download_zip(urls["dorothea"], path)

    if split == "train":
        X = load_sparse_format(
            path.joinpath(f"DOROTHEA/dorothea_train.data"), 800, 100_000, binary=True
        )
        y = np.loadtxt(path.joinpath(f"DOROTHEA/dorothea_train.labels"))
    elif split == "valid":
        X = load_sparse_format(
            path.joinpath(f"DOROTHEA/dorothea_valid.data"), 350, 100_000, binary=True
        )
        y = np.loadtxt(path.joinpath(f"dorothea_valid.labels"))
    elif split == "test":
        X = load_sparse_format(
            path.joinpath(f"dorothea_test.data"), 800, 100_000, binary=True
        )
        y = np.ones_like(X) * np.inf
    else:
        raise ValueError("split must be one of 'train', 'valid', or 'test'")
    y = np.maximum(0, y)  # -1 -> 0
    return X.astype(np.float32), y.astype(np.float32)


def fetch_gisette(root, split="train", download=False):
    return fetch_nips2003(root, "gisette", split, download)


def fetch_madelon(root, split="train", download=False):
    return fetch_nips2003(root, "madelon", split, download)


if __name__ == "__main__":
    X, y = fetch_arcene("data", download=True)
    print("arcene", X.shape, y.shape)

    X, y = fetch_dexter("data", download=True)
    print("dexter", X.shape, y.shape)

    X, y = fetch_dorothea("data", download=True)
    print("dorothea", X.shape, y.shape)

    X, y = fetch_gisette("data", download=True)
    print("gisette", X.shape, y.shape)

    X, y = fetch_madelon("data", download=True)
    print("madelon", X.shape, y.shape)

    dataset = Arcene("data", split="valid", download=True)
    print(len(dataset))
