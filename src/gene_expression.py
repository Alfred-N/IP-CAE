import subprocess
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import TensorDataset, random_split


class GeneExpression(TensorDataset):

    def __init__(self, root, download=False):
        x, y = map(torch.from_numpy, gene_expression(root, download))
        super().__init__(x, y.unsqueeze(1))


def gene_expression(path, download=False, split=False):
    path = Path(path).expanduser()
    folder = path / "TCGA-PANCAN-HiSeq-801x20531"

    if not folder.exists() and download:
        url = "https://archive.ics.uci.edu/static/public/401/gene+expression+cancer+rna+seq.zip"
        zipfile = path / "gene+expression+cancer+rna+seq.zip"
        tarfile = path / "TCGA-PANCAN-HiSeq-801x20531.tar.gz"
        subprocess.run(["wget", url, "-P", path])
        subprocess.run(["unzip", zipfile, "-d", path])
        subprocess.run(["tar", "-xf", tarfile, "-C", path])
        subprocess.run(["rm", zipfile, tarfile])

    X = np.loadtxt(
        folder / "data.csv",
        delimiter=",",
        skiprows=1,
        usecols=range(1, 20532),
        dtype=np.float32,
    )
    y = np.loadtxt(
        folder / "labels.csv", delimiter=",", skiprows=1, usecols=1, dtype=str
    )
    y = LabelEncoder().fit_transform(y)
    return X, y


if __name__ == "__main__":
    X, y = gene_expression("~/datasets", download=True)
    print(X.shape, y.shape)
