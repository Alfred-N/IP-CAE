"""
Includes modified code from
https://github.com/mfbalin/Concrete-Autoencoders/blob/master/experiments/generate_comparison_figures.py
"""

import io
import os
import pickle
import subprocess
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Tuple
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
import torch
import torchvision
from scipy.io import loadmat
from sklearn.datasets import fetch_california_housing, load_breast_cancer
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset


def load_and_strip(filename):
    with open(filename, "r") as file:
        lines = [line.strip().replace("  ", " ") for line in file.readlines()]
    file_content = "\n".join(lines)
    return io.StringIO(file_content)


def load_isolet(root):
    x_train = np.genfromtxt(
        os.path.join(root, "isolet/isolet1+2+3+4.data"),
        delimiter=",",
        usecols=range(0, 617),
        encoding="UTF-8",
    )
    y_train = np.genfromtxt(
        os.path.join(root, "isolet/isolet1+2+3+4.data"),
        delimiter=",",
        usecols=[617],
        encoding="UTF-8",
    )
    x_test = np.genfromtxt(
        os.path.join(root, "isolet/isolet5.data"),
        delimiter=",",
        usecols=range(0, 617),
        encoding="UTF-8",
    )
    y_test = np.genfromtxt(
        os.path.join(root, "isolet/isolet5.data"),
        delimiter=",",
        usecols=[617],
        encoding="UTF-8",
    )
    scaler = MinMaxScaler(feature_range=(0, 1)).fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    return x_train, y_train - 1, x_test, y_test - 1


def load_activity(root):
    x_train = np.loadtxt(
        load_and_strip(
            os.path.join(root, "activity/UCI HAR Dataset/train/X_train.txt")
        ),
        delimiter=" ",
        encoding="UTF-8",
    )
    x_test = np.loadtxt(
        load_and_strip(os.path.join(root, "activity/UCI HAR Dataset/test/X_test.txt")),
        delimiter=" ",
        encoding="UTF-8",
    )
    y_train = np.loadtxt(
        load_and_strip(
            os.path.join(root, "activity/UCI HAR Dataset/train/y_train.txt")
        ),
        delimiter=" ",
        encoding="UTF-8",
    )
    y_test = np.loadtxt(
        load_and_strip(os.path.join(root, "activity/UCI HAR Dataset/test/y_test.txt")),
        delimiter=" ",
        encoding="UTF-8",
    )
    scaler = MinMaxScaler(feature_range=(0, 1)).fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    return x_train, y_train - 1, x_test, y_test - 1


def download_zip(url, path):
    print(f"Downloading data from {url}")
    zip_path, _ = urlretrieve(url)
    print(f"Unzipping data to {path}")
    with zipfile.ZipFile(zip_path, "r") as f:
        f.extractall(path)


def download_activity(root):
    path = os.path.join(root, "activity")
    download_zip(
        "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip",
        path,
    )
    zip_path = os.path.join(path, "UCI HAR Dataset.zip")
    with zipfile.ZipFile(zip_path, "r") as f:
        f.extractall(path)


def download_isolet(root):
    path = os.path.join(root, "isolet")
    download_zip("https://archive.ics.uci.edu/static/public/54/isolet.zip", path)
    subprocess.run(["uncompress", os.path.join(path, "isolet1+2+3+4.data.Z")])
    subprocess.run(["uncompress", os.path.join(path, "isolet5.data.Z")])


def download_mice(root):
    path = os.path.join(root, "mice")
    download_zip(
        "https://archive.ics.uci.edu/static/public/342/mice+protein+expression.zip",
        path,
    )
    xls_path = os.path.join(root, "mice", "Data_Cortex_Nuclear.xls")
    df = pd.read_excel(xls_path)
    csv_path = os.path.join(root, "mice", "Data_Cortex_Nuclear.csv")
    df.to_csv(csv_path, index=False)


# Function adapted from scripts_misc/preprocessing/prepare_mice.py
def load_mice(root, split):
    from sklearn.model_selection import train_test_split

    array_names = ["X_train", "X_val", "X_test", "Y_train", "Y_val", "Y_test"]
    array_paths = [
        os.path.join(root, "mice", "tensors", _split + ".npy") for _split in array_names
    ]
    arrays_exist = [os.path.exists(_path) for _path in array_paths]
    scaler_filename = os.path.join(root, "mice", "scaler.pkl")

    if np.all(arrays_exist):
        X_train = np.load(array_paths[0])
        X_val = np.load(array_paths[1])
        X_test = np.load(array_paths[2])
        Y_train = np.load(array_paths[3])
        Y_val = np.load(array_paths[4])
        Y_test = np.load(array_paths[5])
        with open(scaler_filename, "rb") as scaler_file:
            scaler = pickle.load(scaler_file)
        print("Loaded preprocessed arrays")

    else:
        mice_file = os.path.join(root, "mice", "Data_Cortex_Nuclear.csv")
        if not os.path.exists(mice_file):
            download_mice(root=root)

        print("Processing mice")
        filling_value = -100000
        # ConcreteAE preprocessing
        X = np.genfromtxt(
            mice_file,
            delimiter=",",
            skip_header=1,
            usecols=range(1, 78),
            filling_values=filling_value,
            encoding="UTF-8",
        )
        classes = np.genfromtxt(
            mice_file,
            delimiter=",",
            skip_header=1,
            usecols=range(78, 81),
            dtype=None,
            encoding="UTF-8",
        )

        X_old = np.copy(X)  # added by Alfred
        for i, row in enumerate(X):
            for j, val in enumerate(row):
                if val == filling_value:
                    a = np.mean(
                        [
                            X_old[k, j]
                            for k in range(classes.shape[0])
                            if np.all(classes[i] == classes[k])
                            and X_old[k, j] != filling_value
                        ]
                    )  # added by me
                    # b = np.mean([X_old[k, j] for k in range(classes.shape[0]) if np.all(classes[i] == classes[k])]) #original, like CAE
                    X[i, j] = a

        DY = np.zeros((classes.shape[0]), dtype=np.uint8)
        for i, row in enumerate(classes):
            for j, (val, label) in enumerate(zip(row, ["Control", "Memantine", "C/S"])):
                DY[i] += (2**j) * (val == label)

        Y = np.zeros((DY.shape[0], np.unique(DY).shape[0]))
        for idx, val in enumerate(DY):
            Y[idx, val] = 1

        # N, D = X.shape
        print(f"data X,Y dimensions = {X.shape},{Y.shape}")
        print("Any Nans: ", np.any(X == filling_value))

        # Normalize using entire train set only
        X_train, X_valtest, Y_train, Y_valtest = train_test_split(
            X, Y, train_size=0.6, shuffle=True, random_state=42
        )
        X_val, X_test, Y_val, Y_test = train_test_split(
            X_valtest, Y_valtest, train_size=0.5, shuffle=True, random_state=42
        )

        scaler = MinMaxScaler(feature_range=(0, 1))
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

        print("Saving MICE data as numpy arrays")
        os.makedirs(os.path.join(root, "mice", "tensors"), exist_ok=True)
        for name, path in zip(array_names, array_paths):
            _arr = locals()[name]
            np.save(path, _arr)
        # Save the scaler to a file
        with open(scaler_filename, "wb") as scaler_file:
            pickle.dump(scaler, scaler_file)

    print(f"Train set X = {X_train.shape}, Y = {Y_train.shape}")
    print(f"Valid set X = {X_val.shape},   Y = {Y_val.shape}")
    print(f"Test  set X = {X_test.shape},  Y = {Y_test.shape}")

    mice_train = TensorDataset(
        torch.from_numpy(X_train).float(), torch.from_numpy(Y_train).long()
    )
    mice_val = TensorDataset(
        torch.from_numpy(X_val).float(), torch.from_numpy(Y_val).long()
    )
    mice_test = TensorDataset(
        torch.from_numpy(X_test).float(), torch.from_numpy(Y_test).long()
    )

    return (mice_train, mice_val, mice_test), scaler


class Activity(TensorDataset):
    def __init__(self, root, train=True, download=False):
        if download:
            download_activity(root)
        x_train, y_train, x_test, y_test = load_activity(root)
        NUM_CLASSES = 6
        y_train = torch.nn.functional.one_hot(
            torch.from_numpy(y_train).long(), num_classes=NUM_CLASSES
        )
        y_test = torch.nn.functional.one_hot(
            torch.from_numpy(y_test).long(), num_classes=NUM_CLASSES
        )
        if train:
            super().__init__(torch.tensor(x_train, dtype=torch.float32), y_train)
        else:
            super().__init__(torch.tensor(x_test, dtype=torch.float32), y_test)


class Isolet(TensorDataset):
    def __init__(self, root, train=True, download=False):
        if download:
            download_isolet(root)
        x_train, y_train, x_test, y_test = load_isolet(root)
        NUM_CLASSES = 26
        y_train = torch.nn.functional.one_hot(
            torch.from_numpy(y_train).long(), num_classes=NUM_CLASSES
        )
        y_test = torch.nn.functional.one_hot(
            torch.from_numpy(y_test).long(), num_classes=NUM_CLASSES
        )
        if train:
            super().__init__(torch.tensor(x_train, dtype=torch.float32), y_train)
        else:
            super().__init__(torch.tensor(x_test, dtype=torch.float32), y_test)


# TODO: include the datasets with negative label encoding again
skfeature_datasets = (
    "BASEHOCK",
    "PCMAC",
    "RELATHE",
    "COIL20",
    "ORL",
    "orlraws10P",
    "pixraw10P",
    "warpAR10P",
    "warpPIE10P",
    "Yale",
    "USPS",
    "ALLAML",
    "Carcinom",
    "CLL_SUB_111",
    # "colon",
    "GLI_85",
    "GLIOMA",
    # "leukemia",
    "lung",
    "lung_discrete",
    "lymphoma",
    "nci9",
    "Prostate_GE",
    "SMK_CAN_187",
    "TOX_171",
    # "arcene",
    # "gisette",
    "Isolet",
    # "madelon",
)

old_skfeature_datasets = (
    # MATLAB h5 formatted files. Not supported in current implementation
    # "20newsgroups",
    # "Reuters21578",
    "GLA-BRA-180",
)

all_names = skfeature_datasets + old_skfeature_datasets


def fetch_skfeature(name, folder, download=False):
    if name in skfeature_datasets:
        url = f"https://jundongl.github.io/scikit-feature/files/datasets/{name}.mat"
    elif name in old_skfeature_datasets:
        url = f"https://jundongl.github.io/scikit-feature/OLD/datasets/{name}.mat"
    else:
        raise ValueError(f"Unknown dataset '{name}'")
    path = Path(folder, name + ".mat")
    if not path.exists() and download:
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, path)
    data = loadmat(path)
    X = data["X"]
    Y = data["Y"]
    if name == "COIL20":
        Y = Y - 1
    Y = torch.nn.functional.one_hot(
        torch.tensor(Y, dtype=torch.int64).squeeze(-1)
    ).numpy()
    X = np.asarray(X, dtype=np.float32)
    return X, Y


class SkfeatureDataset(TensorDataset):
    def __init__(self, root, name, download, iscoil20=False):
        super().__init__(*map(torch.from_numpy, fetch_skfeature(name, root, download)))


class BASESHOCK(SkfeatureDataset):
    def __init__(self, root, download=False):
        super().__init__(root, "BASESHOCK", download)


class PCMAC(SkfeatureDataset):
    def __init__(self, root, download=False):
        super().__init__(root, "PCMAC", download)


class RELATHE(SkfeatureDataset):
    def __init__(self, root, download=False):
        super().__init__(root, "RELATHE", download)


class COIL20(SkfeatureDataset):
    def __init__(self, root, download=False):
        super().__init__(root, "COIL20", download)


class ORL(SkfeatureDataset):
    def __init__(self, root, download=False):
        super().__init__(root, "ORL", download)


class OrlRaws10P(SkfeatureDataset):
    def __init__(self, root, download=False):
        super().__init__(root, "orlraws10P", download)


class PixRaws10P(SkfeatureDataset):
    def __init__(self, root, download=False):
        super().__init__(root, "pixraw10P", download)


class WarpAR10P(SkfeatureDataset):
    def __init__(self, root, download=False):
        super().__init__(root, "warpAR10P", download)


class WarpPIE10P(SkfeatureDataset):
    def __init__(self, root, download=False):
        super().__init__(root, "warpPIE10P", download)


class Yale(SkfeatureDataset):
    def __init__(self, root, download=False):
        super().__init__(root, "Yale", download)


class USPS(SkfeatureDataset):
    def __init__(self, root, download=False):
        super().__init__(root, "USPS", download)


class ALLAML(SkfeatureDataset):
    def __init__(self, root, download=False):
        super().__init__(root, "ALLAML", download)


class Carcinom(SkfeatureDataset):
    def __init__(self, root, download=False):
        super().__init__(root, "Carcinom", download)


class CLL_SUB_111(SkfeatureDataset):
    def __init__(self, root, download=False):
        super().__init__(root, "CLL_SUB_111", download)


class Colon(SkfeatureDataset):
    def __init__(self, root, download=False):
        super().__init__(root, "colon", download)


class GLI_85(SkfeatureDataset):
    def __init__(self, root, download=False):
        super().__init__(root, "GLI_85", download)


class GLIOMA(SkfeatureDataset):
    def __init__(self, root, download=False):
        super().__init__(root, "GLIOMA", download)


class Leukemia(SkfeatureDataset):
    def __init__(self, root, download=False):
        super().__init__(root, "Leukemia", download)


class Lung(SkfeatureDataset):
    def __init__(self, root, download=False):
        super().__init__(root, "Lung", download)


class LungDiscrete(SkfeatureDataset):
    def __init__(self, root, download=False):
        super().__init__(root, "lung_discrete", download)


class Lymphoma(SkfeatureDataset):
    def __init__(self, root, download=False):
        super().__init__(root, "Lymphoma", download)


class Nci9(SkfeatureDataset):
    def __init__(self, root, download=False):
        super().__init__(root, "nci9", download)


class ProstateGE(SkfeatureDataset):
    def __init__(self, root, download=False):
        super().__init__(root, "Prostate_GE", download)


class SMK_CAN_187(SkfeatureDataset):
    def __init__(self, root, download=False):
        super().__init__(root, "SMK_CAN_187", download)


class TOX_171(SkfeatureDataset):
    def __init__(self, root, download=False):
        super().__init__(root, "TOX_171", download)


class Arcene(SkfeatureDataset):
    def __init__(self, root, download=False):
        super().__init__(root, "arcene", download)


class Gisette(SkfeatureDataset):
    def __init__(self, root, download=False):
        super().__init__(root, "gisette", download)


class Madelon(SkfeatureDataset):
    def __init__(self, root, download=False):
        super().__init__(root, "madelon", download)


class CustomMNIST(torchvision.datasets.MNIST):
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        data, label = super().__getitem__(index)
        return data, torch.nn.functional.one_hot(
            torch.tensor(label, dtype=torch.int64), num_classes=10
        )


class CustomFashionMNIST(torchvision.datasets.FashionMNIST):
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        data, label = super().__getitem__(index)
        return data, torch.nn.functional.one_hot(
            torch.tensor(label, dtype=torch.int64), num_classes=10
        )


class BreastCancer(TensorDataset):
    def __init__(self):
        X, y = load_breast_cancer(return_X_y=True)
        NUM_CLASSES = 2
        y = torch.nn.functional.one_hot(
            torch.from_numpy(y).long(), num_classes=NUM_CLASSES
        )
        super().__init__(torch.tensor(X, dtype=torch.float32), y)


# -------------------------------------------------------------------------------------------------------------------------------------
# Functions for creating fixed train-valid-test splits


def get_subset_mnist_split(root, split, transform=None, target_transform=None):
    assert split in ["train", "valid", "test"], f"Invalid split: {split}"

    if split == "train" or split == "valid":
        mnist_train = CustomMNIST(
            root=root,
            train=True,
            transform=transform,
            target_transform=target_transform,
            download=True,
        )
        N = len(mnist_train)
        mnist_train, _ = torch.utils.data.random_split(
            mnist_train,
            [int(0.15 * N), int(0.85 * N)],
            generator=torch.Generator().manual_seed(42),
        )
        N = len(mnist_train)
        mnist_train, mnist_val = torch.utils.data.random_split(
            mnist_train,
            [int(0.8 * N), int(0.2 * N)],
            generator=torch.Generator().manual_seed(42),
        )

        if split == "train":
            return mnist_train
        else:
            return mnist_val

    elif split == "test":
        mnist_test = CustomMNIST(
            root=root,
            train=False,
            transform=transform,
            target_transform=target_transform,
            download=True,
        )
        N = len(mnist_test)
        mnist_test, _ = torch.utils.data.random_split(
            mnist_test,
            [int(0.15 * N), int(0.85 * N)],
            generator=torch.Generator().manual_seed(42),
        )
        return mnist_test

    else:
        raise ValueError


def get_subset_mnist_fashion_split(root, split, transform=None, target_transform=None):
    assert split in ["train", "valid", "test"], f"Invalid split: {split}"

    if split == "train" or split == "valid":
        mnistF_train = CustomFashionMNIST(
            root=root,
            train=True,
            transform=transform,
            target_transform=target_transform,
            download=True,
        )
        N = len(mnistF_train)
        mnistF_train, _ = torch.utils.data.random_split(
            mnistF_train,
            [int(0.15 * N), int(0.85 * N)],
            generator=torch.Generator().manual_seed(42),
        )
        N = len(mnistF_train)
        mnistF_train, mnistF_val = torch.utils.data.random_split(
            mnistF_train,
            [int(0.8 * N), int(0.2 * N)],
            generator=torch.Generator().manual_seed(42),
        )

        if split == "train":
            return mnistF_train
        else:
            return mnistF_val

    elif split == "test":
        mnistF_test = CustomFashionMNIST(
            root=root,
            train=False,
            transform=transform,
            target_transform=target_transform,
            download=True,
        )
        N = len(mnistF_test)
        mnistF_test, _ = torch.utils.data.random_split(
            mnistF_test,
            [int(0.15 * N), int(0.85 * N)],
            generator=torch.Generator().manual_seed(42),
        )
        return mnistF_test

    else:
        raise ValueError


def get_isolet_split(root, split, transform=None, target_transform=None):
    assert split in ["train", "valid", "test"], f"Invalid split: {split}"

    if split == "train" or split == "valid":
        isolet_train = Isolet(root=root, download=False, train=True)
        N = len(isolet_train)
        l1 = int(N * 0.8)
        l2 = N - l1
        isolet_train, isolet_val = torch.utils.data.random_split(
            isolet_train, [l1, l2], generator=torch.Generator().manual_seed(42)
        )

        if split == "train":
            return isolet_train
        else:
            return isolet_val

    elif split == "test":
        isolet_test = Isolet(root=root, download=False, train=False)
        return isolet_test


def get_coil20_split(root, split, transform=None, target_transform=None):
    assert split in ["train", "valid", "test"], f"Invalid split: {split}"

    coil20_full = COIL20(root=root, download=True)
    N = len(coil20_full)
    l1 = int(0.6 * N)
    l2 = int(0.2 * N)
    l3 = N - l1 - l2
    coil20_train, coil20_val, coil20_test = torch.utils.data.random_split(
        coil20_full, [l1, l2, l3], generator=torch.Generator().manual_seed(42)
    )

    if split == "train":
        return coil20_train

    elif split == "valid":
        return coil20_val

    elif split == "test":
        return coil20_test

    else:
        raise ValueError


def get_activity_split(root, split, transform=None, target_transform=None):
    assert split in ["train", "valid", "test"], f"Invalid split: {split}"

    if split == "train" or split == "valid":
        activity_train = Activity(root=root, download=False, train=True)
        N = len(activity_train)
        l1 = int(0.8 * N)
        l2 = N - l1
        activity_train, activity_val = torch.utils.data.random_split(
            activity_train, [l1, l2], generator=torch.Generator().manual_seed(42)
        )

        if split == "train":
            return activity_train
        else:
            return activity_val

    elif split == "test":
        activity_test = Activity(root=root, download=False, train=False)
        return activity_test


def get_mice_split(root, split, transform=None, target_transform=None):
    (mice_train, mice_val, mice_test), scaler = load_mice(root=root, split=split)

    if split == "train":
        return mice_train
    elif split == "valid":
        return mice_val
    elif split == "test":
        return mice_test
    else:
        return ValueError


def get_breast_cancer_split(root, split, transform=None, target_transform=None):
    assert split in ["train", "valid", "test"], f"Invalid split: {split}"

    full = BreastCancer()
    N = len(full)
    l1 = int(0.6 * N)
    l2 = int(0.2 * N)
    l3 = N - l1 - l2
    train, val, test = torch.utils.data.random_split(
        full, [l1, l2, l3], generator=torch.Generator().manual_seed(42)
    )

    if split == "train":
        return train

    elif split == "valid":
        return val

    elif split == "test":
        return test

    else:
        raise ValueError


def get_dataset_split(dataset, root, split, transform=None, target_transform=None):
    # the passed transform and target_transform only works for mnist and mnist_fashion, for other datasets it is ignored
    D = {
        "mnist": get_subset_mnist_split,
        "mnist_fashion": get_subset_mnist_fashion_split,
        "isolet": get_isolet_split,
        "coil20": get_coil20_split,
        "activity": get_activity_split,
        "mice": get_mice_split,
        "cancer": get_breast_cancer_split,
    }

    return D[dataset](
        root=root, split=split, transform=transform, target_transform=target_transform
    )


# -------------------------------------------------------------------------------------------------------------------------------------


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Download all datasets (except MNIST & MNIST-Fashion) to the specified data folder"
    )
    parser.add_argument("--data_folder", required=True)
    args = parser.parse_args()

    path = os.path.expanduser(args.data_folder)
    for name in all_names:
        print(f"Downloading {name}")
        X, y = fetch_skfeature(name, path, download=True)
    print("Downloading activity")
    Activity(path, download=True)
    print("Downloading ISOLET")
    # Isolet(path, download=True)
    print("Downloading mice protein")
    download_mice(path)
    # import torchvision
    # transform = torchvision.transforms.ToTensor()
    # mnist_fashion_train = get_subset_mnist_fashion_split("data_CAE", split="train", transform=transform)
    # from utils import get_dataset_mean_std
    # print(get_dataset_mean_std(mnist_fashion_train))
