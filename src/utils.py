import matplotlib.pyplot as plt
import matplotlib
import os
from functools import partial
import torch
from torch.nn.parallel import DistributedDataParallel
import torch.nn as nn
from torch._six import inf
import pandas as pd
import torchvision.transforms as transforms
import yaml
import numpy as np
import fs_datasets

matplotlib.use("Agg")


def _sample_binarize(x):
    return torch.distributions.Bernoulli(probs=x).sample()


def _binarize(x, threshold=0.5):
    x[x >= threshold] = 1
    x[x < threshold] = 0
    return x


TRANSFORMS_DICT = {
    "mnist": {
        "train": lambda _: transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=[0.1306], std=[0.3080])]
        ),
        "test": lambda _: transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=[0.1306], std=[0.3080])]
        ),
    },
    "mnist_fashion": {
        "train": lambda _: transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=[0.2873], std=[0.3529])]
        ),
        "test": lambda _: transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=[0.2873], std=[0.3529])]
        ),
    },
}


def get_dataset(dataset_str, root, input_size=None):
    transform_train = None
    transform_val = None
    transform_test = None
    if dataset_str == "mnist" or dataset_str == "mnist_fashion":
        transform_train = TRANSFORMS_DICT[dataset_str]["train"](input_size)
        transform_val = TRANSFORMS_DICT[dataset_str]["test"](input_size)
        transform_test = TRANSFORMS_DICT[dataset_str]["test"](input_size)

    dataset_train = fs_datasets.get_dataset_split(
        dataset_str, root, "train", transform_train
    )
    dataset_val = fs_datasets.get_dataset_split(
        dataset_str, root, "valid", transform_val
    )
    dataset_test = fs_datasets.get_dataset_split(
        dataset_str, root, "test", transform_test
    )
    return (dataset_train, dataset_val, dataset_test)


def get_dataset_mean_std(dataset, num_workers=0):
    from torch.utils.data import DataLoader

    dataloader = DataLoader(dataset, batch_size=1, num_workers=num_workers)
    from tqdm import tqdm

    probe_im, _ = next(iter(dataloader))
    channels = probe_im.shape[1]
    mins = torch.tensor([torch.inf] * channels)
    maxes = torch.tensor([-torch.inf] * channels)
    sum_ = 0
    sq_sum = 0
    count = 0
    for im, _ in tqdm(dataloader):
        # im = im.squeeze(0) numbers
        dims = im.shape
        sum_ += torch.sum(im, dim=(0, 2, 3))
        sq_sum += torch.sum(torch.pow(im, 2), dim=(0, 2, 3))
        count += im[:, 0, :].flatten().shape[0]
        mins = torch.minimum(mins, torch.amin(im, dim=(0, 2, 3)))
        maxes = torch.maximum(maxes, torch.amax(im, dim=(0, 2, 3)))
    print(dims)
    mean = sum_ / count
    sq_mean = sq_sum / count
    std = torch.sqrt(sq_mean - torch.pow(mean, 2))
    return mean, std, (maxes, mins)


def get_num_parameters(model):
    sum = 0
    for param in list(model.parameters()):
        sum += param.numel()
    return sum


def get_rank() -> int:
    rank_keys = ("RANK", "SLURM_PROCID", "LOCAL_RANK")
    for key in rank_keys:
        rank = os.environ.get(key)
        if rank is not None:
            return int(rank)
    return 0


def create_sparse_grid(side_length=28, k=50):
    # Determine the dimensions of the grid
    n_rows = int(np.ceil(np.sqrt(k)))
    n_cols = int(np.ceil(k / n_rows))

    # Initialize an empty image
    image = np.zeros((side_length, side_length), dtype=int)

    # Calculate spacing based on the more filled dimension
    if n_cols > n_rows:
        spacing = side_length / n_cols
        n_filled_rows = min(n_rows, int(np.ceil(k / n_cols)))
    else:
        spacing = side_length / n_rows
        n_filled_rows = n_rows

    # Calculate offsets to center the grid
    offset_x = (side_length - (spacing * n_cols)) / 2
    offset_y = (side_length - (spacing * n_filled_rows)) / 2

    # Populate the grid
    for i in range(k):
        row = int(i / n_cols)
        col = i % n_cols
        x = int(offset_y + row * spacing)
        y = int(offset_x + col * spacing)

        if x < side_length and y < side_length:
            image[x, y] = 1

    return image
