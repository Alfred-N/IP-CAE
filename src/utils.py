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

    dataset_train = fs_datasets.get_dataset_split(dataset_str, root, "train", transform_train)
    dataset_val = fs_datasets.get_dataset_split(dataset_str, root, "valid", transform_val)
    dataset_test = fs_datasets.get_dataset_split(dataset_str, root, "test", transform_test)
    return (dataset_train, dataset_val, dataset_test)


def plot_distribution(
    pi_raw: torch.tensor,
    base_path: str,
    base_name: str,
    num_observed_patches: int,
    n_patch_per_side=14,
):
    save_name = "pi_marginal" + base_name + ".png"
    save_path = os.path.join(base_path, save_name)
    print(
        "Plotting distribution--",
        save_path,
    )
    plt.matshow(pi_raw.numpy().reshape(n_patch_per_side, n_patch_per_side))
    plt.savefig(save_path)
    plt.close()


def save_pi_snapshot(
    pi_raw: torch.tensor,
    method,
    output_dir: str,
    num_observed_patches: int,
    n_patch_per_side: int,
    save_individual_distribs=False,
    base_name="",
):
    if method == "joint":
        joint_subfolder = os.path.join(output_dir, "joints")
        os.makedirs(joint_subfolder, exist_ok=True)
        pi_raw_combined = torch.zeros(pi_raw[0].shape)
        pi_summed = torch.zeros(pi_raw[0].shape)
        for idx, row in enumerate(pi_raw):
            if save_individual_distribs:
                plot_distribution(
                    row,
                    joint_subfolder,
                    str(idx) + "_" + base_name,
                    num_observed_patches,
                    n_patch_per_side=n_patch_per_side,
                )
            # print("ROW ", row)
            max_, argmax_ = torch.max(row.unsqueeze(0), dim=1)
            row_masked = torch.zeros(row.shape)
            row_masked[argmax_] = max_
            pi_raw_combined += row_masked
            pi_summed += row
        plot_distribution(
            pi_raw_combined,
            output_dir,
            "_COMBINED_" + base_name,
            num_observed_patches,
            n_patch_per_side=n_patch_per_side,
        )
        plot_distribution(
            pi_summed,
            output_dir,
            "_SUMMED_" + base_name,
            num_observed_patches,
            n_patch_per_side=n_patch_per_side,
        )
        _, selected_inds = torch.max(pi_raw, dim=1)

    elif method == "topk":
        plot_distribution(
            pi_raw.squeeze(0),
            output_dir,
            base_name,
            num_observed_patches,
            n_patch_per_side=n_patch_per_side,
        )

        _, selected_inds = torch.topk(pi_raw, num_observed_patches)
    else:
        raise Exception("Invalid sampling method")


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
