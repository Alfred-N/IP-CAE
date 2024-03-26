import numpy as np
import torch
from torch.utils.data import TensorDataset


class BreastCancer(TensorDataset):

    def __init__(self, root, split="train", download=False):
        x = np.load("x.npy")
        y = np.load("y.npy")
        x, y = map(torch.from_numpy, [x, y])
        y = torch.nn.functional.one_hot(y.to(torch.int64), 2)
        super().__init__(x.to(torch.float32), y)
