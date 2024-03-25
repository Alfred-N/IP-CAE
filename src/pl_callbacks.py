import os
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pytorch_lightning as pl
import math
from operator import attrgetter

import torch

matplotlib.use("Agg")


class TemperatureCallback(pl.Callback):
    """
    Callback to adjust the temperature parameter during training epochs.
    Assumes the PL module has an attribute "temperature"

    Args:
        num_epochs (int): Total number of epochs.
        temp_base (float): Base temperature value.
        temp_min (float): Minimum temperature value.
        stop_anneal (int, optional): Epoch at which to stop annealing. Defaults to 0.
        warmup_epochs (int, optional): Number of warm-up epochs. Defaults to 0.
        mode (str, optional): Annealing mode, either "exp" or "linear". Defaults to "exp".
    """

    def __init__(
        self,
        num_epochs,
        temp_base,
        temp_min,
        stop_anneal=0,
        warmup_epochs=0,
        mode="exp",
    ):
        super().__init__()
        self.num_epochs = num_epochs
        self.temp_base = temp_base
        self.temp_min = temp_min
        self.stop_anneal = stop_anneal
        self.warmup_epochs = warmup_epochs
        self.mode = mode

    def get_current_value(self, epoch):
        if epoch < self.warmup_epochs:
            temp = self.temp_base
        else:
            adj_epoch = epoch - (self.warmup_epochs + 1)
            adj_num_epochs = self.num_epochs - (self.warmup_epochs + 1)

            if self.mode == "exp":
                if self.stop_anneal > 0:
                    if adj_epoch > self.stop_anneal:
                        return self.temp_base * (self.temp_min / self.temp_base) ** (
                            self.stop_anneal / adj_num_epochs
                        )
                temp = self.temp_base * (self.temp_min / self.temp_base) ** (
                    adj_epoch / adj_num_epochs
                )
            elif self.mode == "linear":
                k = (self.temp_min - self.temp_base) / (
                    self.num_epochs - self.warmup_epochs
                )
                if self.stop_anneal > 0:
                    if adj_epoch > self.stop_anneal:
                        return k * self.stop_anneal + self.temp_base
                temp = k * adj_epoch + self.temp_base

            temp = max(temp, self.temp_min)
        return temp

    def on_train_epoch_start(self, trainer, pl_module):
        epoch = trainer.current_epoch
        temp = self.get_current_value(epoch)
        pl_module.temperature = temp
        trainer.logger.log_metrics({"temperature": temp}, step=None)


class ThresholdCallback(TemperatureCallback):
    def on_train_epoch_start(self, trainer, pl_module):
        epoch = trainer.current_epoch
        threshold = self.get_current_value(epoch)
        pl_module.eeg_threshold = threshold
        trainer.logger.log_metrics({"eeg_thresh": threshold}, step=None)


class CosineAnnealLRCallback(pl.Callback):
    """Callback to adjust the learning rate using the cosine annealing schedule with warmup.

    Args:
        lr (float): The initial learning rate.
        min_lr (float): The minimum learning rate.
        warmup_epochs (int): The number of warmup epochs before applying cosine annealing.

    """

    def __init__(self, lr, min_lr, warmup_epochs):
        self.lr = lr
        self.min_lr = min_lr
        self.warmup_epochs = warmup_epochs

    def on_train_epoch_start(self, trainer, pl_module):
        epoch = trainer.current_epoch - 1
        tot_epochs = trainer.max_epochs - 1

        if epoch < self.warmup_epochs:
            fac = epoch / self.warmup_epochs
            lr_temp = self.lr * fac + self.min_lr * (1 - fac)
        else:
            lr_temp = self.min_lr + (self.lr - self.min_lr) * 0.5 * (
                1.0
                + math.cos(
                    math.pi
                    * (epoch - self.warmup_epochs)
                    / (tot_epochs - self.warmup_epochs)
                )
            )

        optimizer = trainer.optimizers[0]
        for param_group in optimizer.param_groups:
            if "lr_scale" in param_group:
                param_group["lr"] = lr_temp * param_group["lr_scale"]
            else:
                param_group["lr"] = lr_temp
        pl_module.lr = lr_temp


class WarmupLRCallback(CosineAnnealLRCallback):
    def on_train_epoch_start(self, trainer, pl_module):
        epoch = trainer.current_epoch - 1

        if epoch < self.warmup_epochs:
            fac = epoch / self.warmup_epochs
            lr_temp = self.lr * fac + self.min_lr * (1 - fac)
        else:
            lr_temp = self.lr

        optimizer = trainer.optimizers[0]
        for param_group in optimizer.param_groups:
            if "lr_scale" in param_group:
                param_group["lr"] = lr_temp * param_group["lr_scale"]
            else:
                param_group["lr"] = lr_temp
        pl_module.lr = lr_temp


class FreezeDistribCallback(pl.Callback):
    """Callback to freeze a model's distribution layer for a specified number of epochs.

    Args:
        model: The model containing a distribution layer.
        freeze_distrib (bool): Freeze the distribution permanently.
        freeze_epochs (int): The number of epochs for which the distribution layer should be frozen.

    Returns:
        None
    """

    def __init__(self, model, freeze_distrib=None, freeze_epochs=None):
        super().__init__()
        assert not (
            freeze_distrib and freeze_epochs
        ), "args 'freeze_distrib' and 'freeze_epochs' are incompatible"

        # check if the model has a gumbel distrib
        _err_msg = f"No Gumbel Distribution module was found for {type(model)}. Disable this callback {type(self)}"
        if hasattr(model, "encoder"):
            encoder = getattr(model, "encoder")
            if hasattr(encoder, "feature_select"):
                self._attr = "encoder.feature_select.gumbel_distrib"
            elif hasattr(encoder, "gumbel_distrib"):
                self._attr = "encoder.gumbel_distrib"
            else:
                raise NotImplementedError(_err_msg)
        elif hasattr(model, "feature_select.gumbel_distrib"):
            self._attr = "feature_select.gumbel_distrib"
        elif hasattr(model, "gumbel_distrib"):
            self._attr = "gumbel_disrib"
        else:
            raise NotImplementedError(_err_msg)

        self.attrgetter = attrgetter(self._attr)

        self.freeze_epochs = freeze_epochs

        distrib = self.attrgetter(model)

        if freeze_distrib:
            distrib.freeze(True)

    def on_train_epoch_start(self, trainer, pl_module):
        if self.freeze_epochs is not None:
            distrib = self.attrgetter(pl_module.model)
            distrib.freeze((trainer.current_epoch - 1) < self.freeze_epochs)


def plot_distribution(
    pi_raw: torch.tensor,
    base_path: str,
    base_name: str,
    n_pixels_per_side=14,
):
    save_name = "pi_marginal" + base_name + ".png"
    save_path = os.path.join(base_path, save_name)
    print(
        "Plotting distribution--",
        save_path,
    )
    plt.imshow(
        pi_raw.numpy().reshape(n_pixels_per_side, n_pixels_per_side),
        cmap="viridis",  # Choose the colormap you prefer
        origin="upper",  # Set origin to upper left corner
    )
    plt.axis("off")  # Turn off axis
    plt.savefig(
        save_path, bbox_inches="tight", pad_inches=0
    )  # Save without extra padding
    plt.close()


def save_pi_snapshot(
    pi_raw: torch.tensor,
    method,
    output_dir: str,
    total_pixels: int,
    n_pixels_per_side: int,
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
                    str(idx) + "_" + base_name if base_name else str(idx),
                    n_pixels_per_side=n_pixels_per_side,
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
            n_pixels_per_side=n_pixels_per_side,
        )
        plot_distribution(
            pi_summed,
            output_dir,
            "_SUMMED_" + base_name,
            n_pixels_per_side=n_pixels_per_side,
        )
        _, selected_inds = torch.max(pi_raw, dim=1)

    elif method == "topk":
        plot_distribution(
            pi_raw.squeeze(0),
            output_dir,
            base_name,
            n_pixels_per_side=n_pixels_per_side,
        )

        _, selected_inds = torch.topk(pi_raw, total_pixels)
    else:
        raise Exception("Invalid sampling method")


class SnapshotCallback(pl.Callback):
    def __init__(self, output_dir) -> None:
        super().__init__()
        self.output_dir = os.path.join(output_dir, "snapshots")

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        distrib = pl_module.model.gumbel_distrib
        pi, _, _ = distrib.get_pi(eps=1e-10)
        total_pixels = pi.shape[-1]
        n_pixels_per_side = int(np.sqrt(total_pixels))

        current_epoch = trainer.current_epoch
        base_name = "epoch_" + str(current_epoch)
        save_pi_snapshot(
            pi,
            "joint",
            output_dir=self.output_dir,
            total_pixels=total_pixels,
            n_pixels_per_side=n_pixels_per_side,
            base_name=base_name,
        )

        return super().on_validation_epoch_end(trainer, pl_module)
