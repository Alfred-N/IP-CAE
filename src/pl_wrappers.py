import torch
from torch import nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import timm.optim.optim_factory as optim_factory
import numpy as np
from torchmetrics import (
    StructuralSimilarityIndexMeasure,
    PeakSignalNoiseRatio,
    # MultiScaleStructuralSimilarityIndexMeasure,
)
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)





def fill_masked_patches(x_in, x_pred, masks):
    return x_pred * masks.unsqueeze(-1) + x_in * (1 - masks.unsqueeze(-1))


def calc_classification_metrics(all_outputs, all_targets):
    all_targets = all_targets.detach().cpu().float().numpy()
    all_outputs = all_outputs.detach().cpu().float().numpy()
    # Compute accuracy
    accuracy = accuracy_score(all_targets.argmax(axis=-1), all_outputs.argmax(axis=-1))

    # Compute precision, recall, and F1-score
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_targets.argmax(axis=-1),
        all_outputs.argmax(axis=-1),
        average="weighted",
        zero_division=0,
    )

    # Compute confusion matrix
    # confusion_mat = confusion_matrix(all_targets.argmax(axis=-1), all_outputs.argmax(axis=-1))

    metrics = dict(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        # confusion_mat=confusion_mat,
    )
    return metrics


class BestMetricTracker:
    def __init__(self):
        self.best_metrics = {}
        self.optimization_direction = (
            {}
        )  # Tracks the optimization direction for each metric

    def update_metric(self, name, value, maximize=True):
        if name not in self.best_metrics or self._is_better(
            value, self.best_metrics[name], maximize
        ):
            self.best_metrics[name] = value
            self.optimization_direction[name] = maximize

    def get_best_metric(self, name):
        return self.best_metrics.get(name, None)

    def _is_better(self, new_value, old_value, maximize):
        if maximize:
            return new_value > old_value
        else:
            return new_value < old_value


class PL_ReconstructionWrapper(pl.LightningModule):
    """
    PyTorch Lightning training wrapper module for reconstruction.

    Args:
        model (nn.Module): The model.
        config (dict): Configuration dictionary.
    """

    def __init__(self, model, args, datasets):
        super().__init__()

        self.model = model
        self.use_masked_loss = args.use_masked_loss
        self.jsd_factor = args.jsd_factor
        self.eeg_factor = args.eeg_factor
        self.temperature = args.temp_base
        self.eeg_threshold = args.thresh_base
        self.rao_samples = args.rao_samples
        self.batch_size = args.batch_size
        self.test_batch_size = args.test_batch_size
        self.num_workers = args.num_workers
        self.pin_mem = args.pin_mem
        self.weight_decay = args.weight_decay
        self.datasets = datasets
        self.hard = args.straight_through  # If True, use Straight-Through Gumbel-Softmax
        self.lr = args.lr
        self.norm_pix_loss = args.norm_pix_loss
        self.tracker = BestMetricTracker()

        loss_type = args.loss_type
        if loss_type == "mse":
            self.loss_fn = nn.MSELoss(reduction="none")
        elif loss_type == "ce":
            self.loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
        elif loss_type == "bce":
            self.loss_fn = torch.nn.BCEWithLogitsLoss(reduction="none")
        else:
            raise NotImplementedError(f"Invalid loss func {loss_type} specified")
        self.ssim_measure = StructuralSimilarityIndexMeasure()
        # self.mssim_measure = MultiScaleStructuralSimilarityIndexMeasure()
        self.psnr_measure = PeakSignalNoiseRatio()
        self.loss_type = loss_type

    def forward(self, x, **kwargs):
        return self.model(x, **kwargs)

    def _get_input(self, batch):
        imgs, _ = batch
        return imgs

    def _get_losses(self, returns):
        raise NotImplementedError("Implement _get_losses")

    def _get_unique_percentage(self, pi):
        _, selected_inds = torch.max(pi, dim=-1)
        num_observed_patches = self.model.num_obs_patches
        unique_inds = np.unique(selected_inds.cpu())
        unique_percentage = len(unique_inds) / num_observed_patches
        return unique_percentage

    def _get_train_stats(self, returns):
        raise NotImplementedError("Implement _get_train_stats")

    def _get_eval_stats(self, split, returns, imgs):
        raise NotImplementedError("Implement _get_eval_stats")

    def training_step(self, batch, batch_idx):
        imgs = self._get_input(batch)
        returns = self.forward(
            imgs,
            random=True,
            temperature=self.temperature,
            hard=self.hard,
            eeg_threshold=self.eeg_threshold,
            rao_samples=self.rao_samples,
        )
        loss, train_stats = self._get_train_stats(returns)
        self.log_dict(
            {**train_stats}, on_epoch=True, batch_size=imgs.shape[0], sync_dist=True
        )
        self.log("scalar_value_train", self.model.gumbel_distrib.get_scalar_value(), on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        imgs = self._get_input(batch)
        returns = self.forward(
            imgs,
            random=False,
            temperature=self.temperature,
            hard=self.hard,
            eeg_threshold=self.eeg_threshold,
        )
        val_stats = self._get_eval_stats("val", returns, imgs)
        self.log_dict(
            {**val_stats}, on_epoch=True, batch_size=imgs.shape[0], sync_dist=True
        )
        self.log("scalar_value_valid", self.model.gumbel_distrib.get_scalar_value(), on_step=True)
        return val_stats

    def test_step(self, batch, batch_idx):
        imgs = self._get_input(batch)
        returns = self.forward(
            imgs,
            random=False,
            temperature=self.temperature,
            hard=self.hard,
            eeg_threshold=self.eeg_threshold,
        )
        test_stats = self._get_eval_stats("test", returns, imgs)
        self.log_dict(
            {**test_stats}, on_epoch=True, batch_size=imgs.shape[0], sync_dist=True
        )
        self.log("scalar_value_test", self.model.gumbel_distrib.get_scalar_value(), on_step=True)

    def train_dataloader(self):
        return DataLoader(
            self.datasets[0],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_mem,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.datasets[1],
            batch_size=self.test_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_mem,
            drop_last=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.datasets[2],
            batch_size=self.test_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_mem,
            drop_last=False,
        )

    def configure_optimizers(self):
        param_groups = optim_factory.param_groups_weight_decay(
            self.model, self.weight_decay
        )
        return torch.optim.Adam(param_groups, lr=self.lr, betas=(0.9, 0.999))


class PL_ClassificationWrapper(PL_ReconstructionWrapper):
    """
    PyTorch Lightning training wrapper module for classification.

    Args:
        model (nn.Module): The model.
        config (dict): Configuration dictionary.
    """

    def __init__(
        self,
        model,
        args,
        datasets,
    ):
        super().__init__(model, args, datasets)
        del self.ssim_measure
        del self.psnr_measure
        self.mode = args.classification_mode
        self.layer_decay = args.layer_decay

    def _get_input(self, batch):
        imgs, labels = batch
        return imgs, labels

    def _get_losses(self, returns, labels):
        logits = returns["logits"]
        gjsd = returns["distrib_dict"]["GJS"].squeeze(0)
        if self.loss_type == "bce":
            loss = self.loss_fn(logits, labels).mean()
        elif self.loss_type == "ce":
            loss = self.loss_fn(logits, labels.argmax(dim=-1)).mean()
        else:
            preds = torch.sigmoid(logits)
            loss = self.loss_fn(preds, labels).mean()
        return loss, gjsd

    def _get_train_stats(self, returns, labels):
        loss, gjsd = self._get_losses(returns, labels)
        if self.jsd_factor > 0:
            loss -= self.jsd_factor * gjsd
        if self.eeg_factor > 0:
            eeg = returns["distrib_dict"]["EEG"].squeeze(0)
            loss += self.eeg_factor * eeg
        logits = returns["logits"]
        metrics = calc_classification_metrics(logits, labels)
        metrics["loss"] = loss.detach().item()
        stats = {"train_" + key: val for key, val in metrics.items()}
        stats["gjsd_term"] = gjsd.detach().item()
        stats["lr"] = self.lr
        if self.eeg_factor > 0:
            stats["eeg_term"] = eeg.detach().item()
        return loss, stats

    def _get_eval_stats(self, split, returns, labels):
        loss, _ = self._get_losses(returns, labels)
        logits = returns["logits"]
        metrics = calc_classification_metrics(logits, labels)
        metrics["loss"] = loss.detach().item()
        stats = {split + "_" + key: val for key, val in metrics.items()}
        if split == "val":
            stats["unique_percentage"] = self._get_unique_percentage(
                returns["distrib_dict"]["current_pi"].detach()
            )
        return stats

    def training_step(self, batch, batch_idx):
        imgs, labels = self._get_input(batch)
        returns = self.forward(
            imgs,
            random=True,
            temperature=self.temperature,
            hard=self.hard,
            eeg_threshold=self.eeg_threshold,
            rao_samples = self.rao_samples
        )
        loss, train_stats = self._get_train_stats(returns, labels)
        self.log_dict(
            {**train_stats}, on_epoch=True, batch_size=imgs.shape[0], sync_dist=True
        )
        self.log("scalar_value_train", self.model.gumbel_distrib.get_scalar_value(), on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, labels = self._get_input(batch)
        returns = self.forward(
            imgs,
            random=False,
            temperature=self.temperature,
            hard=self.hard,
            eeg_threshold=self.eeg_threshold,
        )
        val_stats = self._get_eval_stats("val", returns, labels)
        self.log_dict(
            {**val_stats}, on_epoch=True, batch_size=imgs.shape[0], sync_dist=True
        )
        self.log("scalar_value_valid", self.model.gumbel_distrib.get_scalar_value(), on_step=True)
        return val_stats

    def test_step(self, batch, batch_idx):
        imgs, labels = self._get_input(batch)
        returns = self.forward(
            imgs,
            random=False,
            temperature=self.temperature,
            hard=self.hard,
            eeg_threshold=self.eeg_threshold,
        )
        test_stats = self._get_eval_stats("test", returns, labels)
        self.log_dict(
            {**test_stats}, on_epoch=True, batch_size=imgs.shape[0], sync_dist=True
        )
        self.log("scalar_value_test", self.model.gumbel_distrib.get_scalar_value(), on_step=True)
        return test_stats

    def configure_optimizers(self):
        param_groups = optim_factory.param_groups_weight_decay(
            self.model, self.weight_decay
        )
        return torch.optim.Adam(param_groups, lr=self.lr, betas=(0.9, 0.999))

    def on_validation_epoch_end(self):
        # Get the per-epoch metric value from the logged metrics
        cur_epoch = self.trainer.current_epoch
        if self.global_rank == 0:
            if cur_epoch > 0:
                metrics = self.trainer.logged_metrics
                self.tracker.update_metric(
                    "best_gjsd",
                    metrics["gjsd_term_epoch"].detach().cpu().item(),
                    maximize=True,
                )
                self.tracker.update_metric(
                    "best_val_loss",
                    metrics["val_loss"].detach().cpu().item(),
                    maximize=False,
                )
                self.tracker.update_metric(
                    "best_val_accuracy",
                    metrics["val_accuracy"].detach().cpu().item(),
                    maximize=True,
                )
                self.tracker.update_metric(
                    "best_val_f1",
                    metrics["val_f1"].detach().cpu().item(),
                    maximize=True,
                )
                self.tracker.update_metric(
                    "best_val_recall",
                    metrics["val_recall"].detach().cpu().item(),
                    maximize=True,
                )
                self.tracker.update_metric(
                    "best_val_precision",
                    metrics["val_precision"].detach().cpu().item(),
                    maximize=True,
                )
            if cur_epoch == self.trainer.max_epochs - 1:
                self.log_dict(self.tracker.best_metrics)


class PL_CAE_Wrapper(PL_ReconstructionWrapper):
    """
    PyTorch Lightning module for the original Concrete Autoencoder.

    Args:
        model (nn.Module): The CAE model.
        config (dict): Configuration dictionary.
    """

    def _get_input(self, batch):
        input, _ = batch
        inputs = input.view((input.shape[0], -1))
        return inputs

    def _get_losses(self, returns):
        gjsd = returns["distrib_dict"]["GJS"].squeeze(0)
        loss = returns["loss"]
        return loss, gjsd

    def _get_train_stats(self, returns):
        loss, gjsd = self._get_losses(returns)
        if self.jsd_factor > 0:
            loss -= self.jsd_factor * gjsd
        if self.eeg_factor > 0:
            eeg = returns["distrib_dict"]["EEG"].squeeze(0)
            loss += self.eeg_factor * eeg
        stats = {"train_loss": loss}
        stats["train_frob_norm"] = returns["frob_norm"]
        stats["temperature"] = self.temperature
        stats["lr"] = self.lr
        stats["gjsd_term"] = gjsd.detach().item()
        stats["mean_mean_max_m"] = returns["converge_dict"]["mean_mean_max"]
        if self.eeg_factor > 0:
            stats["eeg_term"] = eeg.detach().item()
        return loss, stats

    def _get_unique_percentage(self, pi):
        _, selected_inds = torch.max(pi, dim=-1)
        num_observed_patches = self.model.k
        unique_inds = np.unique(selected_inds.cpu())
        unique_percentage = len(unique_inds) / num_observed_patches
        return unique_percentage

    def _get_eval_stats(self, split, returns, imgs):
        loss, gjsd = self._get_losses(returns)
        stats = {split + "_loss": loss, split + "_frob_norm": returns["frob_norm"]}
        if split == "val":
            stats["unique_percentage"] = self._get_unique_percentage(
                returns["distrib_dict"]["current_pi"].detach()
            )
        return stats

    def on_validation_epoch_end(self):
        # Get the per-epoch metric value from the logged metrics
        cur_epoch = self.trainer.current_epoch
        if self.global_rank == 0:
            if cur_epoch > 0:
                metrics = self.trainer.logged_metrics
                self.tracker.update_metric(
                    "best_gjsd",
                    metrics["gjsd_term_epoch"].detach().cpu().item(),
                    maximize=True,
                )
                self.tracker.update_metric(
                    "best_val_loss",
                    metrics["val_loss"].detach().cpu().item(),
                    maximize=False,
                )
                self.tracker.update_metric(
                    "best_val_frob_norm",
                    metrics["val_frob_norm"].detach().cpu().item(),
                    maximize=False,
                )
            if cur_epoch == self.trainer.max_epochs - 1:
                self.log_dict(self.tracker.best_metrics)

class PL_CAE_Wrapper_CLAS(PL_ClassificationWrapper):
    """
    PyTorch Lightning module for classification for the original Concrete Autoencoder.

    Args:
        model (nn.Module): The CAE model.
        config (dict): Configuration dictionary.
    """

    def _get_input(self, batch):
        input, labels = batch
        inputs = input.view((input.shape[0], -1))
        return inputs, labels
    
    def _get_unique_percentage(self, pi):
        _, selected_inds = torch.max(pi, dim=-1)
        num_observed_patches = self.model.k
        unique_inds = np.unique(selected_inds.cpu())
        unique_percentage = len(unique_inds) / num_observed_patches
        return unique_percentage