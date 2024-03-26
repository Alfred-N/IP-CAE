import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.gumbel_distrib import (
    GumbelDistribution,
    GumbelDistributionBijectiveSoftmax,
    GumbelDistributionLogits,
    GumbelDistributionSoftmaxAct,
    GumbelDistributionSoftPlus,
)


class ConcreteLinear(pl.LightningModule):
    def __init__(
        self,
        input_dim=77,
        mask_ratio= None,
        k = None,
        dim_ip=0,
        pi_dropout=0,
        IP_dropout=0,
        pi_layer_norm=False,
        gumbel_learn_mode="logits",
        IP_initialization="random",
        IP_weights="shared",
        IP_bias=True,
        no_gumbel_noise = False
    ):
        super().__init__()
        self.model_family = "cae"
        self.input_dim = input_dim
        self.mask_ratio = mask_ratio
        self.no_gumbel_noise = no_gumbel_noise
        assert not (mask_ratio is not None and k is not None), "Either specify mask ratio or k"
        if mask_ratio:
            self.k = int((1 - mask_ratio) * input_dim)
        else:
            self.k = k

        gumbel_args = dict(
            num_categories=input_dim,
            num_distributions=self.k,
            marginal_initialization="random",
            dim_ip=dim_ip,
            pi_dropout=pi_dropout,
            IP_dropout=IP_dropout,
            layer_norm=pi_layer_norm,
            IP_initialization=IP_initialization,
            IP_weights=IP_weights,
            IP_bias=IP_bias,
        )

        if gumbel_learn_mode == "pi":
            self.gumbel_distrib = GumbelDistribution(**gumbel_args)
        elif gumbel_learn_mode == "logits":
            self.gumbel_distrib = GumbelDistributionLogits(**gumbel_args)
        elif gumbel_learn_mode == "softmax":
            self.gumbel_distrib = GumbelDistributionSoftmaxAct(**gumbel_args)
        elif gumbel_learn_mode == "softplus":
            self.gumbel_distrib = GumbelDistributionSoftPlus(**gumbel_args)
        elif gumbel_learn_mode == "bij_softmax":
            self.gumbel_distrib = GumbelDistributionBijectiveSoftmax(**gumbel_args)
        else:
            raise NotImplementedError(
                f"Incorrect gumbel_mode'{gumbel_learn_mode}', choose between 'pi', 'softmax', 'logits' or 'bij_softmax'"
            )

        self.decoder = nn.Linear(self.k, input_dim, device=self.device)

    def feature_select(self, X, temperature, random, hard=False, eeg_threshold=None, rao_samples=0):
        num_batches = X.shape[0]
        m, distrib_dict = self.gumbel_distrib.batch_sample_joint(
            num_batches, temperature, random, hard=hard, eeg_threshold=eeg_threshold, rao_samples=rao_samples, no_gumbel_noise=self.no_gumbel_noise
        )
        u = torch.bmm(m, X.unsqueeze(-1))
        u = u.squeeze(-1)
        return u, distrib_dict, m.detach()

    def check_m_convergence(self, m, tol=0.99, mode="all"):
        assert mode in ["any", "all"]

        maxes, _ = torch.max(m, dim=2)
        means = torch.mean(maxes, dim=1)

        convg = (
            torch.any(means > tol).item()
            if mode == "any"
            else torch.all(means > tol).item()
        )
        converge_dict = {
            "convergence": convg,
            "mean_mean_max": torch.mean(means, dim=0).item(),
        }
        return converge_dict

    def forward(self, X, random, temperature, hard=False, eeg_threshold=None, rao_samples=0):
        u, distrib_dict, m = self.feature_select(
            X, temperature, random, hard=hard, eeg_threshold=eeg_threshold, rao_samples=rao_samples
        )
        X_rec = self.decoder(u)
        loss = F.mse_loss(X, X_rec, reduction="mean")
        frob_norm = torch.norm(X - X_rec, p="fro", dim=-1) / self.input_dim
        frob_norm = frob_norm.mean()
        converge_dict = self.check_m_convergence(m)
        returns = {
            "loss": loss,
            "frob_norm": frob_norm,
            "distrib_dict": distrib_dict,
            "converge_dict": converge_dict,
        }
        return returns


class ConcreteClassification(ConcreteLinear):
    def __init__(
        self,
        input_dim=77,
        mask_ratio=None,
        k=None,
        dim_ip=0,
        pi_dropout=0,
        IP_dropout=0,
        pi_layer_norm=False,
        gumbel_learn_mode="logits",
        num_classes=2,
        dropout=0.0,
        norm_layer=nn.LayerNorm,
        decoder_hiddens=[],
        IP_initialization="random",
        IP_weights="shared",
        IP_bias=True,
        no_gumbel_noise = False
    ):
        super().__init__(
            input_dim,
            mask_ratio,
            k,
            dim_ip,
            pi_dropout,
            IP_dropout,
            pi_layer_norm,
            gumbel_learn_mode,
            IP_initialization=IP_initialization,
            IP_weights=IP_weights,
            IP_bias=IP_bias,
            no_gumbel_noise=no_gumbel_noise
        )
        if len(decoder_hiddens) == 0:
            self.decoder = nn.Linear(self.k, num_classes, device=self.device)
        else:
            decoder_hiddens = [self.k] + decoder_hiddens + [num_classes]
            nets_dec = []
            for i in range(len(decoder_hiddens) - 1):
                nets_dec += [nn.Linear(decoder_hiddens[i], decoder_hiddens[i + 1])]
                if i < len(decoder_hiddens) - 2:
                    if norm_layer is not None:
                        nets_dec += [norm_layer(decoder_hiddens[i + 1])]
                    nets_dec += [nn.LeakyReLU(0.2)]
                    nets_dec += [nn.Dropout(dropout)]
            self.decoder = nn.Sequential(*nets_dec)

    def forward(self, X, random, temperature, hard=False, eeg_threshold=None, rao_samples=0):
        u, distrib_dict, m = self.feature_select(
            X, temperature, random, hard=hard, rao_samples=rao_samples, eeg_threshold=eeg_threshold
        )
        y_pred = self.decoder(u)
        return_dict = dict(
            logits=y_pred,
            converge_dict=self.check_m_convergence(m),
            distrib_dict=distrib_dict,
        )
        return return_dict


class ConcreteMLP(ConcreteLinear):
    def __init__(
        self,
        input_dim=77,
        decoder_hiddens=[],
        dropout=0.0,
        norm_layer=nn.LayerNorm,
        mask_ratio=None,
        k=None,
        dim_ip=0,
        pi_dropout=0,
        IP_dropout=0,
        pi_layer_norm=False,
        gumbel_learn_mode="logits",
        IP_initialization="random",
        IP_weights="shared",
        IP_bias=True,
        no_gumbel_noise = False
    ):
        super().__init__(
            input_dim,
            mask_ratio,
            k,
            dim_ip,
            pi_dropout,
            IP_dropout,
            pi_layer_norm,
            gumbel_learn_mode,
            IP_initialization=IP_initialization,
            IP_weights=IP_weights,
            IP_bias=IP_bias,
            no_gumbel_noise = no_gumbel_noise
        )

        decoder_hiddens = [self.k] + decoder_hiddens + [input_dim]
        nets_dec = []
        for i in range(len(decoder_hiddens) - 1):
            nets_dec += [nn.Linear(decoder_hiddens[i], decoder_hiddens[i + 1])]
            if i < len(decoder_hiddens) - 2:
                if norm_layer is not None:
                    nets_dec += [norm_layer(decoder_hiddens[i + 1])]
                nets_dec += [nn.LeakyReLU(0.2)]
                nets_dec += [nn.Dropout(dropout)]
        self.decoder = nn.Sequential(*nets_dec)


# MNIST
def cae_MLP_MNIST(**kwargs): # reconstruction model
    return ConcreteMLP(
        input_dim=784,
        decoder_hiddens=[200],
        **kwargs
    )

def clas_cae_MLP_MNIST(**kwargs): # classification model
    return ConcreteClassification(
        input_dim=784,
        decoder_hiddens=[200],
        num_classes=10,
        **kwargs
    )

# COIL20
def cae_MLP_COIL20(**kwargs):
    return ConcreteMLP(
        input_dim=1024,
        decoder_hiddens=[200],
        **kwargs
    )

def clas_cae_MLP_COIL20(**kwargs): # classification model
    return ConcreteClassification(
        input_dim=1024,
        decoder_hiddens=[200],
        num_classes=20,
        **kwargs
    )

# MICE

def cae_MLP_MICE(**kwargs):
    return ConcreteMLP(
        input_dim=77,
        decoder_hiddens=[200],
        **kwargs
    )

def clas_cae_MLP_MICE(**kwargs):
    return ConcreteClassification(
        input_dim=77,
        num_classes=8,
        decoder_hiddens=[200],
        **kwargs
    )

def cae_linear_MICE(**kwargs):
    return ConcreteMLP(
        input_dim=77,
        **kwargs
    )

def clas_cae_linear_MICE(**kwargs):
    return ConcreteClassification(
        input_dim=77,
        num_classes=8,
        **kwargs
    )

# ISOLET
def cae_MLP_ISOLET(**kwargs):
    return ConcreteMLP(
        input_dim=617,
        decoder_hiddens=[200],
        **kwargs
    )

def clas_cae_MLP_ISOLET(**kwargs): # classification model
    return ConcreteClassification(
        input_dim=617,
        decoder_hiddens=[200],
        num_classes=26,
        **kwargs
    )

# ISOLET
def cae_MLP_ISOLET_linear(**kwargs):
    return ConcreteMLP(
        input_dim=617,
        decoder_hiddens=[],
        **kwargs
    )

def clas_cae_MLP_ISOLET_linear(**kwargs): # classification model
    return ConcreteClassification(
        input_dim=617,
        decoder_hiddens=[],
        num_classes=26,
        **kwargs
    )

# ISOLET
def cae_MLP_ISOLET_2_hiddens(**kwargs):
    return ConcreteMLP(
        input_dim=617,
        decoder_hiddens=[200, 200],
        **kwargs
    )

def clas_cae_MLP_ISOLET_2_hiddens(**kwargs): # classification model
    return ConcreteClassification(
        input_dim=617,
        decoder_hiddens=[200, 200],
        num_classes=26,
        **kwargs
    )

# SMARTPHONE

def cae_MLP_ACTIVITY(**kwargs):
    return ConcreteMLP(
        input_dim=561,
        decoder_hiddens=[200],
        **kwargs
    )

def clas_cae_MLP_ACTIVITY(**kwargs): # classification model
    return ConcreteClassification(
        input_dim=561,
        decoder_hiddens=[200],
        num_classes=6,
        **kwargs
    )


# CANCER

def cae_MLP_CANCER(**kwargs):
    return ConcreteMLP(
        input_dim=30,
        decoder_hiddens=[200],
        **kwargs
    )

def clas_cae_MLP_CANCER(**kwargs):
    return ConcreteClassification(
        input_dim=30,
        num_classes=2,
        decoder_hiddens=[200],
        **kwargs
    )
    
# ARCENE

def cae_MLP_ARCENE(**kwargs):
    return ConcreteMLP(
        input_dim=10_000,
        decoder_hiddens=[200],
        **kwargs
    )

def clas_cae_MLP_ARCENE(**kwargs):
    return ConcreteClassification(
        input_dim=10_000,
        num_classes=2,
        decoder_hiddens=[200],
        **kwargs
    )
    
# DEXTER

def cae_MLP_DEXTER(**kwargs):
    return ConcreteMLP(
        input_dim=20_000,
        decoder_hiddens=[200],
        **kwargs
    )

def clas_cae_MLP_DEXTER(**kwargs):
    return ConcreteClassification(
        input_dim=20_000,
        num_classes=2,
        decoder_hiddens=[200],
        **kwargs
    )
    
# GENE

def cae_MLP_GENE(**kwargs):
    return ConcreteMLP(
        input_dim=20531,
        decoder_hiddens=[200],
        **kwargs
    )

def clas_cae_MLP_GENE(**kwargs):
    return ConcreteClassification(
        input_dim=20531,
        num_classes=2,
        decoder_hiddens=[200],
        **kwargs
    )
    
# BREAST

def cae_MLP_BREAST(**kwargs):
    return ConcreteMLP(
        input_dim=12928,
        decoder_hiddens=[200],
        **kwargs
    )

def clas_cae_MLP_BREAST(**kwargs):
    return ConcreteClassification(
        input_dim=12928,
        num_classes=2,
        decoder_hiddens=[200],
        **kwargs
    )