import time

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class IP(pl.LightningModule):
    def __init__(
        self,
        num_categories,
        dim_ip,
        num_vectors,
        marginal_initialization="random",
        dropout=0,
        mlp_dropout=0,
        hiddens=[],
        norm_layer=nn.LayerNorm,
        IP_initialization="random",
        bias=True,
    ):
        super().__init__()
        self.num_vectors = num_vectors
        self.num_categories = num_categories
        self.dim_ip = dim_ip
        if IP_initialization == "random":
            P_init = torch.randn((num_vectors, dim_ip))
        elif IP_initialization == "identity":
            P_init = torch.eye(dim_ip, requires_grad=True)[:num_vectors]
        else:
            raise ValueError(f"'{IP_initialization}' is not a valid IP initialization")
        self.ip_vectors = nn.Parameter(P_init)  # num_ctp x dim_ip

        net_dims = [dim_ip] + hiddens + [num_categories]
        self.weights = self.make_network(net_dims, norm_layer, mlp_dropout, bias=bias)

        if marginal_initialization == "uniform" and not hiddens:
            pi_init = torch.ones([num_vectors, num_categories]) / num_categories
            P_inv = np.linalg.pinv(P_init.detach())
            W_init = torch.tensor(
                np.dot(P_inv, pi_init).transpose(), requires_grad=True
            )
            self.weights[0].weight = nn.Parameter(W_init)
            nn.init.constant_(self.weights[0].bias, 0)
        self.dropout = nn.Dropout(dropout)

    def make_network(self, net_dims, norm_layer, mlp_dropout, bias):
        nets = []
        for i in range(len(net_dims) - 1):
            nets += [nn.Linear(net_dims[i], net_dims[i + 1], bias=bias)]
            if i < len(net_dims) - 2:
                if norm_layer is not None:
                    nets += [norm_layer(net_dims[i + 1])]
                nets += [nn.ReLU()]
                nets += [nn.Dropout(mlp_dropout)]
        return nn.Sequential(*nets)

    def forward(self):
        # num_ctp x num_categories
        pi_raw = self.weights(self.dropout(self.ip_vectors))
        return pi_raw


class IP_NonShared(IP):
    def __init__(
        self,
        num_categories,
        dim_ip,
        num_vectors,
        marginal_initialization="random",
        dropout=0,
        mlp_dropout=0,
        hiddens=[],
        norm_layer=nn.LayerNorm,
        IP_initialization="random",
        bias=True,
    ):
        super().__init__(
            num_categories,
            dim_ip,
            num_vectors,
            marginal_initialization,
            dropout,
            mlp_dropout,
            hiddens,
            norm_layer,
            IP_initialization,
            bias=bias,
        )

        net_dims = [dim_ip] + hiddens + [num_categories]
        networks = []
        for i in range(num_vectors):
            networks.append(
                self.make_network(net_dims, norm_layer, mlp_dropout, bias=bias)
            )
        self.weights = nn.ModuleList(networks)
        
    def forward(self):
        # num_ctp x num_categories
        ips = self.dropout(self.ip_vectors)
        pi_raws = []
        for i in range(self.num_vectors):
            pi_raws.append(self.weights[i](ips[i]))
        pi_raw_tensor = torch.stack(pi_raws)
        return pi_raw_tensor


class IP_FullyConnected(IP):
    def __init__(
        self,
        num_categories,
        dim_ip,
        num_vectors,
        marginal_initialization="random",
        dropout=0,
        mlp_dropout=0,
        hiddens=[],
        norm_layer=nn.LayerNorm,
        IP_initialization="random",
        bias=True,
    ):
        super().__init__(
            num_categories,
            dim_ip,
            num_vectors,
            marginal_initialization,
            dropout,
            mlp_dropout,
            hiddens,
            norm_layer,
            IP_initialization,
            bias,
        )
        if IP_initialization == "random":
            P_init = torch.randn((dim_ip*num_vectors,))
        else:
            raise ValueError(f"'{IP_initialization}' is not a valid IP initialization for class {type(self)}")
        self.ip_vectors = nn.Parameter(P_init)  # num_ctp x dim_ip

        net_dims = [dim_ip*num_vectors] + hiddens + [num_categories*num_vectors]
        self.weights = self.make_network(net_dims, norm_layer, mlp_dropout, bias=bias)

    def forward(self):
        # num_ctp x num_categories
        ips = self.dropout(self.ip_vectors)
        pi_raw = self.weights(ips).reshape((self.num_vectors, self.num_categories))
        return pi_raw


class GumbelDistribution(pl.LightningModule):
    def __init__(
        self,
        num_categories,
        marginal_initialization="uniform",
        num_distributions=1,
        dim_ip=0,
        pi_dropout=0,
        IP_dropout=0,
        layer_norm=False,
        IP_hiddens=[],
        IP_initialization="identity",
        IP_weights="shared",
        IP_bias=True,
    ):
        super().__init__()
        self.num_categories = num_categories
        self.num_distributions = num_distributions
        self.dim_ip = dim_ip

        ctp_args = dict(
            num_categories=num_categories,
            dim_ip=dim_ip,
            num_vectors=num_distributions,
            marginal_initialization=marginal_initialization,
            dropout=IP_dropout,
            hiddens=IP_hiddens,
            IP_initialization=IP_initialization,
            bias=IP_bias,
        )
        if dim_ip > 0:
            if IP_weights == "shared":
                self.pi_marginal = IP(**ctp_args)
            elif IP_weights == "separate":
                self.pi_marginal = IP_NonShared(**ctp_args)
            elif IP_weights == "fc":
                self.pi_marginal = IP_FullyConnected(**ctp_args)
            else:
                raise NotImplementedError()

        else:
            if marginal_initialization == "uniform":
                prior = (
                    torch.ones([num_distributions, num_categories], requires_grad=True)
                    / num_categories
                )
            elif marginal_initialization == "random":
                prior = torch.rand(
                    [num_distributions, num_categories], requires_grad=True
                )
                prior = prior / prior.sum(dim=1).unsqueeze(1)
            else:
                raise NotImplementedError

            self.pi_marginal = nn.Parameter(prior)

        self.pi_dropout = nn.Dropout(pi_dropout)

        if layer_norm:
            self.pi_layernorm = nn.LayerNorm((num_distributions, num_categories))
        else:
            self.pi_layernorm = None
        self.frozen = False

    def freeze(self, freeze: bool):
        for param in self.parameters():
            param.requires_grad = not freeze
        self.frozen = freeze

    def get_pi(self, eps):
        if self.dim_ip > 0:
            pi_raw = self.pi_marginal()
        else:
            pi_raw = self.pi_marginal

        if self.pi_layernorm:
            pi_raw = self.pi_layernorm(pi_raw)
        else:
            pi_raw = pi_raw

        pi = self.pi_dropout(pi_raw)
        pi = pi**2 + eps
        pi = pi / (pi.sum(dim=1).unsqueeze(1))  # sum to 1
        logits = torch.log(pi)
        return pi, logits, pi_raw

    def batch_sample_joint(
        self,
        num_batches,
        temperature,
        random,
        rao_samples=1,
        eps=1e-4,
        hard=False,
        ret_GJS=True,
        ret_EEG=True,
        temp_jsd=None,
        eeg_threshold=None,
    ):
        pi, logits, logits_raw = self.get_pi(eps=eps)

        distrib_dict = {"num_categories": pi.shape[1], "current_pi": pi}
        if ret_GJS:
            gjs = self.GJS_divergence(logits_raw)
            distrib_dict["GJS"] = gjs

        if ret_EEG:
            eeg = self.EEG_regularization(logits_raw, eeg_threshold)
            distrib_dict["EEG"] = eeg

        if not random:
            pi_deterministic = torch.zeros(pi.shape)
            observed_inds = torch.argmax(pi, dim=1)
            num_distrib, num_cat = pi.shape
            pi_deterministic[torch.arange(num_distrib), observed_inds] = 1
            batch_sampler = pi_deterministic.expand([num_batches, num_distrib, num_cat])
            batch_sampler = batch_sampler.type_as(pi)

        else:
            assert not (
                rao_samples > 1 and hard == False
            ), "Rao samples only works for hard (ST-GS) samples"

            if rao_samples > 1:
                batch_sampler = torch.zeros(
                    num_batches, self.num_distributions, self.num_categories
                ).to(self.device)

                for batch in range(num_batches):
                    sample = self.gumbel_rao(
                        logits, rao_samples, temperature, I=None, hard=hard
                    )
                    batch_sampler[batch] = sample

            else:
                logits = logits.unsqueeze(0).repeat(num_batches, 1, 1)
                batch_sampler = F.gumbel_softmax(logits, temperature, hard=hard)

        return batch_sampler, distrib_dict

    ###------------------ Rao-Blackweilization related code -------------------
    # --------------------------------------------------------------------------

    def conditional_gumbel(self, logits, D, k=1):
        """Outputs k samples of Q = StandardGumbel(), such that argmax(logits
        + Q) is given by D (one hot vector)."""
        # iid. exponential
        E = torch.distributions.exponential.Exponential(
            rate=torch.ones_like(logits)
        ).sample([k])
        # E of the chosen class
        Ei = (D * E).sum(dim=-1, keepdim=True)
        # partition function (normalization constant)
        Z = logits.exp().sum(dim=-1, keepdim=True)
        # Sampled gumbel-adjusted logits
        adjusted = D * (-torch.log(Ei) + torch.log(Z)) + (1 - D) * -torch.log(
            E / torch.exp(logits) + Ei / Z
        )
        return adjusted - logits

    def gumbel_rao(self, logits, k, temp=1.0, I=None, hard=False):
        """Returns a categorical sample from logits (over axis=-1) as a
        one-hot vector, with gumbel-rao gradient.
        k: integer number of samples to use in the rao-blackwellization.
        1 sample reduces to straight-through gumbel-softmax.
        I: optional, categorical sample to use instead of drawing a new
        sample. Should be a tensor(shape=logits.shape[:-1], dtype=int64).
        """
        num_classes = logits.shape[-1]
        if I is None:
            I = torch.distributions.categorical.Categorical(logits=logits).sample()
        D = torch.nn.functional.one_hot(I, num_classes).float()
        adjusted = logits + self.conditional_gumbel(logits, D, k=k)
        surrogate = torch.nn.functional.softmax(adjusted / temp, dim=-1).mean(dim=0)
        if not hard:
            return surrogate
        else:
            shape = surrogate.size()
            _, ind = surrogate.max(dim=-1)
            y_hard = torch.zeros_like(surrogate).view(-1, shape[-1])
            y_hard.scatter_(1, ind.view(-1, 1), 1)
            y_hard = y_hard.view(*shape)

            # Set gradients w.r.t. y_hard gradients w.r.t. surrogate
            y_hard = (y_hard - surrogate).detach() + surrogate
            return y_hard

    ###---------------- Uniqueness Regularization Quantities ------------------
    # --------------------------------------------------------------------------

    def GJS_divergence(self, logits_raw):
        # pi.shape = num_distrib x num_patches = M x C
        M = self.num_distributions
        # w = mixture_weights
        w = torch.ones([M, 1]).type_as(logits_raw) / M

        pi = F.softmax(logits_raw, dim=1)
        log_pi = F.log_softmax(logits_raw, dim=1)

        d_gls = torch.sum(
            w
            * torch.sum(
                pi * (log_pi - torch.log(torch.sum(w * pi, dim=0).repeat([M, 1]))),
                dim=1,
                keepdim=True,
            ),
            dim=0,
        )
        return d_gls

    def EEG_regularization(self, logits_raw: torch.Tensor, thresh: torch.Tensor):
        # pi.shape = num_distrib x num_categories (= k x n in the paper) (their matrix is n x k)

        # thresh: threshold float, hyperparameter, is exp decayed during training
        thresh = torch.tensor(thresh).type_as(logits_raw)

        ### sum across different distribs (k) and apply relu with threshold>0.
        # I assume this should encourage only one distribution to have weight for each feature.

        # my interpretation: penalize only if the sum of weights for a certain feature is above a threshold.
        # as the thresh hparam apporaches 1 (lowest value), the distributions will increasingly be
        # discouraged from having a nonzero value.
        # unclear why this doesn't just force each distribution to have a low probability instead of
        # one with high prob and the others 0.

        pi = F.softplus(logits_raw)
        pi = pi / (pi.sum(dim=1).unsqueeze(1))  # sum to 1

        gated_output = F.relu(pi.sum(dim=0) - thresh)

        # Example with n=1 category and k=4 distributions
        # [0.4]
        # [0.4]
        # [0.4]
        # [0.4]

        # Scenario
        # sum = 1.6
        # thresh = 1.1
        # wich means the sum will be penalized

        # This might happen
        # [0.01]
        # [0.01]
        # [0.01]
        # [0.01]

        # [1] - Don't see any reason why this would happen?
        # [0]
        # [0]
        # [0]

        # sum the gated sum of weights for each category to convert it to a scalar
        return gated_output.sum()


class GumbelDistributionLogits(GumbelDistribution):
    def get_logits(self):
        if self.dim_ip > 0:
            logits_raw = self.pi_marginal()
        else:
            logits_raw = self.pi_marginal

        if self.pi_layernorm:
            logits = self.pi_layernorm(logits_raw)
        else:
            logits = logits_raw
        return logits

    def get_pi(self, eps):
        logits = self.get_logits()
        logits = self.pi_dropout(logits)
        pi = F.softmax(logits, dim=1)
        return pi, logits, logits


class GumbelDistributionSoftPlus(GumbelDistribution):
    def get_pi(self, eps):
        if self.dim_ip > 0:
            pi_raw = self.pi_marginal()
        else:
            pi_raw = self.pi_marginal

        if self.pi_layernorm:
            pi_raw = self.pi_layernorm(pi_raw)

        pi = self.pi_dropout(pi_raw)
        pi = F.softplus(pi) + eps
        pi = pi / (pi.sum(dim=1).unsqueeze(1))  # sum to 1
        logits = torch.log(pi)
        return pi, logits, pi_raw


class GumbelDistributionSoftmaxAct(GumbelDistribution):
    def get_pi(self, eps):
        if self.dim_ip > 0:
            pi_raw = self.pi_marginal()
        else:
            pi_raw = self.pi_marginal

        if self.pi_layernorm:
            pi_raw = self.pi_layernorm(pi_raw)

        pi = self.pi_dropout(pi_raw)
        pi = F.softmax(pi, dim=1)
        logits = torch.log(pi + 1e-8)
        return pi, logits, pi_raw


class GumbelDistributionBijectiveSoftmax(GumbelDistribution):
    def __init__(
        self,
        num_categories,
        marginal_initialization="uniform",
        num_distributions=1,
        dim_ip=0,
        pi_dropout=0,
        IP_dropout=0,
        layer_norm=False,
        shared_weights=True,
    ):
        super().__init__(
            num_categories - 1,
            marginal_initialization,
            num_distributions,
            dim_ip,
            pi_dropout,
            IP_dropout,
            layer_norm,
            shared_weights,
        )

    def get_pi(self, eps):
        if self.dim_ip > 0:
            pi_raw = self.pi_marginal()
        else:
            pi_raw = self.pi_marginal

        if self.pi_layernorm:
            pi_raw = self.pi_layernorm(pi_raw)

        pi_raw = self.pi_dropout(pi_raw)

        x_pad = torch.nn.functional.pad(pi_raw, pad=(0, 1, 0, 0), value=0.0)
        pi = torch.nn.functional.softmax(x_pad, dim=-1)

        logits = torch.log(pi + 1e-8)
        return pi, logits, pi_raw

    def invert_pi(self, pi_normalized):
        x = torch.log(pi_normalized)
        x, log_normalization = torch.split(x, [self.num_categories, 1], dim=-1)
        logits_recovered = x - log_normalization
        return logits_recovered


if __name__ == "__main__":
    # from utils import get_num_parameters

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    random = True
    marginal_initialization = "random"

    ## FFHQ
    # img_size = 128
    # patch_size = 16
    # mask_ratio = 0.75
    # MNIST
    img_size = 28
    patch_size = 2
    mask_ratio = 0.9
    ## Synthetic
    # img_size = 50
    # patch_size = 5
    # mask_ratio = 0.85

    num_patches = int(img_size**2 / patch_size**2)
    assert img_size**2 / patch_size**2 % 1 == 0
    num_obs = int((1 - mask_ratio) * num_patches)

    num_batches = 1
    num_distributions = num_obs
    # dim_ip = 150
    dim_ip = 0
    # IP_hiddens = [78, 78, 78]
    # IP_hiddens = [100]
    IP_hiddens = []
    rao_samples = 0
    hard = False
    freeze = True
    shared_weights = False

    print("------------------------------JOINT----------------------------")
    distrib = GumbelDistributionLogits(
        num_patches,
        num_distributions=num_distributions,
        marginal_initialization=marginal_initialization,
        dim_ip=dim_ip,
        shared_weights=shared_weights,
    ).to(device)
    distrib.freeze(freeze)
    pi = distrib.pi_marginal
    print(pi)
    print(pi.shape)
    print(pi.sum(dim=1))
    batch_sampler = distrib.batch_sample_joint(
        num_batches,
        1.0,
        random=random,
        hard=hard,
        rao_samples=rao_samples,
        ret_EEG=False,
    )
    sampler_first_image = batch_sampler[0]
    print(sampler_first_image)
    print(sampler_first_image.shape)
    print(sampler_first_image.sum(dim=-1))

    pi_norm, _, pi_raw = distrib.get_pi(1e-8)
    print("Raw params", pi_raw)
    # pi_raw_recov = distrib.invert_pi(pi_norm)
    # print("Rocovered raw params", pi_raw_recov)

    print("------------------------------ctp----------------------------")
    ctp_pi = IP(
        num_patches,
        num_vectors=num_distributions,
        dim_ip=dim_ip,
        marginal_initialization=marginal_initialization,
        hiddens=[],
    )
    print(ctp_pi)
    print(ctp_pi())
    # print("Num params ctp = ", get_num_parameters(ctp_pi))
