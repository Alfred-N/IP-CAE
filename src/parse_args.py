import argparse
import os
import datetime
import yaml


def get_args_parser(conf_parser):
    parser = argparse.ArgumentParser("Train CAE", parents=[conf_parser])
    parser.add_argument(
        "--batch_size",
        default=64,
        type=int,
        help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus",
    )
    parser.add_argument("--epochs", default=400, type=int)
    parser.add_argument(
        "--accum_iter",
        default=1,
        type=int,
        help="Accumulate gradient iterations (for increasing the effective batch size under memory constraints)",
    )

    # Model parameters
    parser.add_argument(
        "--model",
        default="cae_MLP_MNIST",
        type=str,
        metavar="MODEL",
        help="Name of model to train",
    )

    parser.add_argument(
        "--input_height", default=224, type=int, help="images input height"
    )
    parser.add_argument(
        "--input_width", default=224, type=int, help="images input width"
    )

    parser.add_argument(
        "--mask_ratio",
        default=None,
        type=float,
        help="Masking ratio (percentage of removed features).  Mutually exlusive with '--k'",
    )

    parser.add_argument(
        "--k",
        default=None,
        type=int,
        help="The number of selected features. Mutually exlusive with '--mask_ratio'",
    )

    parser.add_argument(
        "--norm_pix_loss",
        type=bool,
        default=False,
        help="Use (per-patch) normalized pixels as targets for computing loss",
    )

    # Optimizer parameters
    parser.add_argument(
        "--weight_decay", type=float, default=0, help="weight decay (default: 0)"
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        metavar="LR",
        help="learning rate (absolute lr)",
    )
    parser.add_argument(
        "--blr",
        type=float,
        default=1e-4,
        metavar="LR",
        help="base learning rate: absolute_lr = base_lr * total_batch_size / 256",
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=1e-07,
        metavar="LR",
        help="lower lr bound for cyclic schedulers that hit 0",
    )
    parser.add_argument(
        "--anneal_lr",
        type=str,
        choices=["none", "cosine", "warmup"],
        default=True,
        help="Turn on annealing lr with warmup. 'cosine'=warmup followed by cosine annealing. 'warmup' = warmup only, thereafter fixed lr. ",
    )
    parser.add_argument(
        "--scale_lr_by_batchsize",
        type=bool,
        default=False,
        help="Turn on lr scaling wrt. batch size. Multiplies lr by (eff_batch_size / 256)",
    )

    parser.add_argument(
        "--layer_decay",
        type=float,
        default=1,
        help="layer-wise lr decay factor",
    )

    parser.add_argument(
        "--warmup_epochs", type=int, default=40, metavar="N", help="epochs to warmup LR"
    )

    # Dataset parameters
    parser.add_argument(
        "--data_root_dir",
        default="data/",
        type=str,
        help="path of the root directory containing all datasets",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        choices=["mnist", "mnist_fashion", "isolet", "coil20", "activity", "mice"],
        type=str,
        help="String deciding which dataset to load/download",
    )
    parser.add_argument(
        "--watch_grads",
        type=int,
        default=0,
        help="log grads to wandb",
    )
    parser.add_argument(
        "--save_snapshots",
        type=int,
        default=0,
        help="saves snapshots of the gumbel distribs that can be later turned into a .gif",
    )
    parser.add_argument(
        "--output_dir",
        default="./output_dir",
        help="path where to save, empty for no saving",
    )
    parser.add_argument(
        "--save_top_k",
        type=int,
        default=1,
        help="saves top-K checkpoints based on 'val_loss' metric",
    )
    parser.add_argument(
        "--save_last",
        type=bool,
        default=True,
        help="saves checkpoint from last epoch",
    )
    parser.add_argument(
        "--every_n_epochs",
        type=int,
        default=None,
        help="Number of epochs between checkpoints",
    )
    parser.add_argument(
        "--log_dir", default="./output_dir", help="path where to tensorboard log"
    )
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")

    parser.add_argument(
        "--start_epoch", default=0, type=int, metavar="N", help="start epoch"
    )
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument(
        "--pin_mem",
        type=bool,
        default=True,
        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
    )
    # distributed training parameters
    parser.add_argument(
        "--accelerator",
        type=str,
        choices=["cpu", "gpu", "mps"],
        default="gpu",
        help="Specify the accelerator (cpu or gpu)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        choices=["16-mixed", "bf16-mixed", "32-true", "64-true"],
        default="32-true",
        help=" Double precision, full precision (32), 16bit mixed precision or bfloat16 mixed precision",
    )
    parser.add_argument(
        "--matmul_precision",
        type=str,
        choices=["medium", "high", None],
        default=None,
        help="To fully exploit NVIDIA A100 GPUs, set torch.set_float32_matmul_precision('medium' | 'high') which trades off precision for performance",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["ddp", "deepspeed"],
        default="ddp",
        help="Specify the distributed strategy",
    )
    parser.add_argument(
        "--num_devices", default=1, type=int, help="number of distributed processes"
    )
    parser.add_argument("--num_nodes", default=1, type=int, help="number of nodes")
    parser.add_argument(
        "--anneal_temp", type=str, default=False, help="Turn on temperature annealing"
    )
    parser.add_argument("--temp_min", type=float, default=0.1, help="")
    parser.add_argument("--temp_base", type=float, default=1.0, help="")
    parser.add_argument(
        "--anneal_thresh",
        type=str,
        default=False,
        help="Turn on threshold annealing for EEG reg.",
    )
    parser.add_argument("--thresh_base", type=float, default=3.0, help="")
    parser.add_argument("--thresh_min", type=float, default=1.1, help="")
    parser.add_argument(
        "--stop_anneal",
        type=int,
        default=0,
        help="The epoch to stop annealing at (<=0 means no stop anneal)",
    )
    parser.add_argument(
        "--switch_st",
        type=int,
        default=0,
        help="The epoch to switch to straight-through (<=0 means no stop anneal)",
    )
    parser.add_argument(
        "--rao_samples",
        default=0,
        type=int,
        help="If > 1, use Rao Blackweilization of the STGS gradient estimator",
    )
    parser.add_argument(
        "--straight_through",
        default=False,
        type=bool,
        help="If True, STGS (hard) samples, otherwise used relaxed samples like CAE",
    )
    parser.add_argument(
        "--full_batch",
        default=False,
        type=bool,
        help="If True, use full batch gradient descent",
    )
    parser.add_argument("--clip_grad", type=float, default=None, help="")
    parser.add_argument("--debug", type=bool, default=False, help="Debug with subset")
    parser.add_argument(
        "--wandb", default=None, help="Specify project name to log using WandB"
    )
    parser.add_argument(
        "--wandb_entity", default=None, help="Entity to log as on WandB"
    )
    parser.add_argument(
        "--dim_ip",
        type=int,
        default=0,
        help="Dimension of ip vectors to use for distrib (<=0 means no ip inputs)",
    )
    parser.add_argument(
        "--local_logging",
        type=int,
        default=0,
        help="Enable local logging (to --log_dir)",
    )
    parser.add_argument(
        "--marginal_initialization",
        choices=["random", "uniform", "identity", "grid"],
        default="random",
        help="Initialization of alpha/log alpha depending on activation",
    )
    parser.add_argument(
        "--IP_initialization",
        choices=["random", "identity"],
        default="random",
        help="Initialization of the IP vectors (P matrix)",
    )
    parser.add_argument(
        "--IP_weights",
        choices=["shared", "separate", "fc", "diag", "scalar"],
        default="shared",
        type=str,
        help="The type of transformation network of the IP vectors",
    )
    parser.add_argument(
        "--IP_bias",
        type=int,
        default=1,
        help="If true, inlcude a bias after the transformation in IPs",
    )
    parser.add_argument(
        "--gumbel_learn_mode",
        choices=["pi", "softmax", "logits", "bij_softmax"],
        default="pi",
        help="If to learn pi_raw or the unnorm. logits l_u",
    )
    parser.add_argument(
        "--freeze_epochs",
        type=int,
        default=None,
        help="The number of epochs for which the GumbelDistrib-module will "
        "be frozen (i.e. the first 'args.freeze_epochs' epochs)",
    )

    parser.add_argument(
        "--freeze_distrib",
        type=bool,
        default=False,
        help="Permanently freeze the GumbelDistribution layer)",
    )

    parser.add_argument(
        "--finetune_path",
        default=None,
        type=str,
        help="Used by finetune_pl.py. Path of the checkpoint to finetune",
    )

    parser.add_argument(
        "--classification_mode",
        type=str,
        choices=["finetune", "linprobe", "scratch"],
        default="gpu",
        help="Used by finetune_pl.py",
    )

    parser.add_argument(
        "--pi_dropout",
        type=float,
        default=0,
        help="Probability of dropping out an element of pi",
    )
    parser.add_argument(
        "--IP_dropout",
        type=float,
        default=0,
        help="Probability of dropping out an element of the IP vectors",
    )
    parser.add_argument(
        "--jsd_factor",
        type=float,
        default=0,
        help="Scales the contribution of the JSD to the loss",
    )
    parser.add_argument(
        "--eeg_factor",
        type=float,
        default=0,
        help="Scales the contribution of the EEG regularization to the loss",
    )
    parser.add_argument(
        "--pi_layer_norm",
        type=bool,
        default=False,
        help="If to apply layernorm on pi (before squaring and normalizing)",
    )
    # parser.add_argument( #TODO: add callback for this
    #     "--save_timelapse",
    #     type=bool,
    #     default=False,
    #     help="Save timelapse of pi",
    # )
    parser.add_argument(
        "--subset",
        type=float,
        default=None,
        help="0<value<1 if to train with a subset of the data",
    )
    parser.add_argument(
        "--use_masked_loss",
        type=bool,
        default=True,
        help="If True=> mask the loss",
    )
    parser.add_argument(
        "--loss_type",
        choices=["mse", "bce"],
        default="mse",
        help="MSE or Binary CE (latter only makes sense for binary targets)",
    )
    parser.add_argument(
        "--binarize",
        type=bool,
        default=False,
        help="binarize the data, only affects MNIST",
    )
    parser.add_argument(
        "--no_gumbel_noise",
        type=bool,
        default=False,
        help="flag to use gumbel noise during softmax",
    )
    return parser


def uniquify_path(path):
    # append a number
    counter = 1
    try_path = path
    if not os.path.exists(try_path):
        return path
    else:
        while os.path.exists(try_path):
            try_path = path + "__" + str(counter)
            counter += 1

        return try_path


def parse_args_and_config():
    ### Priority: provided command line args > config values > argparse defaults

    # parse the config arg only
    conf_parser = argparse.ArgumentParser("Config parser", add_help=False)
    conf_parser.add_argument(
        "--config",
        type=str,
        help="config path",
    )
    conf_args, remaining_args = conf_parser.parse_known_args()

    # open config file and set default args to those included in the config
    try:
        with open(conf_args.config, "r") as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        print("Error occurred while loading the configuration file:")
        print(e)

    parser = get_args_parser(conf_parser)
    parser.set_defaults(**config)

    # parse the rest of the args and override defaults/config
    args = parser.parse_args(remaining_args)
    return args


def create_output_dirs(args, is_main_process=True):
    if is_main_process:
        # append datetime str to output dirs
        now = datetime.datetime.now()
        cur_time = now.strftime("_%H_%M_%S_%f__%d_%m_%y")
        args.output_dir = args.output_dir + cur_time
        args.output_dir = uniquify_path(args.output_dir)
        os.makedirs(args.output_dir, exist_ok=False)
        args.log_dir = args.log_dir + cur_time
        args.log_dir = uniquify_path(args.log_dir)
        os.makedirs(args.log_dir, exist_ok=False)


if __name__ == "__main__":
    args = parse_args_and_config()
    create_output_dirs(args)
