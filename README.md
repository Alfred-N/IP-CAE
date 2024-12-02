# Indirectly Parameterized Concrete Autoencoders (IP-CAE)
Official PyTorch implementation of the ICML 2024 paper

> **Indirectly Parameterized Concrete Autoencoders**\
> Alfred Nilsson, Klas Wijk, Sai bharath chandra Gutha, Erik Englesson, Alexandra Hotti, Carlo Saccardi, Oskar Kviman, Jens Lagergren, Ricardo Vinuesa, Hossein Azizpour\
> https://arxiv.org/abs/2403.00563
>
> Abstract: Feature selection is a crucial task in settings where data is high-dimensional or acquiring the full set of features is costly. Recent developments in neural network-based embedded feature selection show promising results across a wide range of applications. Concrete Autoencoders (CAEs), considered state-of-the-art in embedded feature selection, may struggle to achieve stable joint optimization, hurting their training time and generalization. In this work, we identify that this instability is correlated with the CAE learning duplicate selections. To remedy this, we propose a simple and effective improvement: Indirectly Parameterized CAEs (IP-CAEs). IP-CAEs learn an embedding and a mapping from it to the Gumbel-Softmax distributions' parameters. Despite being simple to implement, IP-CAE exhibits significant and consistent improvements over CAE in both generalization and training time across several datasets for reconstruction and classification. Unlike CAE, IP-CAE effectively leverages non-linear relationships and does not require retraining the jointly optimized decoder. Furthermore, our approach is, in principle, generalizable to Gumbel-Softmax distributions beyond feature selection.

![Indirect parametrization](./imgs/indirect.svg#gh-light-mode-only)
![Indirect parametrization](./imgs/indirect_bg.svg#gh-dark-mode-only)

## Install requirements (Conda)
`conda env create -f environment.yml`

`conda activate IPCAE`

## Framework/structure
We utilize Pytorch Lightning. We define the models in `src/models/`, and wrap them with Lightning modules that contain the training code. The training wrappers are found in `src/pl_wrappers.py`.

A number of callbacks such as for annealing the LR and temperature are found in `src/pl_callbacks.py`. They can be toggeled by providing the corresponding argument, i.e. `--anneal_temp=True`.

## Logging
We use WandB for detailed logging.

**Either** sign in to your account
`wandb login`
and set the arguments

`python src/main_pl.py --config=<your-config>  --wandb=<your-project> --wandb_entity=<your-entity>`

**or** run the main script `src/main_pl.py` directly and select:
* option "(1) Private W&B dashboard, no account required" (requires internet connection)
or
* option "(4) Don’t visualize my results" (offline)

`python src/main_pl.py --config=<your-config>`

## Train and evaluate
Example command for reproducing MNIST results:

Reconstruction:

`python src/main_pl.py --config=configs/MNIST/base.yaml --dim_ip=784 --seed=11 --accelerator=cpu`

Classification:

`python src/main_pl.py --config=configs/MNIST/classification.yaml  --dim_ip=784 --seed=11`

## Arguments, configs and priority
The full list of arguments and their descriptions can be found in `src/parse_args.py`. These arguments are the same as the arguments in the config files. You can adjust the configs files or provide the arguments from the command line.

**Priority**: Provided command line args > config values > argparse defaults

## Datasets
To facilitate reproducibility, we include a script `src/fs_datasets.py` for downloading all datasets used in this paper, which includes functions that return the exact train/test/validation splits that were used. The data will be automatically downloaded into `--data_root_dir` when running `src/main_pl.py`

## Model architectures
The exact architectures for each dataset are defined at the end of `src/models/cae.py`.

## Citation

```
@inproceedings{nilsson2024indirectly,
      title={Indirectly Parameterized Concrete Autoencoders}, 
      author={Alfred Nilsson and Klas Wijk and Sai bharath chandra Gutha and Erik Englesson and Alexandra Hotti and Carlo Saccardi and Oskar Kviman and Jens Lagergren and Ricardo Vinuesa and Hossein Azizpour},
      booktitle={International Conference on Machine Learning},
      year={2024}
}
```
