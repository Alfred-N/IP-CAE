# Indirectly Parameterized Concrete Autoencoders (IP-CAE)

#### Install requirements (Conda)
`conda env create -f environment.yml`

`conda activate IPCAE`

### Framework/structure
We utilize Pytorch Lightning. We define the models in `src/models/`, and wrap them with Lightning modules that contain the training code. The training wrappers are found in `src/pl_wrappers.py`.

A number of callbacks such as for annealing the LR and temperature are found in `src/pl_callbacks.py`. They can be toggeled by providing the corresponding argument, i.e. `--anneal_temp=True`.

### Logging
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

### Train and evaluate
Example command for reproducing MNIST results:

Reconstruction:

`python src/main_pl.py --config=configs/MNIST/base.yaml --dim_ip=784 --seed=11`

Classification:

`python src/main_pl.py --config=configs/MNIST/classification.yaml  --dim_ip=784 --seed=11`



### Arguments, configs and priority
The full list of arguments and their descriptions can be found in `src/parse_args.py`. These arguments are the same as the arguments in the config files. You can adjust the configs files or provide the arguments from the command line.

**Priority**: Provided command line args > config values > argparse defaults

### Datasets
To facilitate reproducibility, we include a script `src/fs_datasets.py` for downloading all datasets used in this paper, which includes functions that return the exact train/test/validation splits that were used. The data will be automatically downloaded into `--data_root_dir` when running `src/main_pl.py`

### Model architectures
The exact architectures for each dataset are defined at the end of `src/models/cae.py`.
