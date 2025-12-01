# MasconCube

[![arXiv](https://img.shields.io/badge/arXiv-2509.08607-b31b1b.svg)](https://arxiv.org/abs/2509.08607)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)

Official codebase for the paper _[MasconCube: Fast and Accurate Gravity Modeling with an Explicit Representation](https://arxiv.org/abs/2509.08607v1)_.

MasconCube is a fast and accurate framework for modeling gravitational fields of irregular celestial bodies. It provides an explicit mascon-based representation, enabling efficient training, validation, and comparison against state-of-the-art methods such as GeodesyNets and PINN-GM.

<img width="3387" height="1180" alt="image" src="https://github.com/user-attachments/assets/3e568581-46e5-4624-b102-c92d6f6772df" />

## Installation

```bash
conda env create -f environment.yml
conda activate masconcube
```

## Project structure

```
masconCube/
├── data/                           # Data folder
│   ├── 3dmeshes/                   # 3D meshes (to be downloaded from darioizzo/geodesyNets)
│   ├── ground_truths/              # Ground-truth mascon models (to be generated with scripts/generate_ground_truth.py)
│   ├── output/                     # Output folder for trained models and results
|   ├── train_configs/              # Training configurations for MasconCube (yaml files)
|   ├── train_datasets/             # Training datasets (to be generated with scripts/generate_train_datasets.py)
|   ├── val_datasets/               # Validation datasets (to be generated with scripts/generate_val_datasets.py)
|   └── test_datasets/              # Test datasets (to be generated with scripts/generate_val_datasets.py)
├── mascon_cube/                    # Main package
│   ├── data/                       # Data loading and processing
|   ├── geodesynet/                 # GeodesyNets original implementation, mostly unmodified
|   ├── pinn_gm/                    # PINN-GM III original implementation, mostly unmodified
│   ├── contstants.py               
│   ├── losses.py                   
│   ├── metrics.py
│   ├── models.py
│   ├── training.py
│   ├── trajectory.py
│   ├── utils.py
│   └── visualization.py
├── notebooks/                      # Jupyter notebooks                 
├── scripts/                        # Scripts for data generation, training, and evaluation
└── environment.yml                 # Conda environment file
```

## Data

Data is not stored in this repository to keep the size of the repository small. You can populate the `data` folder by following these steps:

 1. Download the 3D meshes from [darioizzo/geodesyNets/3dmeshes]([darioizzo/geodesyNets/3dmeshes](https://github.com/darioizzo/geodesyNets/tree/1edbb64d1e8e355e124a41eac27a14d7c5c5d881/3dmeshes)) and copy them inside the `data/3dmeshes` folder. For more information, see the [data README](data/README.md).
 2. Generate the ground-truth mascon models in the `data/ground_truths` folder by running the following script:
    ```bash
    python scripts/generate_ground_truth.py
    ```
 3. Generate the training datasets using in the paper in the `data/train_datasets` folders by running the following script:
    ```bash
    python scripts/generate_train_datasets.py random
    ```
 4. Generate the validation datasets in the `data/val_datasets` and `data/test_datasets` folders by running the following script:
    ```bash
    python scripts/generate_val_datasets.py
    ```

## Training MasconCube and other models (legacy method)

MasconCube, GeodesyNet and PINN-GM III can be trained using their respective training scripts in the `scripts` folder: `train_cube.py`, `train_geodesynet.py`, and `train_pinn.py`.
The cli interface for all the training scripts is similar:
```bash
python scripts/train_<model>.py <asteroid_name>
```

It is also possible to train a model for every asteroid in `data/ground_truths` on multiple GPUs using the scripts `train_cubes_all.py`, `train_geodesynet_all.py`, and `train_pinn_all.py`.
Again, the cli interface is similar for all the scripts:
```bash
python scripts/train_<model>_all.py [--gpus <gpu1> <gpu2> ...]
```

Default settings should replicate the results in the paper. However some small implemention details could lead to slightly different results. If you want to replicate exactly the results of the paper checkout [v1.0.0](https://github.com/esa/masconCube/tree/v.1.0.0).

## Training MasconCube (new method)

The recommended way to train MasconCube models is with yaml configuration files stored in the `data/train_configs` folder.
The folder alredy contains a configuration file called `paper.yaml` that you can use as a starting point.
It use similar settings to the one used in the paper, but for an exact comparison follow the instructions in the previous section.

To train a MasconCube model using a configuration file, run the following command:
```bash
python scripts/run_config.py data/train_configs/<config_file>.yaml [--use-wandb]
```

The `--use-wandb` flag is optional, and it enables logging with [Weights & Biases](https://wandb.ai/).

## License
The code is released under the [Apache 2.0 license](https://github.com/esa/masconCube?tab=Apache-2.0-1-ov-file).

## Citation
If you find this repository useful, please kindly consider citing the following paper:

```bibtex
@misc{fanti2025masconcube,
      title={MasconCube: Fast and Accurate Gravity Modeling with an Explicit Representation}, 
      author={Pietro Fanti and Dario Izzo},
      year={2025},
      eprint={2509.08607},
      archivePrefix={arXiv},
      primaryClass={astro-ph.EP},
      url={https://arxiv.org/abs/2509.08607}, 
}
```

## Aknowledgements

This codebase is based on the following open-source projects. We thank their authors for making the source code publically available.

 - [geodesyNets](https://github.com/darioizzo/geodesyNets/tree/master)
