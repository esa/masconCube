# masconCube

[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)

![image](https://github.com/user-attachments/assets/115f03f9-f65c-40c4-a357-b9cdc53eeee2)

## Installation

```bash
conda env create -f environment.yml
conda activate masconcube
```

Or, if you want to install also development dependencies:

```bash
conda env create -f environment_dev.yml
conda activate masconcube
```

Development dependencies include packages for linting and contributing to the project.

## Reprouducing the results from the paper

 1. Download the 3D meshes from [darioizzo/geodesyNets/3dmeshes]([darioizzo/geodesyNets/3dmeshes](https://github.com/darioizzo/geodesyNets/tree/1edbb64d1e8e355e124a41eac27a14d7c5c5d881/3dmeshes)) and copy them inside the `data/3dmeshes` folder. For more information, see the [data README](data/README.md).
 2. Generate the ground-truth mascon models in the `data/ground_truths` folder by running the following script:
    ```bash
    python scripts/generate_ground_truth.py
    ```
3. Generate the validation datasets in the `data/val_datasets` and `data/test_datasets` folders by running the following script:
    ```bash
    python scripts/generate_val_datasets.py
    ```
4. Train MasconCubes with the following command:
    ```bash
    python scripts/train_cubes_all.py [--gpus <gpu1> <gpu2> ...]
    ```
5. Train GeodesyNets with the following command:
    ```bash
    python scripts/train_geodesynet_all.py [--gpus <gpu1> <gpu2> ...]
    ```
6. Train PINN-GM III with the following command:
    ```bash
    python scripts/train_pinn_all.py [--gpus <gpu1> <gpu2> ...]
    ```
7. Evaluate the models and produce plots using the provided notebooks.

Note that steps 5 and 6 are required only to compare the results with previous state-of-the-art methods, and they might take a long time to run. You can skip them if you are only interested in MasconCube. If you want to run them, multiple GPUs are recommended, so that you can run them in parallel. The `--gpus` argument allows you to specify which GPUs to use for training.

### Training on single asteroids

If you want to run the training on single asteroids, you can use the scripts `scripts/train.py`, `scripts/train_geodesynet.py`, and `scripts/train_pinn.py`. For example, to train MasconCube on `eros_uniform`, you can run:

```bash
python scripts/train.py eros_uniform
```

MasconCube trainings also support TensorBoard logging (development dependencies required). You can run the following command to start TensorBoard:

```bash
tensorboard --logdir runs
```

And then enable logging in the training script by passing the `--tensorboard` argument:

```bash
python scripts/train.py eros_uniform --tensorboard
```