from itertools import product

import torch

from mascon_cube.constants import VAL_DATASETS_DIR
from mascon_cube.logs import LogConfig
from mascon_cube.training import TrainingConfig, ValidationConfig, training_loop


def multi_train(
    asteroids: list[str],
    cube_sides: list[int] = [100],
    n_epochs: list[int] = [500],
    n_epochs_before_resampling: list[int] = [10],
    loss_fns: list[str] = ["normalized_l1_loss"],
    batch_sizes: list[int] = [1000],
    sampling_methods: list[str] = ["spherical"],
    sampling_mins: list[float] = [0.5],
    sampling_maxs: list[float] = [1.5],
    lrs: list[float] = [1e-6],
    scheduler_factors: list[float] = [0.8],
    scheduler_patience: list[int] = [200],
    scheduler_min_lrs: list[float] = [1e-8],
    differentials: list[bool] = [False],
    normalize: list[bool] = [True],
    quadratic: list[bool] = [False],
):
    param_grid = product(
        asteroids,
        cube_sides,
        n_epochs,
        n_epochs_before_resampling,
        loss_fns,
        batch_sizes,
        sampling_methods,
        sampling_mins,
        sampling_maxs,
        lrs,
        scheduler_factors,
        scheduler_patience,
        scheduler_min_lrs,
        differentials,
        normalize,
        quadratic,
    )
    for params in param_grid:
        config = TrainingConfig(*params)
        train(config)


def train(config):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    val_dataset = torch.load(VAL_DATASETS_DIR / "itokawa_lp_1000_spherical_0_2.pt").to(
        device
    )
    val_config = ValidationConfig(val_dataset=val_dataset, val_every_n_epochs=50)
    log_config = LogConfig()

    # Train the cube
    training_loop(config, val_config, log_config, device=device)


if __name__ == "__main__":
    multi_train(
        asteroids=["itokawa_lp"],
        n_epochs=[1000],
        batch_sizes=[100, 1000],
        quadratic=[True, False],
        lrs=[1e-5, 1e-4],
        loss_fns=["normalized_l1_loss", "l1_loss"],
    )
