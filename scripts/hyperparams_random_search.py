import math
import random
from argparse import ArgumentParser

import torch
from tqdm import tqdm

from mascon_cube.constants import VAL_DATASETS_DIR
from mascon_cube.logs import LogConfig
from mascon_cube.training import CubeTrainingConfig, ValidationConfig, training_loop

N_SAMPLES = 1000 * 1000 / 10  # n_epochs * batch_size / n_epochs_before_resampling

RANGES = {
    "lr": (1e-8, 1e-4),
    "scheduler_factor": (0.5, 1),
    "scheduler_patience": (100, 300),
    "scheduler_min_lr": (1e-12, 1e-8),
    "batch_size": (100, 1200),
    "n_epochs": (500, 2000),
    "n_epochs_before_resampling": (1, 20),
}


def find_triplet(max_attempts=10000):
    for i in range(max_attempts):
        batch_size = random.randrange(
            RANGES["batch_size"][0], RANGES["batch_size"][1] + 1, 25
        )
        for _ in range(
            10
        ):  # nested attempts to avoid repeating top-level attempts too much
            n_epochs = random.randrange(
                RANGES["n_epochs"][0], RANGES["n_epochs"][1] + 1, 25
            )
            if (batch_size * n_epochs) % N_SAMPLES > 0:
                continue
            n_epochs_before_resampling = batch_size * n_epochs // N_SAMPLES
            if (
                RANGES["n_epochs_before_resampling"][0]
                <= n_epochs_before_resampling
                <= RANGES["n_epochs_before_resampling"][1]
            ):
                return batch_size, n_epochs, n_epochs_before_resampling
    return None


def log_uniform(a, b):
    return math.exp(random.uniform(math.log(a), math.log(b)))


def search(n_runs: int, asteroid: str):
    device = "cuda"
    val_dataset = torch.load(
        VAL_DATASETS_DIR / f"{asteroid.split('_')[0]}_lp_1000_spherical_0_2.pt"
    ).to(device)
    for run in tqdm(range(n_runs)):
        lr = log_uniform(*RANGES["lr"])
        scheduler_factor = random.uniform(*RANGES["scheduler_factor"])
        scheduler_min_lr = log_uniform(*RANGES["scheduler_min_lr"])
        scheuler_patience = random.randint(*RANGES["scheduler_patience"])
        batch_size, n_epochs, n_epochs_before_resampling = find_triplet()
        config = CubeTrainingConfig(
            asteroid=asteroid,
            lr=lr,
            scheduler_factor=scheduler_factor,
            scheduler_min_lr=scheduler_min_lr,
            batch_size=batch_size,
            n_epochs=n_epochs,
            n_epochs_before_resampling=n_epochs_before_resampling,
            scheduler_patience=scheuler_patience,
        )
        log_config = LogConfig()
        val_config = ValidationConfig(val_dataset=val_dataset)
        training_loop(config, val_config, log_config, progressbar=False, device=device)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("asteroid", type=str)
    parser.add_argument("--n-runs", "-n", type=int)
    args = parser.parse_args()
    search(args.n_runs, args.asteroid)
