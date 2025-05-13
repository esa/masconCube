import os
from pathlib import Path
from typing import Union

import torch

from mascon_cube.constants import GROUND_TRUTH_DIR, TEST_DATASETS_DIR, VAL_DATASETS_DIR
from mascon_cube.data.sampling import get_target_point_sampler


def gen_validation_dataset(
    n_points: int,
    asteroid: Union[str, Path],
    method: str,
    bounds: tuple[float, float],
    test: bool = False,
):
    points_sampler = get_target_point_sampler(n_points, asteroid, method, bounds, "cpu")
    dataset = points_sampler()
    dir = TEST_DATASETS_DIR if test else VAL_DATASETS_DIR
    dataset_path = dir / f"{asteroid}_{n_points}_{method}_{bounds[0]}_{bounds[1]}.pt"
    torch.save(dataset, dataset_path)


if __name__ == "__main__":
    asteroids = [f.name for f in os.scandir(GROUND_TRUTH_DIR) if f.is_dir()]
    # Generate validation
    torch.random.manual_seed(42)
    for asteroid in asteroids:
        gen_validation_dataset(1000, asteroid, "spherical", (0, 2))
    # Generate test
    torch.random.manual_seed(43)
    for asteroid in asteroids:
        gen_validation_dataset(1000, asteroid, "spherical", (0, 2), test=True)
