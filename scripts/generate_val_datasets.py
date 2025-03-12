from pathlib import Path
from typing import Union

import torch

from mascon_cube.constants import VAL_DATASETS_DIR
from mascon_cube.data.sampling import get_target_point_sampler


def gen_validation_dataset(
    n_points: int, asteroid: Union[str, Path], method: str, bounds: tuple[float, float]
):
    points_sampler = get_target_point_sampler(n_points, asteroid, method, bounds, "cpu")
    dataset = points_sampler()
    dataset_path = (
        VAL_DATASETS_DIR / f"{asteroid}_{n_points}_{method}_{bounds[0]}_{bounds[1]}.pt"
    )
    torch.save(dataset, dataset_path)


if __name__ == "__main__":
    torch.random.manual_seed(42)
    gen_validation_dataset(1000, "itokawa_lp", "spherical", (0, 2))
