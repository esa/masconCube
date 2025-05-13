from pathlib import Path
from typing import Union

import torch

from mascon_cube.constants import TEST_DATASETS_DIR, VAL_DATASETS_DIR


def load_dataset(
    name: Union[str, Path],
    test: bool = False,
    n: int = 1000,
    method: str = "spherical",
    bounds: tuple[int, int] = (0, 2),
    device: Union[str, torch.device] = "cpu",
) -> torch.Tensor:
    if isinstance(name, Path) or Path(name).exists():
        ds_path = name
    else:
        root = TEST_DATASETS_DIR if test else VAL_DATASETS_DIR
        ds_path = root / f"{name}_{n}_{method}_{bounds[0]}_{bounds[1]}.pt"
        assert ds_path.exists(), f"{ds_path} is not a valid path"
    ds = torch.load(ds_path).to(device)
    return ds
