import pickle as pk
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import torch

from mascon_cube.constants import GROUND_TRUTH_DIR


@dataclass
class MasconModel:
    """Dataclass for MasconModel"""

    coords: torch.Tensor
    masses: torch.Tensor


def get_mascon_model(
    mascon_name: Union[str, Path], device: Union[str, torch.device]
) -> MasconModel:
    """Get the mascon model from the ground truth directory

    Args:
        mascon_name (Union[str, Path]): The name of the mesh file or the path to the mesh file
        device (Union[str, torch.device]): The device to use

    Returns:
        MasconModel: The mascon model
    """
    if isinstance(mascon_name, str):
        if Path(mascon_name).exists():
            mascon_path = Path(mascon_name)
        else:
            mascon_path = GROUND_TRUTH_DIR / f"{mascon_name}.pk"
    else:
        mascon_path = mascon_name
    assert mascon_path.exists(), f"Mesh file {mascon_path} does not exist"

    with open(mascon_path, "rb") as f:
        mascon_points, mascon_masses = pk.load(f)
    mascon_model = MasconModel(
        torch.tensor(mascon_points, device=device),
        torch.tensor(mascon_masses, device=device),
    )
    return mascon_model
