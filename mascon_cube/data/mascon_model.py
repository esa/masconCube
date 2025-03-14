import pickle as pk
from pathlib import Path
from typing import Union

import tetgen
import torch

from mascon_cube.constants import GROUND_TRUTH_DIR
from mascon_cube.data.mesh import get_mesh, points_in_tetrahedra_torch
from mascon_cube.models import MasconCube


class MasconModel:
    def __init__(self, mascon_name: Union[str, Path], device: Union[str, torch.device]):
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
        self.coords = torch.tensor(mascon_points, device=device)
        self.masses = torch.tensor(mascon_masses, device=device)
        self.device = self.coords.device
        self.mascon_name = mascon_path.stem

    def to(self, device: Union[str, torch.device]) -> "MasconModel":
        """Move the model to a different device

        Args:
            device (Union[str, torch.device]): The device to move the model to

        Returns:
            MasconModel: The model on the new device
        """
        return MasconModel(self.coords.to(device), self.masses.to(device))

    def to_cube(self, cube_side: int) -> MasconCube:
        """Convert the mascon model to a mascon cube

        Args:
            cube_side (int): the side of the output cube

        Returns:
            MasconCube: The mascon cube
        """
        cube = MasconCube(
            cube_side, self.mascon_name, device=self.device, differential=False
        )
        mesh_points, mesh_triangles = get_mesh(self.mascon_name)
        # Here we define the surface
        tgen = tetgen.TetGen(mesh_points, mesh_triangles)
        # Here we run the algorithm to mesh the inside with thetrahedrons
        nodes, elem = tgen.tetrahedralize()
        nodes = torch.tensor(nodes, device=self.device)
        elem = torch.tensor(elem, device=self.device)
        indeces = points_in_tetrahedra_torch(cube.coords, nodes, elem)
        grid = tgen.grid
        grid = grid.compute_cell_sizes(volume=True, area=False, length=False)
        volumes = torch.tensor(grid["Volume"], device=self.device)
        masses = self.masses[indeces] / volumes[indeces]
        cube.weights = masses / masses.sum()
        return cube


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
