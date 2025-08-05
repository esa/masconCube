import copy
import pickle as pk
from pathlib import Path
from typing import Union

import tetgen
import torch

from mascon_cube.constants import GROUND_TRUTH_DIR
from mascon_cube.data.mesh import get_mesh, points_in_tetrahedra_torch
from mascon_cube.models import MasconCube


class MasconModel:
    def __init__(
        self,
        mascon_name: Union[str, Path],
        device: Union[str, torch.device],
        uniform: bool = False,
    ):
        if isinstance(mascon_name, str):
            if Path(mascon_name).exists():
                mascon_path = Path(mascon_name)
            else:
                file_name = (
                    "mascon_model.pk" if not uniform else "mascon_model_uniform.pk"
                )
                mascon_path = GROUND_TRUTH_DIR / mascon_name / file_name
        else:
            mascon_path = mascon_name
        assert mascon_path.exists(), f"Mesh file {mascon_path} does not exist"

        with open(mascon_path, "rb") as f:
            mascon_points, mascon_masses = pk.load(f)
        self.coords = torch.tensor(mascon_points, device=device)
        self.masses = torch.tensor(mascon_masses, device=device)
        self.device = self.coords.device
        self.mascon_name = mascon_path.parent.stem

    def to(self, device: Union[str, torch.device]) -> "MasconModel":
        """Move the model to a different device

        Args:
            device (Union[str, torch.device]): The device to move the model to

        Returns:
            MasconModel: The model on the new device
        """
        new_model = copy.deepcopy(self)
        new_model.coords = self.coords.to(device)
        new_model.masses = self.masses.to(device)
        new_model.device = device
        return new_model

    def to_cube(self, cube_side: int) -> MasconCube:
        """Convert the mascon model to a mascon cube

        Args:
            cube_side (int): the side of the output cube

        Returns:
            MasconCube: The mascon cube
        """
        # We do it on cpu as it requires a lot of memory
        cube = MasconCube(cube_side, self.mascon_name, device="cpu", differential=False)
        mesh_points, mesh_triangles = get_mesh(self.mascon_name)
        # Here we define the surface
        tgen = tetgen.TetGen(mesh_points, mesh_triangles)
        # Here we run the algorithm to mesh the inside with thetrahedrons
        nodes, elem = tgen.tetrahedralize()
        nodes = torch.tensor(nodes, device="cpu")
        elem = torch.tensor(elem, device="cpu")
        indeces = points_in_tetrahedra_torch(cube.coords, nodes, elem)
        grid = tgen.grid
        grid = grid.compute_cell_sizes(volume=True, area=False, length=False)
        volumes = torch.tensor(grid["Volume"], device="cpu")
        masses = self.masses[indeces].cpu() / volumes[indeces]
        cube.weights = masses / masses.sum()
        # We move it back to the original device
        return cube.to(self.device)

    def get_volume(self) -> float:
        """Get the volume of the mascon. It is computed only once and then cached.

        Returns:
            float: The volume of the mascon
        """
        if not hasattr(self, "_volume"):
            mesh_points, mesh_triangles = get_mesh(self.mascon_name)
            # Here we define the surface
            tgen = tetgen.TetGen(mesh_points, mesh_triangles)
            # Here we run the algorithm to mesh the inside with thetrahedrons
            nodes, elem = tgen.tetrahedralize()
            nodes = torch.tensor(nodes, device=self.device)
            elem = torch.tensor(elem, device=self.device)
            grid = tgen.grid
            grid = grid.compute_cell_sizes(volume=True, area=False, length=False)
            volumes = torch.tensor(grid["Volume"], device=self.device)
            self._volume = volumes.sum().item()
        return self._volume

    def get_average_density(self) -> float:
        """Get the average density of the mascon.

        Returns:
            float: The average density of the mascon
        """
        return self.masses.sum().item() / self.get_volume()
