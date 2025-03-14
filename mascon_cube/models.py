from typing import Optional, Union

import torch

from mascon_cube.data.mesh import get_mesh, is_outside_torch, unpack_triangle_mesh


class MasconCube:
    def __init__(
        self,
        cube_side: int,
        asteroid_name: Optional[str] = None,
        device: Union[str, torch.device] = "cuda",
        differential: bool = False,
    ):
        linspace = torch.linspace(-1, 1, cube_side)
        x, y, z = torch.meshgrid(linspace, linspace, linspace, indexing="ij")
        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)
        z = z.reshape(-1, 1)
        coords = torch.concat((x, y, z), dim=1).to(device)
        if asteroid_name:
            mesh_vertices, mesh_triangles = get_mesh(asteroid_name)
            triangles = unpack_triangle_mesh(
                mesh_vertices, mesh_triangles, device=device
            )
            outside_mask = is_outside_torch(coords, triangles)
            inside_mask = ~outside_mask
            coords = coords[inside_mask]
        n_points = len(coords)
        uniform_base_mass = 1 / n_points
        weights = (torch.rand((n_points, 1)) * 2 - 1) * uniform_base_mass / 10
        weights = weights.to(device).requires_grad_(True)

        self.coords = coords
        self.weights = weights
        self.device = device
        self.uniform_base_mass = uniform_base_mass
        self.differential = differential
        self._hparams = {
            "cube_side": cube_side,
            "asteroid_name": asteroid_name,
            "differential": differential,
        }

    @property
    def masses(self):
        return (
            self.weights
            if not self.differential
            else self.weights + self.uniform_base_mass
        )

    def get_hparams(self):
        return self._hparams
