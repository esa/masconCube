from typing import Optional, Union

import torch

from mascon_cube.data.mesh import get_mesh, is_outside_torch, unpack_triangle_mesh


class MasconCube:
    """A class representing a mascon cube."""

    def __init__(
        self,
        cube_side: int,
        asteroid_name: Optional[str] = None,
        device: Union[str, torch.device] = "cuda",
        differential: bool = False,
        normalize: bool = False,
        quadratic: bool = False,
    ):
        """Initialize the mascon cube.

        Args:
            cube_side (int): The side length of the cube.
            asteroid_name (Optional[str], optional): The name of the asteroid.
                If passed, the mascon cube will be generated inside the asteroid's mesh. Defaults to None.
            device (Union[str, torch.device], optional): The device to use. Defaults to "cuda".
            differential (bool, optional): if True, the uniform base mass is added to the weights. Defaults to False.
            normalize (bool, optional): if True, the masses are normalized so that they sum to 1. Defaults to False.
        """
        if differential and quadratic:
            raise ValueError(
                "differential and quadratic cannot be True at the same time"
            )

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
        self.normalize = normalize
        self.quadratic = quadratic

    @property
    def masses(self) -> torch.Tensor:
        """Return the masses of the mascon cube.
        If differential is True, the uniform base mass is added to the weights.
        If normalize is True, the masses are normalized so that they sum to 1.

        Returns:
            torch.Tensor: The masses of the mascon cube
        """
        result = self.weights
        if self.quadratic:
            result = result**2
        if self.differential:
            result = result + self.uniform_base_mass
        if self.normalize:
            result = result / result.sum()
        return result

    def get_hparams(self):
        return self._hparams
