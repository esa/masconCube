import warnings
from typing import Optional, Union

import numpy as np
import torch

from .models import GeodesyNet


def ACC_trap(
    target_points: torch.Tensor,
    model: GeodesyNet,
    n: int = 10000,
    noise: float = 0.0,
    sample_points: Optional[torch.Tensor] = None,
    h: Optional[float] = None,
    domain: Optional[torch.Tensor] = None,
):
    """From darioizzo/geodesyNets.
    Uses a 3D trapezoid rule for the evaluation of the integral in the potential from the modeled density

    Args:
        target_points (2-D array-like): a (N,3) array-like object containing the points.
        model (callable (a,b)->1): neural model for the asteroid.
        encoding: the encoding for the neural inputs.
        N (int): number of points. If a grid is passed should match that
        verbose (bool, optional): Print intermediate results. Defaults to False.
        noise (float): random noise added to point positions.
        sample_points (torch tensor): grid to sample the integral on
        h (float): grid spacing, only has to be passed if grid is passed.
        domain (torch.tensor): integration domain [3,2] , pass None for [-1,1]^3

    Returns:
        Tensor: Computed potentials per point
    """
    device = target_points.device

    if domain is None:  # None might be passed as well
        domain = [[-1, 1], [-1, 1], [-1, 1]]

    # init result vector
    retval = torch.empty(len(target_points), 3, device=device)

    # Determine grid to compute on
    if sample_points is None:
        sample_points, h, n = compute_integration_grid(n, noise, domain, device)
    else:
        if h is None:
            raise (ValueError("h has to be passed if sample points are passed."))

    # Evaluate Rho on the grid
    rho = _compute_model_output(model, sample_points)

    for i, target_point in enumerate(target_points):
        # Evaluate all points
        distance = torch.sub(target_point, sample_points)
        f_values = (
            rho / torch.pow(torch.norm(distance, dim=1), 3).view(-1, 1) * distance
        )

        evaluations = f_values.reshape([n, n, n, 3])  # map to z,y,x

        # area = h / 2 * (f0 + f2)
        int_x = h[0] / 2 * (evaluations[:, :, 0:-1, :] + evaluations[:, :, 1:, :])
        int_x = torch.sum(int_x, dim=2)
        int_y = h[1] / 2 * (int_x[:, 0:-1, :] + int_x[:, 1:, :])
        int_y = torch.sum(int_y, dim=1)
        int_z = h[2] / 2 * (int_y[0:-1, :] + int_y[1:, :])
        int_z = torch.sum(int_z, dim=0)

        retval[i] = int_z
    return -retval


def compute_integration_grid(
    n: int,
    noise: float = 0.0,
    domain: list[list[int]] = [[-1, 1], [-1, 1], [-1, 1]],
    device: Union[str, torch.device] = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """from darioizzo/geodesyNets
    Creates a grid which can be used for the trapezoid integration

    Args:
        n (int): Number of points to approximately  generate
        noise (float, optional): Amount of noise to add to points (can be used to sample nearby points). Defaults to 0.
        domain (torch.tensor): integration domain [3,2]
        device (Union[str, torch.device], optional): defaults to cpu.
    Returns:
        torch.Tensor, torch.Tensor, int: sample points, grid h, nr of points
    """
    n = int(np.round(np.cbrt(n)))  # approximate subdivisions

    h = torch.zeros([3], device=device)
    # Create grid and assemble evaluation points

    grid_1d_x = torch.linspace(domain[0][0], domain[0][1], n, device=device)
    grid_1d_y = torch.linspace(domain[1][0], domain[1][1], n, device=device)
    grid_1d_z = torch.linspace(domain[2][0], domain[2][1], n, device=device)

    h[0] = grid_1d_x[1] - grid_1d_x[0]
    h[1] = grid_1d_y[1] - grid_1d_y[0]
    h[2] = grid_1d_z[1] - grid_1d_z[0]

    x, y, z = torch.meshgrid(grid_1d_x, grid_1d_y, grid_1d_z)
    eval_points = (
        torch.stack((x.flatten(), y.flatten(), z.flatten())).transpose(0, 1).to(device)
    )

    # We add some noise to the evaluated grid points to ensure the networks learns all
    if noise > 0:
        eval_points += torch.rand(n**3, 3, device=device) * noise

    return eval_points, h, n


def _compute_model_output(
    model: GeodesyNet, sample_points: torch.Tensor
) -> torch.Tensor:
    """From darioizzo/geodesyNets
    Computes model output on the passed points using the passed encoding

    Args:
        model (GeodesyNet): neural network to eval
        sample_points (torch.Tensor): points to sample at

    Returns:
        torch tensor: computed values
    """
    # 2 - check if any values were NaN
    if torch.any(sample_points != sample_points):
        warnings.warn("The network generated NaN outputs!")
        sample_points[sample_points != sample_points] = 0.0  # set Nans to 0

    # 3 - compute the predicted density at the points
    return model(sample_points)
