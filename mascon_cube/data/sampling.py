from pathlib import Path
from typing import Union

import torch

from mascon_cube.data.mesh import get_mesh, is_outside_torch, unpack_triangle_mesh


def get_target_point_sampler(
    n: int,
    asteroid_mesh: Union[str, Path],
    method: str,
    bounds: tuple[float, float],
    device: Union[str, torch.device],
    sample_step_size: int = 32,
) -> callable:
    """Get a function to sample n target points from. Points may differ each call depending on selected method.
    See specific implementations for details.

    Args:
        n (int): number of points to get each call.
        asteroid_mesh (Union[str, Path]): path to the asteroid mesh.
        method (str): method to use for sampling.
        bounds (tuple[float, float]): bounds for the sampling method.
        device (Union[str, torch.device]): device to use.
        sample_step_size (int, optional): how many points to sample in the inner loop of the sampler.
            It does not affect the final number of points, only the speed. Defaults to 32.

    Returns:
        callable: function to call to get n points.
    """
    # Load asteroid triangles
    mesh_vertices, mesh_triangles = get_mesh(asteroid_mesh)
    triangles = unpack_triangle_mesh(mesh_vertices, mesh_triangles, device)

    # Create a sampler to get some points
    if method == "cubical":

        def sampler():
            return _sample_cubical(sample_step_size, bounds, device)
    elif method == "spherical":

        def sampler():
            return _sample_spherical(sample_step_size, bounds, device)

    # Create a new sampler inside the sampler so to speak
    return lambda: _get_n_points_outside_asteroid(
        n, sampler, triangles, sample_step_size, device
    )


def _sample_spherical(
    n: int, radius_bounds: tuple[float, float], device: Union[str, torch.device]
) -> torch.Tensor:
    """Generates n uniform random samples from a sphere with passed radius.

    Args:
        n (int): number of points to create.
        radius_bounds (tuple[float, float]): bounds for the radius of the sphere.
        device (Union[str, torch.device]): device to use.

    Returns:
        torch.Tensor: sampled points
    """
    theta = 2.0 * torch.pi * torch.rand(n, 1, device=device)

    # The acos here allows us to sample uniformly on the sphere
    phi = torch.acos(1.0 - 2.0 * torch.rand(n, 1, device=device))

    minimal_radius_scale = radius_bounds[0] / radius_bounds[1]
    # Create uniform between
    uni = minimal_radius_scale + (1.0 - minimal_radius_scale) * torch.rand(
        n, 1, device=device
    )
    r = radius_bounds[1] * torch.pow(uni, 1 / 3)

    x = r * torch.sin(phi) * torch.cos(theta)
    y = r * torch.sin(phi) * torch.sin(theta)
    z = r * torch.cos(phi)

    points = (
        torch.stack((x.flatten(), y.flatten(), z.flatten())).transpose(0, 1).to(device)
    )
    return points


def _sample_cubical(
    n: int, scale_bounds: tuple[float, float], device: Union[str, torch.device]
):
    """Generate n uniform random samples from a cube with passed scale.

    Args:
        n (int): number of points to create.
        scale_bounds (tuple[float, float]): bounds for the scale of the cube.
        device (Union[str, torch.device]): device to use.

    Returns:
        torch.Tensor: sampled points
    """
    # Approximation of percentage points in Unitsphere
    approx = (scale_bounds[0] / scale_bounds[1]) ** 3

    # Sample twice the expected number of points necessary to achieve n
    approx_necessary_samples = int(2 * n * (1.0 / (1.0 - approx)))
    points = (
        torch.rand(approx_necessary_samples, 3, device=device) * 2 - 1
    ) * scale_bounds[1]

    # Discard points inside unitcube
    domain = scale_bounds[0] * torch.tensor([[-1, 1], [-1, 1], [-1, 1]], device=device)
    points = _limit_to_domain(points, domain)

    # Take first n points (n.B. that in super unlikely event of
    # less than n points this will not crash. I tried. :))
    return points[:n]


def _limit_to_domain(points: torch.Tensor, domain: torch.Tensor) -> torch.Tensor:
    """Throws away all passed points that were inside the passed domain. Domain has to be cuboid.

    Args:
        points (torch.Tensor): points to check.
        domain (torch.Tensor): cuboid domain to check against.

    Returns:
        torch.Tensor: points that were outside the domain.
    """
    a = torch.logical_or((points[:, 0] < domain[0][0]), (points[:, 0] > domain[0][1]))
    b = torch.logical_or((points[:, 1] < domain[1][0]), (points[:, 1] > domain[1][1]))
    c = torch.logical_or((points[:, 2] < domain[2][0]), (points[:, 2] > domain[2][1]))
    d = torch.logical_or(torch.logical_or(a, b), c)
    return points[d]


def _get_n_points_outside_asteroid(
    n: int,
    sampler: callable,
    triangles: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    sample_step_size: int,
    device: Union[str, torch.device],
) -> torch.Tensor:
    """Sample points until n outside asteroid reached with given sampler and triangles

    Args:
        n (int): number of points to get.
        sampler (callable): sampler to use.
        triangles (tuple[torch.Tensor, torch.Tensor, torch.Tensor]): triangles of the asteroid.
        sample_step_size (int): how many points to sample in the inner loop of the sampler.
        device (Union[str, torch.device]): device to use.

    Returns:
        torch.Tensor: n points outside the asteroid.
    """
    # We allocate a few more just to avoid having to check, will discard in return
    points = torch.zeros([n + sample_step_size, 3], device=device)
    found_points = 0

    # Sample points till we sufficient amount
    while found_points < n:
        # Get some points
        candidates = sampler()
        candidates_outside = candidates[is_outside_torch(candidates, triangles)]

        # Add those that were outside to our collection
        new_points = len(candidates_outside)
        if new_points > 0:
            points[found_points : found_points + new_points, :] = candidates_outside
            found_points += new_points

    return points[:n]
