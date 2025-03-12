import pickle as pk
from pathlib import Path
from typing import Union

import numpy as np
import pyvista as pv
import tetgen
import torch

from mascon_cube.constants import GROUND_TRUTH_DIR, MESH_DIR


def mesh_to_gt(
    mesh_path: Union[Path, str],
    mask_generator: callable,
    mask_scalar: float,
    save_image: bool = False,
) -> None:
    """Generate the mascon model ground truth from a mesh file

    Args:
        # mesh_path (Union[Path, str]): Path to the mesh file or the name of the mesh file in the data/3dmeshes folder
        mask_generator (callable): A function that takes the mascon points as input and returns a boolean mask
        mask_scalar (float): The scalar to apply to the mascon masses inside the mask
        save_image (bool, optional): Whether to save the image of the ground truth. Defaults to False.
    """
    mesh_points, mesh_triangles = get_mesh(mesh_path)
    # Here we define the surface
    tgen = tetgen.TetGen(mesh_points, mesh_triangles)
    # Here we run the algorithm to mesh the inside with thetrahedrons
    tgen.tetrahedralize()
    # get all cell centroids
    grid = tgen.grid

    grid = grid.compute_cell_sizes(volume=True, area=False, length=False)
    mascon_masses = grid["Volume"]
    mascon_masses = mascon_masses / sum(mascon_masses)
    mascon_points_nu = np.array(grid.cell_centers().points)
    mascon_masses_nu = grid["Volume"]
    mascon_masses_nu = mascon_masses_nu / sum(mascon_masses_nu)
    mask = mask_generator(mascon_points_nu)
    mascon_masses_nu[mask] = mascon_masses_nu[mask] * mask_scalar
    mascon_masses_nu = mascon_masses_nu / sum(mascon_masses_nu)

    with open(GROUND_TRUTH_DIR / mesh_path.name, "wb") as file:
        pk.dump((mascon_points_nu, mascon_masses_nu), file)

    if save_image:
        pv.start_xvfb()
        pv.set_jupyter_backend("static")
        cell_ind = mask.nonzero()[0]
        subgrid = grid.extract_cells(cell_ind)
        plotter = pv.Plotter(off_screen=True)
        plotter.add_mesh(subgrid, "lightgrey", lighting=True, show_edges=True)
        plotter.add_mesh(grid, "r", "wireframe")
        plotter.screenshot(GROUND_TRUTH_DIR / f"{mesh_path.stem}.png", return_img=False)


def convert_mesh(
    mesh_path: Union[Path, str], output_name: str, brilluoin_radius: float = 1.0
) -> None:
    """Convert a mesh to a non-dimensionalized mesh with the center of mass at the origin and save it

    Args:
        mesh_path (Union[Path, str]): Path to the mesh file or the name of the mesh file in the data/3dmeshes folder
        output_name (str): The name of the output file
        brilluoin_radius (float, optional): The radius of the Brillouin zone. Defaults to 1.0.
    """
    if output_name[-3:] != ".pk":
        output_name = output_name + ".pk"
    mesh_points, mesh_triangles = get_mesh(mesh_path)
    # Convert to non-dimensional units
    length = max(mesh_points[:, 0]) - min(mesh_points[:, 0])
    mesh_points = mesh_points / length * 2 * brilluoin_radius
    # Put the model in a frame made of a principal axis of inertia
    tgen = tetgen.TetGen(mesh_points, mesh_triangles)
    tgen.tetrahedralize()
    grid = tgen.grid
    grid = grid.compute_cell_sizes(volume=True, area=False, length=False)
    mascon_points = np.array(grid.cell_centers().points)
    mascon_masses = grid["Volume"]
    mascon_masses = mascon_masses / sum(mascon_masses)
    offset = np.sum(mascon_points * mascon_masses.reshape((-1, 1)), axis=0) / np.sum(
        mascon_masses
    )
    mascon_points = mascon_points - offset
    mesh_points = mesh_points - offset
    with open(MESH_DIR / output_name, "wb") as file:
        pk.dump((mesh_points.tolist(), mesh_triangles), file)


def unpack_triangle_mesh(
    mesh_vertices: np.array, mesh_triangles: np.array, device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Unpacks the encoded triangles from vertices and faces

    Args:
        mesh_vertices (np.array): Nx3 vertices
        mesh_triangles (np.array): Vx3 indices of respectively three vertices

    Returns:
        tuple of torch.tensor: (first_vertices,second_vertices,third_vertices)
    """
    mesh_vertices = torch.tensor(mesh_vertices).float()
    mesh_triangles = torch.tensor(mesh_triangles)

    # Unpack vertices
    v0 = torch.zeros([len(mesh_triangles), 3], device=device)
    v1 = torch.zeros([len(mesh_triangles), 3], device=device)
    v2 = torch.zeros([len(mesh_triangles), 3], device=device)
    for idx, t in enumerate(mesh_triangles):
        v0[idx] = mesh_vertices[t[0]]
        v1[idx] = mesh_vertices[t[1]]
        v2[idx] = mesh_vertices[t[2]]

    return (v0, v1, v2)


def is_outside_torch(points, triangles):
    """Memory-efficient check if points are outside a 3D mesh."""
    device = triangles[0].device
    direction = torch.tensor([0.0, 0.0, 1.0], device=device)

    v0, v1, v2 = triangles

    batch_size = 50000  # Reduce further if OOM persists
    total_points = points.shape[0]

    counter = torch.zeros(total_points, device=device, dtype=torch.int32)

    for i in range(0, total_points, batch_size):
        end = min(i + batch_size, total_points)
        counter[i:end] = rays_triangle_intersect_torch(
            points[i:end], direction, v0, v1, v2
        )

    return (counter % 2) == 0


def rays_triangle_intersect_torch(ray_o, ray_d, v0, v1, v2):
    """Memory-efficient Möller–Trumbore intersection algorithm (vectorized)."""
    edge1 = v1 - v0  # Shape (M, 3)
    edge2 = v2 - v0  # Shape (M, 3)

    h = torch.cross(ray_d[None, :], edge2, dim=-1)  # Shape (M, 3)
    a = torch.sum(edge1 * h, dim=-1)  # Shape (M,)

    mask = torch.abs(a) > 1e-7
    f = torch.zeros_like(a)
    f[mask] = 1.0 / a[mask]  # Avoid division by zero

    s = ray_o[:, None, :] - v0  # Shape (N, M, 3)
    u = torch.sum(s * h, dim=-1) * f  # Shape (N, M)

    valid_u = (u >= 0.0) & (u <= 1.0)

    # Fix: Ensure both tensors have shape (N, M, 3) before cross product
    q = torch.cross(s, edge1[None, :, :], dim=-1)  # Shape (N, M, 3)

    v = torch.sum(q * ray_d[None, None, :], dim=-1) * f  # Shape (N, M)
    valid_v = (v >= 0.0) & ((u + v) <= 1.0)

    t = torch.sum(q * edge2[None, :, :], dim=-1) * f  # Shape (N, M)
    valid_t = t > 0.0

    return (valid_u & valid_v & valid_t & mask).sum(dim=-1)


def get_mesh(mesh_name: Union[str, Path]) -> tuple[np.ndarray, np.ndarray]:
    """Get the mesh points and triangles from a mesh file

    Args:
        mesh_name (Union[str, Path]): The name of the mesh file or the path to the mesh file

    Returns:
        tuple[np.ndarray, np.ndarray]: The mesh points and triangles
    """
    if isinstance(mesh_name, str):
        if Path(mesh_name).exists():
            mesh_path = Path(mesh_name)
        else:
            mesh_path = MESH_DIR / f"{mesh_name}.pk"
    else:
        mesh_path = mesh_name
    assert mesh_path.exists(), f"Mesh file {mesh_path} does not exist"

    with open(mesh_path, "rb") as f:
        mesh_points, mesh_triangles = pk.load(f)
    mesh_points = np.array(mesh_points)
    mesh_triangles = np.array(mesh_triangles)
    return mesh_points, mesh_triangles
