import pickle as pk
from pathlib import Path
from typing import Union

import lazy_import
import numpy as np
import pyvista as pv
import torch

from mascon_cube.constants import GROUND_TRUTH_DIR, MESH_DIR

tetgen = lazy_import.lazy_module("tetgen")


def mesh_to_gt(
    mesh_path: Union[Path, str],
    mask_generator: callable,
    mask_scalar: float,
    save_image: bool = False,
    save_uniform: bool = False,
) -> None:
    """Generate the mascon model ground truth from a mesh file.

    Args:
        # mesh_path (Union[Path, str]): Path to the mesh file or the name of the mesh file in the data/3dmeshes folder.
        mask_generator (callable): A function that takes the mascon points as input and returns a boolean mask.
        mask_scalar (float): The scalar to apply to the mascon masses inside the mask.
        save_image (bool, optional): Whether to save the image of the ground truth. Defaults to False.
        save_uniform (bool, optional): Whether to save also the uniform ground truth. Defaults to False.
    """
    mesh_path = get_mesh_path(mesh_path)
    mesh_points, mesh_triangles = get_mesh(mesh_path)
    # Here we define the surface
    tgen = tetgen.TetGen(mesh_points, mesh_triangles)
    # Here we run the algorithm to mesh the inside with thetrahedrons
    tgen.tetrahedralize()
    # get all cell centroids
    grid = tgen.grid

    grid = grid.compute_cell_sizes(volume=True, area=False, length=False)
    mascon_points_nu = np.array(grid.cell_centers().points)
    mascon_masses_nu = grid["Volume"]
    mascon_masses_nu = mascon_masses_nu / sum(mascon_masses_nu)
    if save_uniform:
        with open(GROUND_TRUTH_DIR / f"{mesh_path.stem}_uniform.pk", "wb") as file:
            pk.dump((mascon_points_nu, mascon_masses_nu), file)
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
        plotter.camera_position = [(1.08, -1.88, 1.25), (0, 0, 0), (0, 0, 1)]
        plotter.camera.zoom(0.9)
        plotter.screenshot(GROUND_TRUTH_DIR / f"{mesh_path.stem}.png", return_img=False)
        # plot slices - yz
        pv.global_theme.allow_empty_mesh = True
        plotter = pv.Plotter(off_screen=True)
        subslice = subgrid.slice("x", origin=[0, 0, 0])
        slice = grid.slice("x", origin=[0, 0, 0])
        plotter.add_mesh(subslice, "lightgrey", lighting=True, show_edges=True)
        plotter.add_mesh(slice, "r", "wireframe")
        plotter.camera_position = "yz"
        plotter.screenshot(
            GROUND_TRUTH_DIR / f"{mesh_path.stem}_yz.png", return_img=False
        )
        # plot slices - xz
        plotter = pv.Plotter(off_screen=True)
        subslice = subgrid.slice("y", origin=[0, 0, 0])
        slice = grid.slice("y", origin=[0, 0, 0])
        plotter.add_mesh(subslice, "lightgrey", lighting=True, show_edges=True)
        plotter.add_mesh(slice, "r", "wireframe")
        plotter.camera_position = "xz"
        plotter.screenshot(
            GROUND_TRUTH_DIR / f"{mesh_path.stem}_xz.png", return_img=False
        )
        # plot slices - xy
        plotter = pv.Plotter(off_screen=True)
        subslice = subgrid.slice("z", origin=[0, 0, 0])
        slice = grid.slice("z", origin=[0, 0, 0])
        plotter.add_mesh(subslice, "lightgrey", lighting=True, show_edges=True)
        plotter.add_mesh(slice, "r", "wireframe")
        plotter.camera_position = "xy"
        plotter.screenshot(
            GROUND_TRUTH_DIR / f"{mesh_path.stem}_xy.png", return_img=False
        )


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
    mesh_path = get_mesh_path(mesh_name)
    with open(mesh_path, "rb") as f:
        mesh_points, mesh_triangles = pk.load(f)
    mesh_points = np.array(mesh_points)
    mesh_triangles = np.array(mesh_triangles)
    return mesh_points, mesh_triangles


def get_mesh_path(mesh_name: Union[str, Path]) -> Path:
    """Return a valid path to a mesh if it exists

    Args:
        mesh_name (Union[str, Path]): The name of the mesh file or the path to the mesh file

    Returns:
        Path: the mesh Path
    """
    if isinstance(mesh_name, str):
        if Path(mesh_name).exists():
            mesh_path = Path(mesh_name)
        else:
            mesh_path = MESH_DIR / f"{mesh_name}.pk"
    else:
        mesh_path = mesh_name
    assert mesh_path.exists(), f"Mesh file {mesh_path} does not exist"
    return mesh_path


def points_in_tetrahedra_torch(
    x: torch.Tensor, nodes: torch.Tensor, elem: torch.Tensor
) -> torch.Tensor:
    """
    Find the tetrahedrons that contain each point in x

    Args:
        x (torch.Tensor): (L, 3) tensor, coordinates of all points
        nodes (torch.Tensor): (M, 3) tensor, coordinates of all nodes
        elem (torch.Tensor): (N, 4) tensor, indices of the nodes of each tetrahedron

    Returns:
        torch.Tensor: (L,) tensor, index of the tetrahedron that contains each point, or -1 if not found
    """
    assert (
        x.device == nodes.device == elem.device
    ), "All tensors must be on the same device"
    device = x.device

    # Move data to device
    x = torch.as_tensor(x, dtype=torch.float32, device=device)
    nodes = torch.as_tensor(nodes, dtype=torch.float32, device=device)
    elem = torch.as_tensor(elem, dtype=torch.long, device=device)

    L = x.shape[0]

    # Get tetrahedron vertices (M, 3)
    v0, v1, v2, v3 = (
        nodes[elem[:, 0]],
        nodes[elem[:, 1]],
        nodes[elem[:, 2]],
        nodes[elem[:, 3]],
    )

    # Compute transformation matrices (M, 3, 3)
    T = torch.stack([v1 - v0, v2 - v0, v3 - v0], dim=-1)

    # Precompute inverse transformation matrices (M, 3, 3)
    T_inv = torch.linalg.inv(T)

    # Initialize output with -1 (meaning "not found")
    result = torch.full((L,), -1, dtype=torch.long, device=device)

    # Compute barycentric coordinates for all points at once
    b = x[:, None, :] - v0  # (L, M, 3)
    lambdas = torch.einsum("mij, lmj -> lmi", T_inv, b)  # (L, M, 3)
    bary_coords = torch.cat(
        [lambdas, 1 - lambdas.sum(dim=2, keepdim=True)], dim=2
    )  # (L, M, 4)

    # Check which tetrahedron contains each point
    inside = (bary_coords >= 0) & (bary_coords <= 1)  # (L, M, 4)
    inside = inside.all(dim=2)  # (L, M)

    # Find first valid tetrahedron index for each point
    valid_tets = torch.argmax(inside.int(), dim=1)
    valid_mask = inside.any(dim=1)  # (L,)

    result[valid_mask] = valid_tets[valid_mask]  # Assign only valid results

    return result
