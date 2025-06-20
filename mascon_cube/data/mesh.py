import pickle as pk
import shutil
from pathlib import Path
from typing import Optional, Union

import lazy_import
import numpy as np
import pyvista as pv
import torch

from mascon_cube.constants import GROUND_TRUTH_DIR, MESH_DIR

tetgen = lazy_import.lazy_module("tetgen")


def mesh_to_gt(
    mesh_path: Union[Path, str],
    mask_generators: Union[list[callable], callable],
    mask_scalars: Union[list[float], float],
    save_image: bool = True,
    save_uniform: bool = True,
    save_mesh: bool = True,
    output_name: Optional[str] = None,
) -> None:
    """Generate the mascon model ground truth from a mesh file.

    Args:
        mesh_path (Union[Path, str]): Path to the mesh file or the name of the mesh file in the data/3dmeshes folder.
        mask_generators (Union[list[callable], callable]): One or more functions that take a numpy array of points and
            return a boolean mask. Each mask will create a different density region. They should not overlap.
        mask_scalars (Union[list[float], float]): The scalar to multiply the mass of the points in the mask.
            If a list is provided, it should have the same length as the number of masks.
        save_image (bool, optional): Whether to save the image of the ground truth. Defaults to True.
        save_uniform (bool, optional): Whether to save also the uniform ground truth. Defaults to True.
        save_mesh (bool, optional): Whether to save the mesh. Defaults to True.
        output_name (str, optional): The name of the output directory.
            Defaults to None, in which case the mesh name is used.
    """
    if not isinstance(mask_generators, (list, tuple)):
        mask_generators = [mask_generators]
    if not isinstance(mask_scalars, (list, tuple)):
        mask_scalars = [mask_scalars]
    assert len(mask_generators) == len(
        mask_scalars
    ), "mask_generator and mask_scalar must have the same length"
    mesh_path = get_mesh_path(mesh_path)
    if output_name is None:
        output_name = mesh_path.stem
    output_dir = GROUND_TRUTH_DIR / output_name
    output_dir.mkdir(exist_ok=True)
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
        with open(output_dir / "mascon_model_uniform.pk", "wb") as file:
            pk.dump((mascon_points_nu, mascon_masses_nu), file)
    for mask_generator, mask_scalar in zip(mask_generators, mask_scalars):
        mask = mask_generator(mascon_points_nu)
        mascon_masses_nu[mask] = mascon_masses_nu[mask] * mask_scalar
    mascon_masses_nu = mascon_masses_nu / sum(mascon_masses_nu)
    mascon_densities = mascon_masses_nu / grid["Volume"]
    grid.cell_data["mass"] = mascon_densities

    with open(output_dir / "mascon_model.pk", "wb") as file:
        pk.dump((mascon_points_nu, mascon_masses_nu), file)

    if save_image:
        __plot_model(grid, output_dir)

    if save_mesh:
        shutil.copy(mesh_path, output_dir / "mesh.pk")


def mesh_to_gt_function(
    mesh_path: Union[Path, str],
    function: callable,
    save_image: bool = True,
    save_uniform: bool = True,
    save_mesh: bool = True,
    output_name: Optional[str] = None,
) -> None:
    mesh_path = get_mesh_path(mesh_path)
    if output_name is None:
        output_name = mesh_path.stem
    output_dir = GROUND_TRUTH_DIR / output_name
    output_dir.mkdir(exist_ok=True)
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
        with open(output_dir / "mascon_model_uniform.pk", "wb") as file:
            pk.dump((mascon_points_nu, mascon_masses_nu), file)
    mascon_masses_nu *= np.apply_along_axis(function, 1, mascon_points_nu)
    mascon_masses_nu = mascon_masses_nu / sum(mascon_masses_nu)
    mascon_densities = mascon_masses_nu / grid["Volume"]
    grid.cell_data["mass"] = mascon_densities

    with open(output_dir / "mascon_model.pk", "wb") as file:
        pk.dump((mascon_points_nu, mascon_masses_nu), file)

    if save_image:
        __plot_model(grid, output_dir)

    if save_mesh:
        shutil.copy(mesh_path, output_dir / "mesh.pk")


def mesh_to_gt_random_spots(
    mesh_path: Union[Path, str],
    frequency: int = 0.1,
    seed: int = 42,
    save_image: bool = True,
    save_uniform: bool = True,
    save_mesh: bool = True,
    output_name: Optional[str] = None,
) -> None:
    """Generate the mascon model ground truth from a mesh file.

    Args:
        mesh_path (Union[Path, str]): Path to the mesh file or the name of the mesh file in the data/3dmeshes folder.
        frequency (int, optional): The relative frequency of heterogeneities. Defaults to 0.1.
        seed (int, optional): The seed for the random number generator. Defaults to 42.
        save_image (bool, optional): Whether to save the image of the ground truth. Defaults to True.
        save_uniform (bool, optional): Whether to save also the uniform ground truth. Defaults to True.
        save_mesh (bool, optional): Whether to save the mesh. Defaults to True.
        output_name (str, optional): The name of the output directory.
            Defaults to None, in which case the mesh name is used.
    """
    mesh_path = get_mesh_path(mesh_path)
    if output_name is None:
        output_name = mesh_path.stem + "_spots"
    output_dir = GROUND_TRUTH_DIR / f"{output_name}"
    output_dir.mkdir(exist_ok=True)
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
        with open(output_dir / "mascon_model_uniform.pk", "wb") as file:
            pk.dump((mascon_points_nu, mascon_masses_nu), file)
    n_mascons = len(mascon_masses_nu)
    mask = np.zeros(n_mascons, dtype=bool)
    np.random.seed(seed)
    mask[np.random.choice(n_mascons, int(n_mascons * frequency), replace=False)] = True
    values = np.random.uniform(0, 2, mask.shape)
    mascon_masses_nu[mask] = mascon_masses_nu[mask] * values[mask]
    mascon_masses_nu = mascon_masses_nu / sum(mascon_masses_nu)
    mascon_densities = mascon_masses_nu / grid["Volume"]
    grid.cell_data["mass"] = mascon_densities

    with open(output_dir / "mascon_model.pk", "wb") as file:
        pk.dump((mascon_points_nu, mascon_masses_nu), file)

    if save_image:
        __plot_model(grid, output_dir)

    if save_mesh:
        shutil.copy(mesh_path, output_dir / "mesh.pk")


def __plot_model(grid: pv.UnstructuredGrid, output_dir: Path) -> None:
    pv.start_xvfb()
    pv.set_jupyter_backend("static")

    vmin, vmax = 0, np.max(grid["mass"])
    scalar_bar_args = {
        "title": "Density",
        "vertical": False,
        "position_x": 0.3,
        "position_y": 0.02,
        "height": 0.1,
        "width": 0.65,
        "title_font_size": 32,
        "label_font_size": 28,
        "n_labels": 4,
        "unconstrained_font_size": True,
    }

    plotter = pv.Plotter(shape=(2, 2), window_size=(1200, 1200), off_screen=True)

    # 3D plot (top-left)
    plotter.subplot(0, 0)
    plotter.add_mesh(
        grid,
        scalars="mass",
        show_edges=True,
        clim=[vmin, vmax],
        scalar_bar_args=scalar_bar_args,
    )
    plotter.view_vector((1, 1, 1))  # 45° elev, 45° azim is along (1,1,1) vector
    plotter.add_axes(label_size=(0.2, 0.2), line_width=3)
    plotter.add_text("3D View", position="upper_edge", font_size=18)

    # Frontal slice helpers
    def add_frontal_slice(row, col, orientation, label, position):
        plotter.subplot(row, col)
        slice_ = grid.slice(orientation)
        plotter.add_mesh(
            slice_,
            scalars="mass",
            show_edges=True,
            clim=[vmin, vmax],
            show_scalar_bar=False,
        )
        # plotter.show_grid()
        plotter.add_axes(label_size=(0.2, 0.2), line_width=3)
        plotter.camera_position = position
        plotter.add_text(label, position="upper_edge", font_size=18)

    # XY slice (z=0) - look along +z
    add_frontal_slice(0, 1, "z", "XY Slice", "xy")
    # YZ slice (x=0) - look along +x
    add_frontal_slice(1, 1, "x", "YZ Slice", "yz")
    # XZ slice (y=0) - look along +y
    add_frontal_slice(1, 0, "y", "XZ Slice", "xz")

    # plotter.link_views()  # Optional: link views for zoom/pan
    plotter.screenshot(output_dir / "combined_plot.png", return_img=False)


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
        elif (MESH_DIR / f"{mesh_name}.pk").exists():
            mesh_path = MESH_DIR / f"{mesh_name}.pk"
        else:
            mesh_path = GROUND_TRUTH_DIR / mesh_name / "mesh.pk"
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
