import pickle as pk
from pathlib import Path
from typing import Union

import numpy as np
import pyvista as pv
import tetgen

from mascon_cube.constants import GROUND_TRUTH_DIR, MESH_DIR
from mascon_cube.data.utils import get_mesh


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
