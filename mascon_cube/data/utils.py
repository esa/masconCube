import pickle as pk
from pathlib import Path
from typing import Union

import numpy as np

from mascon_cube.constants import MESH_DIR


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
