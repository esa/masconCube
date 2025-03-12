import numpy as np
from matplotlib import pyplot as plt

from mascon_cube.models import MasconCube


def plot_mascon_cube(
    mascon_cube: MasconCube,
    s: int = 1.8,
    marker: str = "s",
    cmap: str = "viridis",
    threshold: float = 1e-16,
):
    """
    plot the mascon model in 3D and in 2D sections (XY, XZ, YZ) where the color represents the mass
    Args:
        mascon_points (torch.tensor): mascon points
        mascon_model (torch.nn): mascon model
    """
    fig = plt.figure(figsize=(10, 10), dpi=100, facecolor="white")
    ax = fig.add_subplot(221, projection="3d", aspect="equal")
    ax2 = fig.add_subplot(222, aspect="equal")
    ax3 = fig.add_subplot(223, aspect="equal")
    ax4 = fig.add_subplot(224, aspect="equal")
    x = mascon_cube.coords[:, 0].cpu().numpy()
    y = mascon_cube.coords[:, 1].cpu().numpy()
    z = mascon_cube.coords[:, 2].cpu().numpy()
    mass = mascon_cube.weights.detach().cpu().numpy() + mascon_cube.uniform_base_mass
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    sc = ax.scatter(x, y, z, c=mass, cmap=cmap, s=s)
    # select the points in the XY plane (z=0)
    # compute closest point to z=0
    closest = np.abs(z).min()
    mask = np.abs(z) - closest < threshold
    ax2.set_xlim([-1, 1])
    ax2.set_ylim([-1, 1])
    sc2 = ax2.scatter(x[mask], y[mask], c=mass[mask], cmap=cmap, marker=marker, s=s)
    ax2.set_title("XY plane")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    # select the points in the XZ plane (y=0)
    closest = np.abs(y).min()
    mask = np.abs(y) - closest < threshold
    ax3.set_xlim([-1, 1])
    ax3.set_ylim([-1, 1])
    sc3 = ax3.scatter(x[mask], z[mask], c=mass[mask], cmap=cmap, marker=marker, s=s)
    ax3.set_title("XZ plane")
    ax3.set_xlabel("X")
    ax3.set_ylabel("Z")
    # select the points in the YZ plane (x=0)
    closest = np.abs(x).min()
    mask = np.abs(x) - closest < threshold
    ax4.set_xlim([-1, 1])
    ax4.set_ylim([-1, 1])
    sc4 = ax4.scatter(y[mask], z[mask], c=mass[mask], cmap=cmap, marker=marker, s=s)
    ax4.set_title("YZ plane")
    ax4.set_xlabel("Y")
    ax4.set_ylabel("Z")
    # colorbar
    fig.colorbar(sc, ax=ax, orientation="vertical")
    fig.colorbar(sc2, ax=ax2, orientation="vertical")
    fig.colorbar(sc3, ax=ax3, orientation="vertical")
    fig.colorbar(sc4, ax=ax4, orientation="vertical")
    return fig
