from typing import Optional

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

from mascon_cube.models import MasconCube


def plot_mascon_cube(
    mascon_cube: MasconCube,
    s: int = 1.8,
    marker: str = "s",
    cmap: str = "viridis",
    threshold: float = 1e-16,
    range: Optional[tuple[float, float]] = None,
) -> plt.Figure:
    """
    plot the mascon model in 3D and in 2D sections (XY, XZ, YZ) where the color represents the mass

    Args:
        mascon_cube (MasconCube): The mascon cube to plot
        s (int, optional): The size of the points. Defaults to 1.8.
        marker (str, optional): The marker of the points. Defaults to "s".
        cmap (str, optional): The colormap to use. Defaults to "viridis".
        threshold (float, optional): The threshold to select the points in the planes. Defaults to 1e-16.
        range (Optional[tuple[float, float]], optional): The range of the colormap. Defaults to None.

    Returns:
        plt.Figure: The figure with the 4 subplots (3D, XY, XZ, YZ).
    """
    fig = plt.figure(figsize=(10, 10), dpi=100, facecolor="white")
    ax = fig.add_subplot(221, projection="3d", aspect="equal")
    ax2 = fig.add_subplot(222, aspect="equal")
    ax3 = fig.add_subplot(223, aspect="equal")
    ax4 = fig.add_subplot(224, aspect="equal")
    x = mascon_cube.coords[:, 0].cpu().numpy()
    y = mascon_cube.coords[:, 1].cpu().numpy()
    z = mascon_cube.coords[:, 2].cpu().numpy()
    mass = mascon_cube.masses.detach().cpu().numpy()
    if range is None:
        range = (mass.min(), mass.max())
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    sc = ax.scatter(x, y, z, c=mass, cmap=cmap, s=s, vmin=range[0], vmax=range[1])
    # select the points in the XY plane (z=0)
    # compute closest point to z=0
    closest = np.abs(z).min()
    mask = np.abs(z) - closest < threshold
    ax2.set_xlim([-1, 1])
    ax2.set_ylim([-1, 1])
    sc2 = ax2.scatter(
        x[mask],
        y[mask],
        c=mass[mask],
        cmap=cmap,
        marker=marker,
        s=s,
        vmin=range[0],
        vmax=range[1],
    )
    ax2.set_title("XY plane")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    # select the points in the XZ plane (y=0)
    closest = np.abs(y).min()
    mask = np.abs(y) - closest < threshold
    ax3.set_xlim([-1, 1])
    ax3.set_ylim([-1, 1])
    sc3 = ax3.scatter(
        x[mask],
        z[mask],
        c=mass[mask],
        cmap=cmap,
        marker=marker,
        s=s,
        vmin=range[0],
        vmax=range[1],
    )
    ax3.set_title("XZ plane")
    ax3.set_xlabel("X")
    ax3.set_ylabel("Z")
    # select the points in the YZ plane (x=0)
    closest = np.abs(x).min()
    mask = np.abs(x) - closest < threshold
    ax4.set_xlim([-1, 1])
    ax4.set_ylim([-1, 1])
    sc4 = ax4.scatter(
        y[mask],
        z[mask],
        c=mass[mask],
        cmap=cmap,
        marker=marker,
        s=s,
        vmin=range[0],
        vmax=range[1],
    )
    ax4.set_title("YZ plane")
    ax4.set_xlabel("Y")
    ax4.set_ylabel("Z")
    # colorbar
    fig.colorbar(sc, ax=ax, orientation="vertical")
    fig.colorbar(sc2, ax=ax2, orientation="vertical")
    fig.colorbar(sc3, ax=ax3, orientation="vertical")
    fig.colorbar(sc4, ax=ax4, orientation="vertical")
    return fig


def stokes_heatmap(
    pred: np.ndarray,
    gt: np.ndarray,
    title: str = "",
    vmin: float = 1e-10,
    vmax: float = 1e-4,
    xlabel: str = "n",
    ylabel: str = "m",
) -> plt.Figure:
    """Create a heatmap of the difference between the prediction and the ground truth.

    Args:
        pred (np.ndarray): NxM array of predictions
        gt (np.ndarray): NxM array of ground truth
        title (str, optional): plot title. Defaults to "".
        vmin (float, optional): minimum value for the colormap. Defaults to 1e-10.
        vmax (float, optional): maximum value for the colormap. Defaults to 1e-4.
        xlabel (str, optional): label for the x axis. Defaults to "n".
        ylabel (str, optional): label for the y axis. Defaults to "m".

    Returns:
        plt.Figure: The figure with the heatmap.
    """
    fig, ax = plt.subplots()
    diff = (pred - gt).abs()
    sns.heatmap(
        diff, annot=True, norm=LogNorm(), annot_kws={"fontsize": 8}, fmt=".1e", ax=ax
    )
    ax.collections[0].set_clim(vmin, vmax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return fig
