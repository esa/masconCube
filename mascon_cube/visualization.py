from typing import Optional

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from torch import Tensor

from mascon_cube.constants import GROUND_TRUTH_DIR
from mascon_cube.models import MasconCube


def plot_asteroid(asteroid: str) -> plt.Figure:
    """plot the thetraedrons model of the asteroid in 3D and in 2D sections (XY, XZ, YZ) where the color represents the
        density.

    Args:
        asteroid (str): name of the asteroid. Must be an existing folder in `mascon_cube.constants.GROUND_TRUTH_DIR`

    Returns:
        plt.Figure: The figure with the 4 subplots (3D, XY, XZ, YZ).
    """
    img = plt.imread(GROUND_TRUTH_DIR / asteroid / "plot.png")
    img_xy = plt.imread(GROUND_TRUTH_DIR / asteroid / "plot_xy.png")
    img_xz = plt.imread(GROUND_TRUTH_DIR / asteroid / "plot_xz.png")
    img_yz = plt.imread(GROUND_TRUTH_DIR / asteroid / "plot_yz.png")

    # Concatenate horizontally
    combined = np.vstack((np.hstack((img, img_xy)), np.hstack((img_xz, img_yz))))

    # Show or save the result
    fig = plt.figure(figsize=(10, 10))
    plt.imshow(combined)
    plt.title(asteroid)
    plt.axis("off")
    return fig


def plot_mascon_cube(
    mascon_cube: MasconCube,
    s: int = 1.8,
    marker: str = "s",
    cmap: str = "viridis",
    threshold: float = 1e-16,
    range: Optional[tuple[float, float]] = None,
    return_range: bool = False,
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
        return_range (bool, optional): Whether to return the range of the mass. Defaults to False.

    Returns:
        plt.Figure: The figure with the 4 subplots (3D, XY, XZ, YZ).
    """
    fig = plt.figure(figsize=(6, 6), dpi=100, facecolor="white")
    ax = fig.add_subplot(221, projection="3d", aspect="equal")
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.view_init(elev=45.0, azim=45.0)
    ax.tick_params(labelsize=6)
    ax.set_xlabel("X", fontsize=8)
    ax.set_ylabel("Y", fontsize=8)
    ax.set_zlabel("Z", fontsize=8)
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.set_zticks([-1, -0.5, 0, 0.5, 1])
    ax2 = fig.add_subplot(222, aspect="equal")
    ax3 = fig.add_subplot(223, aspect="equal")
    ax4 = fig.add_subplot(224, aspect="equal")
    x = mascon_cube.coords[:, 0].cpu().numpy()
    y = mascon_cube.coords[:, 1].cpu().numpy()
    z = mascon_cube.coords[:, 2].cpu().numpy()
    mass = mascon_cube.masses.detach().cpu().numpy()
    if range is None:
        # take 99th percentile of the mass
        range = (0, np.percentile(mass, 99))
    ax.scatter(x, y, z, c=mass, cmap=cmap, s=s, vmin=range[0], vmax=range[1])
    # X Rectangle
    ax.plot_wireframe(
        np.asarray([[0, 0], [0, 0]]),
        np.asarray([[1, 1], [-1, -1]]),
        np.asarray([[-1, 1], [-1, 1]]),
        color="red",
        linestyle="--",
        alpha=0.5,
    )
    # Y Rectangle
    ax.plot_wireframe(
        np.asarray([[1, 1], [-1, -1]]),
        np.asarray([[0, 0], [0, 0]]),
        np.asarray([[-1, 1], [-1, 1]]),
        color="blue",
        linestyle="--",
        alpha=0.5,
    )
    # Z Rectangle
    ax.plot_wireframe(
        np.asarray([[-1, 1], [-1, 1]]),
        np.asarray([[1, 1], [-1, -1]]),
        np.asarray([[0, 0], [0, 0]]),
        color="green",
        linestyle="--",
        alpha=0.5,
    )
    ax.set_title("3D View", fontsize=7)
    # select the points in the XY plane (z=0)
    # compute closest point to z=0
    closest = np.abs(z).min()
    mask = np.abs(z) - closest < threshold
    ax2.set_xlim([-1, 1])
    ax2.set_ylim([-1, 1])
    ax2.tick_params(labelsize=6, color="green")
    ax2.spines["bottom"].set_color("green")
    ax2.spines["top"].set_color("green")
    ax2.spines["right"].set_color("green")
    ax2.spines["left"].set_color("green")
    ax2.set_aspect("equal", "box")
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
    ax2.set_title("X-Y cross section (green slice)", fontsize=8)
    ax2.set_xlabel("X", fontsize=8)
    ax2.set_ylabel("Y", fontsize=8)
    # select the points in the XZ plane (y=0)
    closest = np.abs(y).min()
    mask = np.abs(y) - closest < threshold
    ax3.set_xlim([-1, 1])
    ax3.set_ylim([-1, 1])
    ax3.tick_params(labelsize=6, color="blue")
    ax3.spines["bottom"].set_color("blue")
    ax3.spines["top"].set_color("blue")
    ax3.spines["right"].set_color("blue")
    ax3.spines["left"].set_color("blue")
    ax3.set_aspect("equal", "box")
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
    ax3.set_title("X-Z cross section (blue slice)", fontsize=8)
    ax3.set_xlabel("X", fontsize=8)
    ax3.set_ylabel("Z", fontsize=8)
    # select the points in the YZ plane (x=0)
    closest = np.abs(x).min()
    mask = np.abs(x) - closest < threshold
    ax4.set_xlim([-1, 1])
    ax4.set_ylim([-1, 1])
    ax4.tick_params(labelsize=6, color="red")
    ax4.spines["bottom"].set_color("red")
    ax4.spines["top"].set_color("red")
    ax4.spines["right"].set_color("red")
    ax4.spines["left"].set_color("red")
    ax4.set_aspect("equal", "box")
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
    ax4.set_title("Y-Z cross section (red slice)", fontsize=8)
    ax2.set_xlabel("Y", fontsize=8)
    ax2.set_ylabel("Z", fontsize=8)
    # colorbar
    # fig.colorbar(sc, ax=ax, orientation="vertical")
    cb = fig.colorbar(sc2, ax=ax2, orientation="vertical")
    cb.ax.tick_params(labelsize=6)
    cb.set_label("Mass", rotation=270, labelpad=15, fontsize=8)
    cb.ax.yaxis.get_offset_text().set_fontsize(6)
    cb = fig.colorbar(sc3, ax=ax3, orientation="vertical")
    cb.ax.tick_params(labelsize=6)
    cb.set_label("Mass", rotation=270, labelpad=15, fontsize=8)
    cb.ax.yaxis.get_offset_text().set_fontsize(6)
    cb = fig.colorbar(sc4, ax=ax4, orientation="vertical")
    cb.ax.tick_params(labelsize=6)
    cb.set_label("Mass", rotation=270, labelpad=15, fontsize=8)
    cb.ax.yaxis.get_offset_text().set_fontsize(6)
    plt.tight_layout()
    if return_range:
        return fig, range
    else:
        return fig


def stokes_heatmap(
    pred: np.ndarray,
    gt: np.ndarray,
    title: str = "",
    vmin: float = 1e-4,
    vmax: float = 5e-1,
    xlabel: str = r"$l$",
    ylabel: str = r"$m$",
    relative: bool = True,
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
    if isinstance(pred, Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(gt, Tensor):
        gt = gt.detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=(7, 5))
    diff = np.abs(pred - gt)
    if relative:
        diff = diff / (np.abs(gt) + 1e-16)
    sns.heatmap(
        diff,
        annot=True,
        norm=LogNorm(),
        annot_kws={"fontsize": 7},
        fmt=".1e",
        ax=ax,
        cmap="YlOrRd",
        square=True,
        cbar_kws={"pad": 0.1},
    )
    ax.collections[0].set_clim(vmin, vmax)
    ax.set_title(title, pad=15, loc="center")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax_top = ax.secondary_xaxis("top")
    ax_top.set_xlabel(xlabel)
    ax_top.set_xticks(np.arange(1.5, pred.shape[1], 1), np.arange(1, pred.shape[1], 1))
    ax_right = ax.secondary_yaxis("right")
    ax_right.set_ylabel(ylabel)
    ax_right.set_yticks(
        np.arange(0.5, pred.shape[0] - 1, 1), np.arange(0, pred.shape[0] - 1, 1)
    )
    ax_right.spines["right"].set_visible(False)
    ax_top.spines["top"].set_visible(False)
    for i in range(min(diff.shape)):
        # Top border
        if i > 0:
            ax.plot([i, i + 1], [i, i], color="white", linewidth=3)
        # Right border
        if i < diff.shape[0] - 1:
            ax.plot([i + 1, i + 1], [i, i + 1], color="white", linewidth=3)
    plt.text(
        1.01,
        1.01,
        r"$\tilde{S}_{m,l}$",
        fontsize=14,
        horizontalalignment="left",
        verticalalignment="bottom",
        transform=ax.transAxes,
    )
    plt.text(
        -0.01,
        -0.01,
        r"$\tilde{C}_{m,l}$",
        fontsize=14,
        horizontalalignment="right",
        verticalalignment="top",
        transform=ax.transAxes,
    )
    return fig


def stokes_boxplot(
    errors: list[np.ndarray], labels: list[str], title: str = ""
) -> plt.Figure:
    """Create a boxplot of the errors.

    Args:
        errors (list[np.ndarray]): list of errors

    Returns:
        plt.Figure: The figure with the boxplot.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.boxplot(
        data=errors,
        ax=ax,
        palette="Set2",
        showfliers=False,
        orient="h",
    )
    ax.xaxis.grid(True, linestyle="--", alpha=0.5)
    # ax.set_xscale("log")
    ax.set_title(title, pad=15, loc="center")

    ax.set_yticklabels(labels, fontsize=8, rotation=90, va="center")
    ax.set_xlabel("Relative error", fontsize=8)
    ax.tick_params(labelsize=8)
    return fig


def _plot_ellipsoid(ax, a, b, c, **kwargs):
    """Plot a wireframe ellipsoid."""
    u, v = np.mgrid[0 : 2 * np.pi : 40j, 0 : np.pi : 40j]
    X = a * np.cos(u) * np.sin(v)
    Y = b * np.sin(u) * np.sin(v)
    Z = c * np.cos(v)
    ax.plot_wireframe(X, Y, Z, **kwargs)
    return ax


def plot_trajectory(
    trajectory: np.ndarray,
    rotated_trajectory: np.ndarray,
    mascon_points: np.ndarray,
    safety_coefficient: float = 1.4,
    exit_radius: float = 2.0,
):
    """
    Plot the body-frame and inertial-frame trajectories with asteroid model.
    """
    fig = plt.figure(figsize=(6, 6))
    a = (
        (np.max(mascon_points[:, 0]) - np.min(mascon_points[:, 0]))
        / 2
        * safety_coefficient
    )
    b = (
        (np.max(mascon_points[:, 1]) - np.min(mascon_points[:, 1]))
        / 2
        * safety_coefficient
    )
    c = (
        (np.max(mascon_points[:, 2]) - np.min(mascon_points[:, 2]))
        / 2
        * safety_coefficient
    )
    D = 3

    def plot_panel(ax, traj, az, el, D, title):
        ax.scatter3D(
            mascon_points[:, 0],
            mascon_points[:, 1],
            mascon_points[:, 2],
            alpha=0.05,
            s=2,
            c="k",
        )
        _plot_ellipsoid(ax, a, b, c, color="r", alpha=0.05)
        _plot_ellipsoid(
            ax, exit_radius, exit_radius, exit_radius, color="y", alpha=0.05
        )
        ax.plot3D(traj[:, 0], traj[:, 1], traj[:, 2])
        ax.set_xlim(-D, D)
        ax.set_ylim(-D, D)
        ax.set_zlim(-D, D)
        ax.view_init(az, el)
        ax.set_title(title)
        ax.set_xticks([-2, -1, 0, 1, 2])
        ax.set_yticks([-2, -1, 0, 1, 2])
        ax.set_zticks([])

    # Body frame views
    plot_panel(
        fig.add_subplot(221, projection="3d"), trajectory, 0, 90, D, "body frame"
    )
    plot_panel(
        fig.add_subplot(222, projection="3d"), trajectory, 90, 0, D, "body frame"
    )
    plot_panel(fig.add_subplot(223, projection="3d"), trajectory, 0, 0, D, "body frame")
    # Inertial frame view
    plot_panel(
        fig.add_subplot(224, projection="3d"),
        rotated_trajectory,
        0,
        90,
        D,
        "inertial frame",
    )

    plt.tight_layout()
    return fig
