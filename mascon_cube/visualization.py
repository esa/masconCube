from typing import Optional

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from torch import Tensor

from mascon_cube.constants import GROUND_TRUTH_DIR
from mascon_cube.data.datasets import AccelerationDataset
from mascon_cube.models import MasconCube


def plot_asteroid(asteroid: str) -> plt.Figure:
    """plot the thetraedrons model of the asteroid in 3D and in 2D sections (XY, XZ, YZ) where the color represents the
        density.

    Args:
        asteroid (str): name of the asteroid. Must be an existing folder in `mascon_cube.constants.GROUND_TRUTH_DIR`

    Returns:
        plt.Figure: The figure with the 4 subplots (3D, XY, XZ, YZ).
    """
    img = plt.imread(GROUND_TRUTH_DIR / asteroid / "combined_plot.png")
    fig = plt.figure(figsize=(10, 10))
    plt.imshow(img)
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


def mascon_cube_to_point_cloud(
    mascon_cube: MasconCube,
    cmap: str = "viridis",
    range: Optional[tuple[float, float]] = None,
) -> np.ndarray:
    """
    Convert the mascon cube to a point cloud numpy array of shape (N, 6): [[x, y, z, r, g, b], ...]
    where r, g, b are values in the range [0, 255] representing the mass color in the given colormap.

    Args:
        mascon_cube (MasconCube): The mascon cube to convert.
        cmap (str, optional): The colormap to use for coloring the points. Defaults to "viridis".
        range (Optional[tuple[float, float]], optional): The range of the colormap. Defaults to None.

    Returns:
        np.ndarray: The point cloud numpy array.
    """
    coords = mascon_cube.coords.detach().cpu().numpy()
    masses = mascon_cube.masses.detach().cpu().numpy()
    colormap = plt.get_cmap(cmap)
    if range is None:
        range = (0, np.percentile(masses, 99))
    # clip masses to the range
    masses_clipped = np.clip(masses, range[0], range[1])
    colors = colormap((masses_clipped - range[0]) / (range[1] - range[0]))[
        :, 0, :3
    ]  # drop alpha channel
    colors = (colors * 255).astype(np.uint8)
    point_cloud = np.hstack((coords, colors))
    return point_cloud


def stokes_degree_error(
    preds: np.ndarray,
    gt: np.ndarray,
    labels: list[str],
    title: str = "",
    vmin: float = 1e-4,
    vmax: float = 5e-1,
    relative: bool = False,
    markers: Optional[list[str]] = None,
) -> plt.Figure:
    if isinstance(preds, Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(gt, Tensor):
        gt = gt.detach().cpu().numpy()
    if preds.ndim == 2:
        preds = np.expand_dims(preds, axis=0)
    fig, ax = plt.subplots(figsize=(14, 4), dpi=200)
    if markers is None:
        markers = ["o"] * len(labels)
    for j, (pred, label) in enumerate(zip(preds, labels)):
        diff = np.abs(pred - gt)
        if relative:
            diff = diff / (np.abs(gt) + 1e-16)
        assert diff.ndim == 2, "pred and gt must be 2D arrays"
        assert diff.shape[0] == diff.shape[1], "pred and gt must be square matrices"
        max_degree = diff.shape[0]
        errors = []
        for i in range(0, max_degree):
            error = diff[: i + 1, : i + 1]
            error = np.mean(error).item()
            errors.append(error)
        errors = np.array(errors)
        ax.semilogy(np.arange(0, max_degree), errors, marker=markers[j], label=label)
    ax.legend()

    ax.set_xlabel(r"Degree $n$")
    ax.set_ylabel(r"$MAE_n$")
    return fig


def stokes_heatmap(
    pred: np.ndarray,
    gt: np.ndarray,
    title: str = "",
    vmin=1e-7,
    vmax=1e-2,
    xlabel: str = r"$m$",
    ylabel: str = r"$l$",
    relative: bool = False,
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
    ax_top.set_xlabel(ylabel)
    ax_top.set_xticks(np.arange(1.5, pred.shape[1], 1), np.arange(1, pred.shape[1], 1))
    ax_right = ax.secondary_yaxis("right")
    ax_right.set_ylabel(xlabel)
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
        r"$\Delta\tilde{S}_{l,m}$",
        fontsize=14,
        horizontalalignment="left",
        verticalalignment="bottom",
        transform=ax.transAxes,
    )
    plt.text(
        -0.01,
        -0.01,
        r"$\Delta\tilde{C}_{l,m}$",
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
    fig = plt.figure(figsize=(9, 4), dpi=100)
    ax0 = fig.add_subplot(121, projection="3d", aspect="equal")
    ax1 = fig.add_subplot(122, projection="3d", aspect="equal")

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
    D = 2

    def plot_panel(ax, traj, az, el, D, title, ticks):
        ax.scatter3D(
            mascon_points[:, 0],
            mascon_points[:, 1],
            mascon_points[:, 2],
            alpha=0.05,
            s=2,
            c="k",
        )
        _plot_ellipsoid(ax, a, b, c, color="r", alpha=0.1)
        _plot_ellipsoid(ax, exit_radius, exit_radius, exit_radius, color="y", alpha=0.1)
        ax.plot3D(traj[:, 0], traj[:, 1], traj[:, 2], color="b")

        ax.view_init(az, el)
        ax.set_title(title)
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")
        ax.set_zlabel(r"$z$")

    plot_panel(ax1, trajectory, 45, 45, D, "Body frame", ticks=["x", "y", "z"])
    plot_panel(
        ax0, rotated_trajectory, 45, 45, D, "Inertial frame", ticks=["x", "y", "z"]
    )

    return fig


def plot_dataset(
    ds: AccelerationDataset,
    cube: MasconCube,
) -> plt.figure:
    fig = plt.figure(figsize=(5, 5), dpi=300)
    ax = fig.add_subplot(111, projection="3d", aspect="equal")
    ax.scatter3D(
        cube.coords[:, 0].cpu().numpy(),
        cube.coords[:, 1].cpu().numpy(),
        cube.coords[:, 2].cpu().numpy(),
        c="k",
        s=2,
        alpha=0.05,
    )
    ax.scatter3D(
        ds.data[:, 0],
        ds.data[:, 1],
        ds.data[:, 2],
        c="b",
        s=1,
        alpha=0.9,
    )
    ax.view_init(45, 45)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_zlabel(r"$z$")
    return fig
