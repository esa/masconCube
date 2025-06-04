import matplotlib as mpl
import numpy as np
import torch
import tqdm
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation as rotation

from ._utils import compute_density


def plot_model_contours(
    model,
    encoding,
    heatmap=False,
    section=np.array([0, 0, 1]),
    N=100,
    save_path=None,
    offset=0.0,
    axes=None,
    c=1.0,
    levels=[0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
    add_shape_base_value=None,
    add_const_density=1.0,
    geomscale=False,
):
    """Takes a mass density model and plots the density contours of its section with
       a 2D plane

    Args:
        model (callable (N,M)->1): neural model for the asteroid.
        encoding: the encoding for the neural inputs.
        section (Numpy array (3)): the section normal (can also be not of unitary magnitude)
        N (int): number of points in each axis of the 2D grid
        save_path (str, optional): Pass to store plot, if none will display. Defaults to None.
        offset (float): an offset to apply to the plane in the direction of the section normal
        axes (matplolib axes): the axes where to plot. Defaults to None, in which case axes are created.
        levels (list optional): the contour levels to be plotted. Defaults to [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7].
        add_shape_base_value (str): path to asteroid mesh which is then used to add 1 to density inside asteroid
        add_const_density (float): density to add inside asteroid if add_shape_base_value was passed
    """
    # Builds a 2D grid on the z = 0 plane
    x, y = np.meshgrid(np.linspace(-1, 1, N), np.linspace(-1, 1, N))
    x = np.reshape(x, (-1,))
    y = np.reshape(y, (-1,))
    z = np.zeros(N**2)
    p = np.zeros((N**2, 3))
    p[:, 0] = x
    p[:, 1] = y
    p[:, 2] = z

    # The cross product between the vertical and the desired direction ...
    section = section / np.linalg.norm(section)
    cp = np.cross(np.array([0, 0, 1]), section)
    # safeguard against singularity
    if np.linalg.norm(cp) > 1e-8:
        # ... allows to find the rotation  amount ...
        sint = np.linalg.norm(cp)
        # ... and the axis ...
        axis = cp
        # ... which we transform into a rotation vector (scipy convention)
        rotvec = axis * (np.arcsin(sint))
    else:
        rotvec = np.array([0.0, 0.0, 0.0])
    # ... used to build the rotation matrix
    Rm = rotation.from_rotvec(rotvec).as_matrix()
    # We rotate the points ...
    newp = [np.dot(Rm.transpose(), p[i, :]) for i in range(N**2)]
    # ... and translate
    newp = newp + section * offset
    # ... and compute them

    position = torch.tensor(newp, dtype=torch.float32).requires_grad_(True)
    potential = model(encoding(position))
    # scale proxy potential into true potential, see section 3.3 of PINN paper
    r = torch.norm(position, dim=1).view(-1, 1)
    n = torch.where(r > 1, r, torch.ones_like(r))
    potential = potential / n
    # enforce boundary conditions, see section 3.4 of PINN paper
    k = 0.5
    r_ref = 3
    w_bc = (1 + torch.tanh(k * (r - r_ref))) / 2
    w_nn = 1 - w_bc
    u_bc = 1 / r  # u_bc = mu / r.    mu = M * G = 1 assuming G = 1 and M = 1
    u_nn = potential
    potential = w_bc * u_bc + w_nn * u_nn
    rho = compute_density(position, potential, G=1)

    Z = rho.reshape((N, N)).cpu().detach().numpy()

    X, Y = np.meshgrid(np.linspace(-1, 1, N), np.linspace(-1, 1, N))
    if axes is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    else:
        ax = axes

    # CAREFUL: ugly workaround to fix axis ...
    if (section == np.array([1, 0, 0])).all():
        X, Y = Y, X

    if heatmap:
        gradient = np.linspace(0, 255, len(levels), dtype=np.uint8).reshape(-1, 1)
        cmap = plt.get_cmap("YlOrRd")
        colors = [cmap(i) for i in gradient]
        ticks = [f"{lev:.2f}" for lev in levels]
        ticks[0] = ""
        ticks[1] = f"< {ticks[1]}"
        ticks[-1] = ""
        ticks[-2] = f"> {ticks[-2]}"
        p = ax.contourf(X, Y, Z, levels=levels, colors=colors)
        cb = plt.colorbar(p, ax=ax)
        cb.set_ticklabels(ticks)
    else:
        cmap = mpl.cm.viridis
        p = ax.contour(X, Y, Z, cmap=cmap, levels=levels)
        norm = mpl.colors.BoundaryNorm(levels, cmap.N)
        cb = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
    cb.ax.tick_params(labelsize=6)
    cb.set_label("Density", rotation=270, labelpad=15, fontsize=8)

    if save_path is not None:
        plt.savefig(save_path, dpi=300)

    if axes is None:
        return ax


def plot_pinn_vs_mascon_contours(
    model,
    encoding,
    mascon_points,
    mascon_masses=None,
    N=2500,
    crop_p=1e-2,
    s=100,
    save_path=None,
    c=1.0,
    progressbar=False,
    offset=0.0,
    heatmap=False,
    mascon_alpha=0.05,
    add_shape_base_value=None,
    add_const_density=1.0,
    geomscale=False,
):
    """Plots both the mascon and model contours in one figure for direct comparison

    Args:
        model (callable (N,M)->1): neural model for the asteroid.
        encoding: the encoding for the neural inputs.
        mascon_points (2-D array-like): an (N, 3) array-like object containing the coordinates of the mascon points.
        mascon_masses (1-D array-like): a (N,) array-like object containing the values for the mascon masses.
        N (int): number of points to be considered.
        views_2d (bool): activates also the 2d projections.
        crop_p (float): all points below this density are rejected.
        s (int): size of the non rejected points visualization.
        save_path (str, optional): Pass to store plot, if none will display. Defaults to None.
        c (float, optional): Normalization constant. Defaults to 1.
        backcolor (list, optional): Plot background color. Defaults to [0.15, 0.15, 0.15].
        progressbar (bool, optional): activates a progressbar. Defaults to False.
        offset (float): an offset to apply to the plane in the direction of the section normal
        heatmap (bool): determines if contour lines or heatmap are displayed
        mascon_alpha (float): alpha of the overlaid mascon model. Defaults to 0.05.
        add_shape_base_value (str): path to asteroid mesh which is then used to add 1 to density inside asteroid
        add_const_density (float): density to add inside asteroid if add_shape_base_value was passed
    """

    # Mascon masses
    x = mascon_points[:, 0].cpu()
    y = mascon_points[:, 1].cpu()
    z = mascon_points[:, 2].cpu()

    s = 22000 / len(mascon_points)

    if mascon_masses is None:
        normalized_masses = torch.tensor(
            [1.0 / len(mascon_points)] * len(mascon_points)
        )
    else:
        normalized_masses = mascon_masses / sum(mascon_masses)
    normalized_masses = (normalized_masses * s * len(x)).cpu()

    torch.manual_seed(42)  # Seed torch to always get the same points
    points = []
    rho = []
    batch_size = 4096
    found = 0
    if progressbar:
        pbar = tqdm(desc="Sampling points...", total=N)
    while found < N:
        candidates = torch.rand(batch_size, 3) * 2 - 1
        if True:
            outside_pos = torch.stack([torch.zeros(3), torch.ones(3)]).requires_grad_(
                True
            )
            potential_outside = model(encoding(outside_pos))
            r = torch.norm(outside_pos, dim=1).view(-1, 1)
            n = torch.where(r > 1, r, torch.ones_like(r))
            potential_outside = potential_outside / n
            # enforce boundary conditions, see section 3.4 of PINN paper
            k = 0.5
            r_ref = 3
            w_bc = (1 + torch.tanh(k * (r - r_ref))) / 2
            w_nn = 1 - w_bc
            u_bc = 1 / r  # u_bc = mu / r.    mu = M * G = 1 assuming G = 1 and M = 1
            u_nn = potential_outside
            potential_outside = w_bc * u_bc + w_nn * u_nn
            position = candidates.requires_grad_(True)
            potential = model(encoding(position))
            # scale proxy potential into true potential, see section 3.3 of PINN paper
            r = torch.norm(position, dim=1).view(-1, 1)
            n = torch.where(r > 1, r, torch.ones_like(r))
            potential = potential / n
            # enforce boundary conditions, see section 3.4 of PINN paper
            w_bc = (1 + torch.tanh(k * (r - r_ref))) / 2
            w_nn = 1 - w_bc
            u_bc = 1 / r  # u_bc = mu / r.    mu = M * G = 1 assuming G = 1 and M = 1
            u_nn = potential
            potential = w_bc * u_bc + w_nn * u_nn
            rho_candidates = compute_density(position, potential, G=1).unsqueeze(1)

        mask = (torch.abs(rho_candidates) > (torch.rand(batch_size, 1) + crop_p)) & (
            ~torch.isnan(rho_candidates)
        )
        rho_candidates = rho_candidates[mask]
        candidates = [
            [it[0].item(), it[1].item(), it[2].item()]
            for it, m in zip(candidates, mask)
            if m
        ]
        if len(candidates) == 0:
            print("All points rejected! Plot is empty, try cropping less?")
            return
        points.append(torch.tensor(candidates))
        rho.append(rho_candidates)
        found += len(rho_candidates)
        if progressbar:
            pbar.update(len(rho_candidates))
    if progressbar:
        pbar.close()
    points = torch.cat(points, dim=0)[:N]  # concat and discard after N
    rho = torch.cat(rho, dim=0)[:N]  # concat and discard after N
    levels = np.arange(
        np.min(rho.cpu().detach().numpy()),
        np.max(rho.cpu().detach().numpy()) + 0.002,
        0.001,
    )

    fig = plt.figure(figsize=(6, 6), dpi=100, facecolor="white")
    ax = fig.add_subplot(2, 2, 1, projection="3d")
    # ax.set_facecolor(backcolor)
    rejection_col = "yellow"
    mascon_color = "green"

    # And we plot it
    ax.scatter(x, y, z, color="k", s=normalized_masses, alpha=0.01)
    ax.scatter(
        points[:, 0].cpu(),
        points[:, 1].cpu(),
        points[:, 2].cpu(),
        marker=".",
        c=rejection_col,
        s=s * 2,
        alpha=0.1,
    )
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.view_init(elev=45.0, azim=45.0)
    ax.tick_params(labelsize=6)
    ax.set_xlabel("X", fontsize=8)
    ax.set_ylabel("Y", fontsize=8)
    ax.set_zlabel("Z", fontsize=8)

    # X Rectangle
    ax.plot_wireframe(
        np.asarray([[0, 0], [0, 0]]) + offset,
        np.asarray([[1, 1], [-1, -1]]),
        np.asarray([[-1, 1], [-1, 1]]),
        color="red",
        linestyle="--",
        alpha=0.75,
    )
    # Y Rectangle
    ax.plot_wireframe(
        np.asarray([[1, 1], [-1, -1]]),
        np.asarray([[0, 0], [0, 0]]) + offset,
        np.asarray([[-1, 1], [-1, 1]]),
        color="blue",
        linestyle="--",
        alpha=0.75,
    )
    # Z Rectangle
    ax.plot_wireframe(
        np.asarray([[-1, 1], [-1, 1]]),
        np.asarray([[1, 1], [-1, -1]]),
        np.asarray([[0, 0], [0, 0]]) + offset,
        color="green",
        linestyle="--",
        alpha=0.75,
    )
    ax.set_title("3D View", fontsize=7)

    mascon_slice_thickness = 0.01

    ax2 = fig.add_subplot(2, 2, 2)
    # ax2.set_facecolor(backcolor)
    mask = torch.logical_and(
        z - offset < mascon_slice_thickness, z - offset > -mascon_slice_thickness
    )
    _ = plot_model_contours(
        model,
        encoding,
        section=np.array([0, 0, 1]),
        axes=ax2,
        levels=levels,
        c=c,
        offset=offset,
        heatmap=heatmap,
        add_shape_base_value=add_shape_base_value,
        add_const_density=add_const_density,
        geomscale=geomscale,
    )
    ax2.scatter(
        x[mask],
        y[mask],
        color=mascon_color,
        s=normalized_masses[mask],
        alpha=mascon_alpha,
    )

    ax2.set_xlim([-1, 1])
    ax2.set_ylim([-1, 1])
    ax2.tick_params(labelsize=6, color="green")
    ax2.set_xlabel("X", fontsize=8)
    ax2.set_ylabel("Y", fontsize=8)
    ax2.spines["bottom"].set_color("green")
    ax2.spines["top"].set_color("green")
    ax2.spines["right"].set_color("green")
    ax2.spines["left"].set_color("green")
    ax2.set_title("X-Y cross section (green slice)", fontsize=8)
    ax2.set_aspect("equal", "box")

    ax3 = fig.add_subplot(2, 2, 3)
    # ax3.set_facecolor(backcolor)
    mask = torch.logical_and(
        y - offset < mascon_slice_thickness, y - offset > -mascon_slice_thickness
    )
    _ = plot_model_contours(
        model,
        encoding,
        section=np.array([0, 1, 0]),
        axes=ax3,
        levels=levels,
        c=c,
        offset=offset,
        heatmap=heatmap,
        add_shape_base_value=add_shape_base_value,
        add_const_density=add_const_density,
        geomscale=geomscale,
    )
    ax3.scatter(
        x[mask],
        z[mask],
        color=mascon_color,
        s=normalized_masses[mask],
        alpha=mascon_alpha,
    )

    ax3.set_xlim([-1, 1])
    ax3.set_ylim([-1, 1])
    ax3.set_xlabel("X", fontsize=8)
    ax3.set_ylabel("Z", fontsize=8)
    ax3.set_title("X-Z cross section (blue slice)", fontsize=8)
    ax3.tick_params(labelsize=6, color="blue")
    ax3.spines["bottom"].set_color("blue")
    ax3.spines["top"].set_color("blue")
    ax3.spines["right"].set_color("blue")
    ax3.spines["left"].set_color("blue")
    ax3.set_aspect("equal", "box")

    ax4 = fig.add_subplot(2, 2, 4)
    # ax4.set_facecolor(backcolor)
    mask = torch.logical_and(
        x - offset < mascon_slice_thickness, x - offset > -mascon_slice_thickness
    )
    _ = plot_model_contours(
        model,
        encoding,
        section=np.array([1, 0, 0]),
        axes=ax4,
        levels=levels,
        c=c,
        offset=offset,
        heatmap=heatmap,
        add_shape_base_value=add_shape_base_value,
        add_const_density=add_const_density,
        geomscale=geomscale,
    )
    ax4.scatter(
        y[mask],
        z[mask],
        color=mascon_color,
        s=normalized_masses[mask],
        alpha=mascon_alpha,
    )
    ax4.set_xlim([-1, 1])
    ax4.set_ylim([-1, 1])
    ax4.set_xlabel("Y", fontsize=8)
    ax4.set_ylabel("Z", fontsize=8)
    ax4.set_title("Y-Z cross section (red slice)", fontsize=8)
    ax4.tick_params(labelsize=6, color="red")
    ax4.spines["bottom"].set_color("red")
    ax4.spines["top"].set_color("red")
    ax4.spines["right"].set_color("red")
    ax4.spines["left"].set_color("red")
    ax4.set_aspect("equal", "box")

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)

    return ax, fig
