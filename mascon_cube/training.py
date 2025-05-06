from abc import ABC
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Optional, Union

import torch
from tqdm import tqdm

from mascon_cube import losses
from mascon_cube.constants import TENSORBOARD_DIR
from mascon_cube.data.mascon_model import MasconModel
from mascon_cube.data.sampling import get_target_point_sampler
from mascon_cube.logs import LogConfig, SummaryWriter
from mascon_cube.models import MasconCube
from mascon_cube.visualization import plot_mascon_cube


@dataclass
class AbstractTrainingConfig(ABC):
    """Abstract training config"""

    asteroid: str
    n_epochs: int = 1000
    n_epochs_before_resampling: int = 10
    loss_fn: str = "normalized_l1_loss"
    batch_size: int = 1000
    sampling_method: str = "spherical"
    sampling_min: float = 0.0
    sampling_max: float = 1.0
    lr: float = 1e-5
    scheduler_factor: float = 0.8
    scheduler_patience: int = 200
    scheduler_min_lr: float = 1e-8


@dataclass
class CubeTrainingConfig(AbstractTrainingConfig):
    """Dataclass for training configuration"""

    cube_side: int = 100
    differential: bool = False
    normalize: bool = True
    activation_function: str = "linear"


@dataclass
class ValidationConfig:
    """Dataclass for validation configuration"""

    val_dataset: Optional[torch.Tensor] = None
    val_every_n_epochs: int = 50


def training_loop(
    config: CubeTrainingConfig,
    val_config: Optional[ValidationConfig] = None,
    log_config: Optional[LogConfig] = None,
    device: Union[str, torch.device] = "cuda",
    progressbar: bool = False,
) -> MasconCube:
    """Train the mascon cube to fit the ground truth

    Args:
        config (CubeTrainingConfig): Training configuration
        val_config (Optional[ValidationConfig]): Validation configuration. Defaults to None (no validation).
        log_config (Optional[LogConfig]): Logging configuration. Defaults to None (no logging).
        device (Union[str, torch.device]): Device to use for training. Defaults to "cuda".
        progressbar (bool, optional): If True show a progressbar on command line. Defaults to False.

    Returns:
        MasconCube: The trained MasconCube
    """
    cube = MasconCube(
        config.cube_side,
        config.asteroid,
        device=device,
        differential=config.differential,
        normalize=config.normalize,
        activation_function=config.activation_function,
    )
    ground_truth = MasconModel(config.asteroid, device=device)
    optimizer = torch.optim.Adam([cube.weights], lr=config.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=config.scheduler_factor,
        patience=config.scheduler_patience,
        min_lr=config.scheduler_min_lr,
    )
    data_sampler = get_target_point_sampler(
        n=config.batch_size,
        asteroid_mesh=config.asteroid,
        method=config.sampling_method,
        bounds=(config.sampling_min, config.sampling_max),
        device=device,
    )
    loss_fn = getattr(losses, config.loss_fn)

    best_cube = deepcopy(cube)
    best_loss = float("inf")

    if log_config is not None:
        log_dir = (
            TENSORBOARD_DIR
            / config.asteroid
            / datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        )
        writer = SummaryWriter(log_dir=log_dir)
    iterator = tqdm(range(config.n_epochs)) if progressbar else range(config.n_epochs)
    for i in iterator:
        if (i % config.n_epochs_before_resampling) == 0:
            target_points = data_sampler()
            labels = compute_acceleration(
                target_points, ground_truth.coords, ground_truth.masses
            )

        predicted = compute_acceleration(target_points, cube.coords, cube.masses)
        loss = loss_fn(predicted, labels)

        if val_config is None and loss.item() < best_loss:
            # If we don't have a validation set, we use the training loss to determine the best model
            best_loss = loss.item()
            best_cube = deepcopy(cube)

        if val_config and i % val_config.val_every_n_epochs == 0:
            # If we have a validation set, we use the validation loss to determine the best model
            with torch.no_grad():
                val_labels = compute_acceleration(
                    val_config.val_dataset, ground_truth.coords, ground_truth.masses
                )
                val_predicted = compute_acceleration(
                    val_config.val_dataset, cube.coords, cube.masses
                )
                val_loss = loss_fn(val_predicted, val_labels).item()
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_cube = deepcopy(cube)

        # Tensorboard logging
        if log_config is not None:
            if i % log_config.log_every_n_epochs == 0:
                writer.add_scalar("Loss/train", loss.item(), i)
            if val_config is not None and i % val_config.val_every_n_epochs == 0:
                writer.add_scalar("Loss/val", val_loss, i)
            if i % log_config.draw_every_n_epochs == 0:
                fig = plot_mascon_cube(cube, range=(1e-5, 3e-5))
                writer.add_figure("Cube", fig, i)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(loss.item())

    if log_config is not None:
        writer.add_hparams(asdict(config), {"best_loss": best_loss})
        torch.save(best_cube, log_dir / "best_cube.pt")
        writer.close()

    return best_cube


def compute_acceleration(
    target_points: torch.Tensor,
    mascon_points: torch.Tensor,
    mascon_masses: torch.Tensor,
):
    """
    Computes the acceleration due to the mascon at the target points. (to be used as Label in the training)

    Args:
        target_points (2-D array-like): an (N, 3) array-like object containing the coordinates of the points where the
            acceleration should be computed.
        mascon_points (2-D array-like): an (N, 3) array-like object containing the points that belong to the mascon
        mascon_masses (1-D array-like): a (N,) array-like object containing the values for the mascon masses.
            Can also be a scalar containing the mass value for all points.

    Returns:
        1-D array-like: a (N, 3) torch tensor containing the acceleration (G=1) at the target points
    """
    device = target_points.device
    mm = mascon_masses.view(-1, 1)
    retval = torch.empty(len(target_points), 3, device=device)
    for i, target_point in enumerate(target_points):
        dr = torch.sub(mascon_points, target_point)
        retval[i] = torch.sum(
            mm / torch.pow(torch.norm(dr, dim=1), 3).view(-1, 1) * dr, dim=0
        )
    return retval
