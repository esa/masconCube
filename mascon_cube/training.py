from abc import ABC
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from tqdm import tqdm

from mascon_cube import losses
from mascon_cube.constants import TENSORBOARD_DIR
from mascon_cube.data.datasets import AccelerationDataset
from mascon_cube.data.mascon_model import MasconModel
from mascon_cube.logs import LogConfig, SummaryWriter
from mascon_cube.models import MasconCube
from mascon_cube.utils import compute_acceleration
from mascon_cube.visualization import plot_mascon_cube


@dataclass
class AbstractTrainingConfig(ABC):
    """Abstract training config"""

    asteroid: str
    train_set_path: Path
    n_epochs: int = 10
    batch_size: int = 1000
    loss_fn: str = "normalized_l1_loss"
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
    data_from_trajectory: bool = False
    traj_start_orb_params: tuple[float, float, float, float, float, float] = (
        1.5,
        0.0,
        np.pi / 2,
        0.0,
        0.0,
        np.pi / 2,
    )


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

    training_data = AccelerationDataset(config.train_set_path)
    dataloader = torch.utils.data.DataLoader(
        training_data, batch_size=config.batch_size, shuffle=True
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
    iterator = (
        tqdm(range(config.n_epochs), unit="epoch", desc="Training", position=0)
        if progressbar
        else range(config.n_epochs)
    )
    for i in iterator:
        internal_iterator = (
            tqdm(
                dataloader, leave=False, unit="batch", desc="Current epoch", position=1
            )
            if progressbar
            else dataloader
        )
        for target_points, labels in internal_iterator:
            target_points = target_points.to(device)
            labels = labels.to(device)
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
                    fig = plot_mascon_cube(cube)
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


def training_from_trajectory(
    config: CubeTrainingConfig,
    traj_start_orb_params: tuple[float, float, float, float, float, float] = (
        1.5,
        0.0,
        np.pi / 2,
        0.0,
        0.0,
        np.pi / 2,
    ),
    val_config: Optional[ValidationConfig] = None,
    log_config: Optional[LogConfig] = None,
    device: Union[str, torch.device] = "cuda",
    progressbar: bool = False,
) -> MasconCube:
    """Train the mascon cube to fit the ground truth using data from a trajectory

    Args:
        config (CubeTrainingConfig): Training configuration
        traj_start_orb_params (tuple): Initial orbital parameters for the trajectory
        val_config (Optional[ValidationConfig]): Validation configuration. Defaults to None (no validation).
        log_config (Optional[LogConfig]): Logging configuration. Defaults to None (no logging).
        device (Union[str, torch.device]): Device to use for training. Defaults to "cuda".
        progressbar (bool, optional): If True show a progressbar on command line. Defaults to False.
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
    training_data = AccelerationDataset(
        n=config.n_samples,
        asteroid=config.asteroid,
        sampling_method=config.sampling_method,
        sampling_min=config.sampling_min,
        sampling_max=config.sampling_max,
        seed=42,
        cache=True,
    )
    dataloader = torch.utils.data.DataLoader(
        training_data, batch_size=config.batch_size, shuffle=True
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
    iterator = (
        tqdm(range(config.n_epochs), unit="epoch", desc="Training", position=0)
        if progressbar
        else range(config.n_epochs)
    )
    for i in iterator:
        internal_iterator = (
            tqdm(
                dataloader, leave=False, unit="batch", desc="Current epoch", position=1
            )
            if progressbar
            else dataloader
        )
        for target_points, labels in internal_iterator:
            target_points = target_points.to(device)
            labels = labels.to(device)
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
                    fig = plot_mascon_cube(cube)
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
