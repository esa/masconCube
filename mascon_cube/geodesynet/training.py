import warnings
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import torch
from tqdm import tqdm

from mascon_cube import losses
from mascon_cube.constants import TENSORBOARD_DIR
from mascon_cube.data.mascon_model import MasconModel
from mascon_cube.data.sampling import get_target_point_sampler
from mascon_cube.logs import LogConfig, SummaryWriter
from mascon_cube.training import (
    AbstractTrainingConfig,
    ValidationConfig,
    compute_acceleration,
)

from .integration import ACC_trap
from .models import GeodesyNet


@dataclass
class GeodesyNetTrainingConfig(AbstractTrainingConfig):
    hidden_features: int = 100
    hidden_layers: int = 9
    n_quadrature: int = 300000
    uniform_asteroid: Optional[Union[str, Path]] = None


def geodesynet_training_loop(
    config: GeodesyNetTrainingConfig,
    val_config: Optional[ValidationConfig] = None,
    log_config: Optional[LogConfig] = None,
    device: Union[str, torch.device] = "cuda",
    progressbar: bool = False,
) -> GeodesyNet:
    """Train the GeodesyNet to fit the ground truth

    Args:
        config (GeodesyNetTrainingConfig): Training configuration
        val_config (Optional[ValidationConfig]): Validation configuration. Defaults to None (no validation).
        log_config (Optional[LogConfig]): Logging configuration. Defaults to None (no logging).
        device (Union[str, torch.device]): Device to use for training. Defaults to "cuda".
        progressbar (bool, optional): If True show a progressbar on command line. Defaults to False.

    Returns:
        GeodesyNet: The trained GeodesyNet
    """
    do_differential_training = config.uniform_asteroid is not None
    net = GeodesyNet(
        hidden_layers=config.hidden_layers, hidden_features=config.hidden_features
    )
    net = net.to(device=device)
    ground_truth = MasconModel(config.asteroid, device=device)
    if do_differential_training:
        uniform_ground_truth = MasconModel(config.uniform_asteroid, device=device)
    optimizer = torch.optim.Adam(net.parameters(), lr=config.lr)
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

    best_net = deepcopy(net)
    best_loss = float("inf")

    if log_config is not None:
        log_dir = (
            TENSORBOARD_DIR
            / config.asteroid
            / "geodesynet"
            / datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        )
        writer = SummaryWriter(log_dir=log_dir)
    iterator = tqdm(range(config.n_epochs)) if progressbar else range(config.n_epochs)
    for i in iterator:
        if (i % config.n_epochs_before_resampling) == 0:
            target_points = data_sampler()
            if do_differential_training:
                labels = compute_acceleration(
                    target_points, ground_truth.coords, ground_truth.masses
                )
            else:
                labels_u = compute_acceleration(
                    target_points,
                    uniform_ground_truth.coords,
                    uniform_ground_truth.masses,
                )
                labels_nu = compute_acceleration(
                    target_points, ground_truth.coords, ground_truth.masses
                )
                labels = labels_nu - labels_u

        predicted = ACC_trap(target_points, net, n=config.n_quadrature, noise=0.0)
        # We learn the scaling constant (k in the paper)
        c = torch.sum(predicted * labels) / torch.sum(predicted * predicted)  # noqa: F841 : could be useful later
        loss = loss_fn(predicted, labels)

        if val_config is None and loss.item() < best_loss:
            # If we don't have a validation set, we use the training loss to determine the best model
            best_loss = loss.item()
            best_net = deepcopy(net)

        if val_config and i % val_config.val_every_n_epochs == 0:
            # If we have a validation set, we use the validation loss to determine the best model
            with torch.no_grad():
                val_labels = compute_acceleration(
                    val_config.val_dataset, ground_truth.coords, ground_truth.masses
                )
                val_predicted = ACC_trap(
                    val_config.val_dataset, net, net, n=config.n_quadrature, noise=0.0
                )
                val_loss = loss_fn(val_predicted, val_labels).item()
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_net = deepcopy(net)

        # Tensorboard logging
        if log_config is not None:
            if i % log_config.log_every_n_epochs == 0:
                writer.add_scalar("Loss/train", loss.item(), i)
            if val_config is not None and i % val_config.val_every_n_epochs == 0:
                writer.add_scalar("Loss/val", val_loss, i)
            if i % log_config.draw_every_n_epochs == 0:
                # TODO: implement visualization function
                warnings.warn("Visualization for GeodesyNet not implemented yet!")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(loss.item())

    if log_config is not None:
        writer.add_hparams(asdict(config), {"best_loss": best_loss})
        torch.save(best_net, log_dir / "best_net.pt")
        writer.close()

    return best_net
