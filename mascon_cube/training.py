from copy import deepcopy
from dataclasses import dataclass

import torch
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.tensorboard import SummaryWriter

from mascon_cube.constants import TENSORBOARD_DIR
from mascon_cube.data.mascon_model import MasconModel
from mascon_cube.models import MasconCube


@dataclass
class TrainingConfig:
    """Dataclass for training configuration"""

    n_epochs: int
    n_epochs_before_resampling: int
    loss_fn: callable
    data_sampler: callable
    optimizer: Optimizer
    scheduler: LRScheduler
    use_tensorboard: bool = False


def training_loop(
    cube: MasconCube, ground_truth: MasconModel, config: TrainingConfig
) -> MasconCube:
    """Train the mascon cube to fit the ground truth

    Args:
        cube (MasconCube): MasconCube to train
        ground_truth (MasconModel): Ground truth to fit
        config (TrainingConfig): Training configuration

    Returns:
        MasconCube: The trained MasconCube
    """
    best_cube = deepcopy(cube)
    best_loss = float("inf")
    if config.use_tensorboard:
        writer = SummaryWriter(log_dir=TENSORBOARD_DIR)

    for i in range(config.n_epochs):
        if (i % config.n_epochs_before_resampling) == 0:
            target_points = config.data_sampler()
            labels = compute_acceleration(
                target_points, ground_truth.coords, ground_truth.masses
            )

        predicted = compute_acceleration(target_points, cube.coords, cube.weights)
        loss = config.loss_fn(predicted, labels)

        config.optimizer.zero_grad()
        loss.backward()
        config.optimizer.step()
        config.scheduler.step(loss.item())

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_cube = deepcopy(cube)

        # Tensorboard logging
        if config.use_tensorboard:
            writer.add_scalar("Loss/train", loss.item(), i)

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
    if mascon_masses is None:
        mm = torch.tensor(
            [1.0 / len(mascon_points)] * len(mascon_points), device=device
        ).view(-1, 1)
    elif type(mascon_masses) is int:
        mm = torch.tensor([mascon_masses] * len(mascon_points), device=device).view(
            -1, 1
        )
    else:
        mm = mascon_masses.view(-1, 1)
    retval = torch.empty(len(target_points), 3, device=device)
    for i, target_point in enumerate(target_points):
        dr = torch.sub(mascon_points, target_point)
        retval[i] = torch.sum(
            mm / torch.pow(torch.norm(dr, dim=1), 3).view(-1, 1) * dr, dim=0
        )
    return retval
