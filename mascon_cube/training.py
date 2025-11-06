import warnings
from abc import ABC
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Union

import seaborn as sns
import torch
from tqdm import tqdm

import wandb
from mascon_cube import losses
from mascon_cube.constants import MASS_VMAX
from mascon_cube.data.datasets import AccelerationDataset
from mascon_cube.data.mascon_model import MasconModel
from mascon_cube.models import MasconCube
from mascon_cube.utils import compute_acceleration
from mascon_cube.visualization import mascon_cube_to_point_cloud, plot_dataset


@dataclass
class AbstractTrainingConfig(ABC):
    """Abstract training config"""

    asteroid: str
    train_set_path: Path
    val_set_path: Path
    starting_training_data: int = -1
    add_data_every_n_epochs: int = 0
    data_to_add: int = 0
    warmup_epochs: int = 0
    n_epochs: int = 10
    batch_size: int = 1000
    loss_fn: str = "normalized_l1_loss"
    lr: float = 1e-5
    scheduler_factor: float = 0.8
    scheduler_patience: int = 200
    scheduler_min_lr: float = 1e-8
    val_set_path: Optional[Path] = None
    val_every_n_epochs: int = 50
    batch_persistency: int = 10


@dataclass
class CubeTrainingConfig(AbstractTrainingConfig):
    """Dataclass for training configuration"""

    cube_side: int = 100
    differential: bool = False
    normalize: bool = True
    activation_function: str = "linear"


def training_loop(
    config: CubeTrainingConfig,
    device: Union[str, torch.device] = "cuda",
    progressbar: bool = False,
    use_wandb: bool = False,
) -> MasconCube:
    """Train the mascon cube to fit the ground truth

    Args:
        config (CubeTrainingConfig): Training configuration
        device (Union[str, torch.device]): Device to use for training. Defaults to "cuda".
        progressbar (bool, optional): If True show a progressbar on command line. Defaults to False.
        use_wandb (bool, optional): If True, log training to Weights and Biases. Defaults to False.

    Returns:
        MasconCube: The trained MasconCube
    """
    wandb_run = (
        wandb.init(
            project="mascon-cube",
            config=asdict(config),
        )
        if use_wandb
        else None
    )
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

    def lr_lambda(i):
        if config.warmup_epochs == 0:
            return 1.0
        else:
            return min(1.0, i / config.warmup_epochs)

    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=config.scheduler_factor,
        patience=config.scheduler_patience,
        min_lr=config.scheduler_min_lr,
    )

    training_data = AccelerationDataset(
        config.train_set_path, config.starting_training_data
    )
    if wandb_run is not None:
        sns.set_theme()
        sns.set_style("whitegrid")
        fig = plot_dataset(
            training_data,
            cube,
        )
        wandb_run.log({"data": wandb.Image(fig)})
    dataloader = torch.utils.data.DataLoader(
        training_data, batch_size=config.batch_size, shuffle=True
    )
    loss_fn = getattr(losses, config.loss_fn)

    best_cube = deepcopy(cube)
    best_loss = float("inf")
    val_dataset = torch.load(config.val_set_path).to("cpu")

    iterator = (
        tqdm(range(config.n_epochs), unit="epoch", desc="Training", position=0)
        if progressbar
        else range(config.n_epochs)
    )
    for i in iterator:
        _train(
            dataloader,
            progressbar,
            device,
            cube,
            loss_fn,
            i,
            optimizer,
            wandb_run,
            config.batch_persistency,
        )
        val_loss = _validate(val_dataset, device, cube, ground_truth, loss_fn)
        if val_loss < best_loss:
            best_loss = val_loss
            best_cube = deepcopy(cube)
        if wandb_run is not None:
            wandb_run.log({"val_loss": val_loss, "epoch": i})
            point_cloud = mascon_cube_to_point_cloud(
                cube, range=(0, MASS_VMAX[config.asteroid])
            )
            wandb_run.log({"cube": wandb.Object3D(point_cloud), "epoch": i})
            # fig = plot_mascon_cube(cube, range=(0, MASS_VMAX[config.asteroid]))
            # wandb_run.log({"Cube": wandb.Image(fig), "epoch": i})
        if i < config.warmup_epochs:
            warmup_scheduler.step()
        else:
            plateau_scheduler.step(val_loss)
        if wandb_run is not None:
            wandb_run.log({"lr": optimizer.param_groups[0]["lr"], "epoch": i})
        if (
            config.add_data_every_n_epochs > 0
            and (i + 1) % config.add_data_every_n_epochs == 0
        ):
            training_data.add_data(config.data_to_add)

    if wandb_run is not None:
        wandb_run.finish()

    if training_data.n != len(training_data.data):
        warnings.warn("Training data was not fully used during training.")
    return best_cube


def _train(
    dataloader, progressbar, device, cube, loss_fn, i, optimizer, wandb_run, persistency
):
    iterator = (
        tqdm(dataloader, leave=False, unit="batch", desc="Current epoch", position=1)
        if progressbar
        else dataloader
    )
    for target_points, labels in iterator:
        target_points = target_points.to(device)
        labels = labels.to(device)
        for _ in range(persistency):
            predicted = compute_acceleration(target_points, cube.coords, cube.masses)
            loss = loss_fn(predicted, labels)

            if wandb_run is not None:
                wandb_run.log({"train_loss": loss.item(), "epoch": i})

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


@torch.inference_mode()
def _validate(val_dataset, device, cube, ground_truth, loss_fn):
    val_dataset = val_dataset.to(device)
    val_labels = compute_acceleration(
        val_dataset, ground_truth.coords, ground_truth.masses
    )
    val_predicted = compute_acceleration(val_dataset, cube.coords, cube.masses)
    val_dataset = val_dataset.to("cpu")
    val_loss = loss_fn(val_predicted, val_labels).item()
    return val_loss
