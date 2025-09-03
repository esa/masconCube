from copy import deepcopy
from dataclasses import asdict
from datetime import datetime

import numpy as np
import torch
from tqdm import tqdm

from mascon_cube import geodesynet
from mascon_cube.constants import TENSORBOARD_DIR
from mascon_cube.data.mascon_model import MasconModel
from mascon_cube.data.sampling import get_target_point_sampler
from mascon_cube.logs import LogConfig, SummaryWriter
from mascon_cube.pinn_gm import PinnGM, pinn_loss
from mascon_cube.training import AbstractTrainingConfig, ValidationConfig


class PinnTrainingConfig(AbstractTrainingConfig):
    """Dataclass for training configuration for PINN models"""

    n_epochs: int = 8192
    batch_size: int = 2048
    n_data: int = 100000
    sampling_method: str = "spherical"
    sampling_min: float = 0.0
    sampling_max: float = 2.4
    lr: float = 0.00390625


def training_loop(
    config: PinnTrainingConfig,
    val_config: ValidationConfig | None = None,
    log_config: LogConfig | None = None,
    device: str | torch.device = "cuda",
    progressbar: bool = True,
) -> PinnGM:
    model = PinnGM().to(device)
    mascon_model = MasconModel(config.asteroid, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    data_sampler = get_target_point_sampler(
        n=config.n_data,
        asteroid_mesh=config.asteroid,
        method=config.sampling_method,
        bounds=(config.sampling_min, config.sampling_max),
        device=device,
    )
    dataset = data_sampler().to(device)
    mascon_points = mascon_model.coords
    mascon_masses = mascon_model.masses
    ground_truth = geodesynet.ACC_L(dataset, mascon_points, mascon_masses).to(device)
    val_dataset = val_config.val_dataset.requires_grad_(True) if val_config else None
    val_labels = (
        geodesynet.ACC_L(val_dataset, mascon_points, mascon_masses)
        .requires_grad_(True)
        .to(device)
    )
    loss_fn = pinn_loss
    # And init the best results
    best_loss = np.inf
    best_model = None

    if log_config is not None:
        log_dir = (
            TENSORBOARD_DIR
            / config.asteroid
            / datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        )
        writer = SummaryWriter(log_dir=log_dir)

    iterator = tqdm(range(config.n_epochs)) if progressbar else range(config.n_epochs)

    for i in iterator:
        # shuffle the dataset
        indeces = torch.randperm(dataset.size()[0])
        dataset = dataset[indeces]
        ground_truth = ground_truth[indeces]
        # batchify the dataset
        for j in range(0, len(dataset), config.batch_size):
            batch = dataset[j : j + config.batch_size]
            labels = ground_truth[j : j + config.batch_size]
            # Require grad
            batch = batch.requires_grad_(True)
            # We compute the values predicted by the neural density field
            predicted = model(batch)
            predicted = torch.autograd.grad(
                predicted,
                batch,
                grad_outputs=torch.ones_like(predicted),
                create_graph=True,
            )[0]
            # Compute the labels whenever the target points are changed
            # labels = geodesynet.ACC_L(batch, mascon_points, mascon_masses).to(device)
            # Compute the loss
            loss = loss_fn(predicted, labels)
            if val_config is None and loss.item() < best_loss:
                # If we don't have a validation set, we use the training loss to determine the best model
                best_loss = loss.item()
                best_model = deepcopy(model).cpu()
            # Backpropagate
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if val_config and i % val_config.val_every_n_epochs == 0:
            # If we have a validation set, we use the validation loss to determine the best model
            val_predicted = model(val_dataset)
            val_predicted = torch.autograd.grad(
                val_predicted,
                val_dataset,
                grad_outputs=torch.ones_like(val_predicted),
                create_graph=True,
            )[0]
            val_loss = loss_fn(val_predicted, val_labels).item()
            if val_loss < best_loss:
                best_loss = val_loss
                best_model = deepcopy(model).cpu()
            optimizer.zero_grad()

        # Tensorboard logging
        if log_config is not None:
            if i % log_config.log_every_n_epochs == 0:
                writer.add_scalar("Loss/train", loss.item(), i)
            if val_config is not None and i % val_config.val_every_n_epochs == 0:
                writer.add_scalar("Loss/val", val_loss, i)

    if log_config is not None:
        writer.add_hparams(asdict(config), {"best_loss": best_loss})

        torch.save(best_model.state_dict(), log_dir / "best_model_state_dict.pt")
        writer.close()

    return best_model
