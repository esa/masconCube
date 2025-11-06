from pathlib import Path
from typing import Union

import torch
from torch.utils.data import Dataset

from mascon_cube.constants import TRAIN_DATASETS_DIR
from mascon_cube.data.mascon_model import MasconModel
from mascon_cube.data.sampling import get_target_point_sampler
from mascon_cube.trajectory import simulate_trajectory
from mascon_cube.utils import compute_acceleration


class AccelerationDataset(Dataset):
    def __init__(
        self,
        path: Union[str, Path],
    ):
        data = torch.load(path)
        self.data = data
        self.n = len(self.data)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return self.data[idx][:3], self.data[idx][3:]


def create_random_dataset(
    asteroid: Union[str, Path],
    n: int,
    sampling_method: str = "spherical",
    sampling_min: float = 0.0,
    sampling_max: float = 1.0,
    seed: int = 42,
) -> Path:
    """Create a dataset and save it to disk.

    Args:
        asteroid (Union[str, Path]): Name of the asteroid.
        n (int): Number of samples.
        sampling_method (str, optional): Sampling method to use. Defaults to "spherical".
        sampling_min (float, optional): Minimum value for sampling. Defaults to 0.0.
        sampling_max (float, optional): Maximum value for sampling. Defaults to 1.0.
        seed (int, optional): Random seed for reproducibility. Defaults to 42.

    Returns:
        Path: Path to the saved dataset.
    """

    asteroid_name = asteroid if isinstance(asteroid, str) else asteroid.stem
    dataset_name = (
        f"{asteroid_name}_{n}_{sampling_method}_{sampling_min}_{sampling_max}_{seed}.pt"
    )
    dataset_path = TRAIN_DATASETS_DIR / dataset_name
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sampler = get_target_point_sampler(
        n, asteroid, sampling_method, [sampling_min, sampling_max], device=device
    )
    data = sampler()
    ground_truth = MasconModel(asteroid, device=device)
    labels = compute_acceleration(data, ground_truth.coords, ground_truth.masses)
    data = torch.cat([data.to("cpu"), labels.to("cpu")], dim=1)
    torch.save(data, dataset_path)
    return dataset_path


def create_trajectory_dataset(
    asteroid: Union[str, Path],
    n: int = 1000,
    safety_coefficient: float = 1.4,
    exit_radius: float = 2.0,
    starting_orb_params: tuple[tuple[float, float, float, float, float, float]] = (
        (1.5, 0.0, 3.1415 / 2, 0.0, 0.0, 3.1415 / 2),
    ),
    propagation_days: float = 1.0,
) -> Path:
    """Create a dataset from a simulated trajectory and save it to disk.

    Args:
        asteroid (Union[str, Path]): Name of the asteroid.
        n (int): Number of samples. Defaults to 1000.
        safety_coefficient (float, optional): Safety coefficient for the trajectory simulation. Defaults to 1.4.
        exit_radius (float, optional): Exit radius for the trajectory simulation. Defaults to 2.0.
        starting_orb_params (tuple, optional): Starting orbital parameters.
            Defaults to (1.5, 0.0, np.pi / 2, 0.0, 0.0, np.pi / 2).
    Returns:
        Path: Path to the saved dataset.
    """
    asteroid_name = asteroid if isinstance(asteroid, str) else asteroid.stem
    ground_truth = MasconModel(asteroid, device="cpu")
    assert isinstance(
        starting_orb_params, tuple
    ), "starting_orb_params must be a tuple of tuples"
    dataset_name = (
        f"{asteroid_name}_traj_{len(starting_orb_params)}_{n}_{propagation_days}.pt"
    )
    dataset_path = TRAIN_DATASETS_DIR / dataset_name
    positions = []
    for params in starting_orb_params:
        body_frame_traj, _ = simulate_trajectory(
            asteroid_name.split("_")[0],
            ground_truth.coords.numpy(),
            ground_truth.masses.numpy(),
            safety_coefficient,
            exit_radius,
            params,
            n,
            propagation_days,
        )
        positions.append(torch.tensor(body_frame_traj[:, :3], dtype=torch.float32))
    positions = torch.stack(positions, dim=0)
    positions = positions.permute(1, 0, 2).reshape(-1, 3)
    labels = compute_acceleration(positions, ground_truth.coords, ground_truth.masses)
    data = torch.cat([positions, labels], dim=1)
    torch.save(data, dataset_path)
    return dataset_path
