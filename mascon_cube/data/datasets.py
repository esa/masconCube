from pathlib import Path
from typing import Union

import torch
from torch.utils.data import Dataset

from mascon_cube.constants import TRAIN_DATASETS_DIR
from mascon_cube.data.mascon_model import MasconModel
from mascon_cube.data.sampling import get_target_point_sampler
from mascon_cube.utils import compute_acceleration


class RandomDataset(Dataset):
    def __init__(
        self,
        n: int,
        asteroid: Union[str, Path],
        sampling_method: str = "spherical",
        sampling_min: float = 0.0,
        sampling_max: float = 1.0,
        seed: int = 42,
        cache: bool = True,
    ):
        if cache:
            asteroid_name = asteroid if isinstance(asteroid, str) else asteroid.stem
            dataset_name = f"{asteroid_name}_{n}_{sampling_method}_{sampling_min}_{sampling_max}_{seed}.pt"
            dataset_path = TRAIN_DATASETS_DIR / dataset_name
            if dataset_path.exists():
                self.data = torch.load(dataset_path)
                self.n = len(self.data)
            else:
                self.__sample_data(
                    n, asteroid, sampling_method, sampling_min, sampling_max, seed
                )
                torch.save(self.data, dataset_path)
        else:
            self.__sample_data(
                n, asteroid, sampling_method, sampling_min, sampling_max, seed
            )

    def __sample_data(
        self, n, asteroid, sampling_method, sampling_min, sampling_max, seed
    ):
        torch.manual_seed(seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        sampler = get_target_point_sampler(
            n, asteroid, sampling_method, [sampling_min, sampling_max], device=device
        )
        data = sampler()
        ground_truth = MasconModel(asteroid, device=device)
        labels = compute_acceleration(data, ground_truth.coords, ground_truth.masses)
        self.data = torch.cat([data.to("cpu"), labels.to("cpu")], dim=1)
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return self.data[idx][:3], self.data[idx][3:]


class TrajectoryDataset(Dataset):
    def __init__(self, trajectory_data: torch.Tensor):
        self.data = trajectory_data
        self.n = len(self.data)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return self.data[idx]
