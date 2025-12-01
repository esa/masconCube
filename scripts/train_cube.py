import os
from argparse import ArgumentParser
from pathlib import Path

import torch

from mascon_cube.constants import OUTPUT_DIR, TRAIN_DATASETS_DIR, VAL_DATASETS_DIR
from mascon_cube.models import MasconCube
from mascon_cube.training import CubeTrainingConfig, training_loop


def train(
    asteroid: str,
    train_set_path: None | Path = None,
    use_wandb: bool = False,
    progressbar: bool = False,
) -> MasconCube:
    if train_set_path is None:
        train_set_path = (
            TRAIN_DATASETS_DIR / f"{asteroid}_100000_spherical_0.0_1.0_42.pt"
        )
    val_set_path = VAL_DATASETS_DIR / f"{asteroid}_1000_spherical_0_2.pt"
    config = CubeTrainingConfig(
        asteroid=asteroid, train_set_path=train_set_path, val_set_path=val_set_path
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Train the cube
    return training_loop(
        config, device=device, use_wandb=use_wandb, progressbar=progressbar
    )


if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    parser = ArgumentParser()
    parser.add_argument("asteroid", type=str)
    parser.add_argument(
        "--train-set-path",
        "-t",
        type=Path,
        default=None,
        help="Path to the training dataset",
    )
    parser.add_argument(
        "--use-wandb",
        "-w",
        action="store_true",
        help="Use Weights & Biases for logging",
    )
    parser.add_argument(
        "--progressbar",
        "-p",
        action="store_true",
        help="Show progress bar during training",
    )
    args = parser.parse_args()

    cube = train(
        args.asteroid,
        args.train_set_path,
        use_wandb=args.use_wandb,
        progressbar=args.progressbar,
    )
    output_path = OUTPUT_DIR / "mascon_cube"
    model_path = output_path / args.asteroid / "model.pt"
    os.makedirs(model_path.parent, exist_ok=True)
    torch.save(cube, model_path)
