import os
from argparse import ArgumentParser

import torch

from mascon_cube.constants import OUTPUT_DIR, VAL_DATASETS_DIR
from mascon_cube.logs import LogConfig
from mascon_cube.models import MasconCube
from mascon_cube.training import CubeTrainingConfig, ValidationConfig, training_loop


def train(asteroid: str, use_tensorboard: bool) -> MasconCube:
    config = CubeTrainingConfig(asteroid=asteroid)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    val_dataset = torch.load(
        VAL_DATASETS_DIR / f"{asteroid.split('_')[0]}_lp_1000_spherical_0_2.pt"
    ).to(device)
    val_config = ValidationConfig(val_dataset=val_dataset, val_every_n_epochs=50)
    log_config = LogConfig() if use_tensorboard else None

    # Train the cube
    return training_loop(config, val_config, log_config, device=device)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("asteroid", type=str)
    parser.add_argument("--tensorboard", "-t", action="store_true")
    args = parser.parse_args()

    cube = train(args.asteroid, args.tensorboard)

    if not args.tensorboard:
        # If not using tensorboard, save the output in the default directory
        output_path = OUTPUT_DIR / "mascon_cube"
        model_path = output_path / args.asteroid / "model.pt"
        os.makedirs(model_path.parent, exist_ok=True)
        torch.save(cube, model_path)
