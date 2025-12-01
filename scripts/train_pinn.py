import os
from argparse import ArgumentParser

import torch

from mascon_cube.constants import OUTPUT_DIR, VAL_DATASETS_DIR
from mascon_cube.pinn_gm import PinnGM, PinnTrainingConfig, training_loop


def train(asteroid: str, progressbar: bool = False) -> PinnGM:
    config = PinnTrainingConfig(
        asteroid=asteroid,
        val_set_path=VAL_DATASETS_DIR / f"{asteroid}_1000_spherical_0_2.pt",
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Train the cube
    return training_loop(config, device=device, progressbar=progressbar)


if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    parser = ArgumentParser()
    parser.add_argument("asteroid", type=str)
    parser.add_argument(
        "--progressbar",
        "-p",
        action="store_true",
        help="Show progress bar during training",
    )
    args = parser.parse_args()

    pinn = train(args.asteroid, progressbar=args.progressbar)
    output_path = OUTPUT_DIR / "pinn_gm"
    model_path = output_path / args.asteroid / "model.pt"
    os.makedirs(model_path.parent, exist_ok=True)
    torch.save(pinn, model_path)
