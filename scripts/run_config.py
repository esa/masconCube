from argparse import ArgumentParser
from pathlib import Path

import yaml

from mascon_cube.constants import CONFIGS_DIR, TRAIN_DATASETS_DIR
from mascon_cube.training import CubeTrainingConfig


def load_config(config_path: Path) -> list[CubeTrainingConfig]:
    """Load a YAML configuration file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        dict: Configuration as a dictionary.
    """
    configs = []
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    for ast, ds in config.asteroid.items():
        configs.append(
            CubeTrainingConfig(
                asteroid=ast,
                train_set_path=TRAIN_DATASETS_DIR / ds["train"],
                **{k: v for k, v in config["train"].items()},
            )
        )
    return configs


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "config-name",
        type=str,
        help="Name of the YAML configuration file. It must be located in mascon_cube/data/train_configs",
    )
    args = parser.parse_args()
    config_name = (
        args.config_name
        if args.config_name.endswith(".yaml")
        else f"{args.config_name}.yaml"
    )
    config_path = CONFIGS_DIR / config_name
    configs = load_config(config_path)
    for cfg in configs:
        pass
    # TODO
