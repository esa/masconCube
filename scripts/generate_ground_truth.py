import numpy as np
from tqdm import tqdm

from mascon_cube.data.mesh import (
    mesh_to_gt,
    mesh_to_gt_function,
    mesh_to_gt_random_spots,
)

ground_truth_configs = {
    "planetesimal": {
        "mesh": "planetesimal_lp",
        "mask_generator": lambda x: x[:, 0] ** 2 + x[:, 1] ** 2 + x[:, 2] ** 2 < 0.2,
        "mask_scalar": 0,
    },
    "planetesimal_decentered": {
        "mesh": "planetesimal_lp",
        "mask_generator": lambda x: ((x[:, 0] - 0.25) * 2) ** 2
        + x[:, 1] ** 2
        + x[:, 2] ** 2
        < 0.2,
        "mask_scalar": 0,
    },
    "planetesimal_uniform": {
        "mesh": "planetesimal_lp",
        "lambda": lambda x: 1.0,
    },
    "itokawa_cos": {
        "mesh": "itokawa_lp",
        "lambda": lambda x: np.cos(2 * x[0]),
    },
    "itokawa": {
        "mesh": "itokawa_lp",
        "mask_generator": lambda x: x[:, 0] - 0.5 * x[:, 2] > 0.4,
        "mask_scalar": 2850 / 1750,
    },
    "bennu": {
        "mesh": "bennu_lp",
        "mask_generator": lambda x: np.bitwise_or(x[:, 2] > 0.3, x[:, 2] < -0.3),
        "mask_scalar": 3,
    },
    "eros_uniform": {
        "mesh": "eros_lp",
        "lambda": lambda x: 1.0,
    },
    "eros_2": {
        "mesh": "eros_lp",
        "mask_generator": lambda x: x[:, 1] < -0.05,
        "mask_scalar": 1.5,
    },
    "eros_3": {
        "mesh": "eros_lp",
        "mask_generator": [
            lambda x: x[:, 0] < -0.3,
            lambda x: x[:, 0] - 0.5 * x[:, 2] > 0.4,
        ],
        "mask_scalar": [2.5, 0.2],
    },
}


pbar = tqdm(
    total=len(ground_truth_configs), desc="Generating mascon models", unit="model"
)
for asteroid, config in ground_truth_configs.items():
    mesh = config["mesh"]
    if "spot_frequency" in config:
        spot_frequency = config["spot_frequency"]
        mesh_to_gt_random_spots(
            mesh, frequency=spot_frequency, seed=42, output_name=asteroid
        )
    elif "lambda" in config:
        lambda_function = config["lambda"]
        mesh_to_gt_function(mesh, lambda_function, output_name=asteroid)
    else:
        mask_generator = config["mask_generator"]
        mask_scalar = config["mask_scalar"]
        mesh_to_gt(mesh, mask_generator, mask_scalar, output_name=asteroid)
    pbar.update(1)
