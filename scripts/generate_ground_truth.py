import numpy as np
from tqdm import tqdm

from mascon_cube.constants import DENSITY_VMAX
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
        "vmax": DENSITY_VMAX["planetesimal"],
    },
    "planetesimal_decentered": {
        "mesh": "planetesimal_lp",
        "mask_generator": lambda x: ((x[:, 0] - 0.25) * 2) ** 2
        + x[:, 1] ** 2
        + x[:, 2] ** 2
        < 0.2,
        "mask_scalar": 0,
        "vmax": DENSITY_VMAX["planetesimal_decentered"],
    },
    "planetesimal_uniform": {
        "mesh": "planetesimal_lp",
        "lambda": lambda x: 1.0,
        "vmax": DENSITY_VMAX["planetesimal_uniform"],
    },
    "itokawa_cos": {
        "mesh": "itokawa_lp",
        "lambda": lambda x: np.cos(1.5 * x[0]),
        "vmax": DENSITY_VMAX["itokawa_cos"],
    },
    "itokawa": {
        "mesh": "itokawa_lp",
        "mask_generator": lambda x: x[:, 0] - 0.5 * x[:, 2] > 0.4,
        "mask_scalar": 2850 / 1750,
        "vmax": DENSITY_VMAX["itokawa"],
    },
    "bennu": {
        "mesh": "bennu_lp",
        "mask_generator": lambda x: np.bitwise_or(x[:, 2] > 0.3, x[:, 2] < -0.3),
        "mask_scalar": 3,
        "vmax": DENSITY_VMAX["bennu"],
    },
    "eros_uniform": {
        "mesh": "eros_lp",
        "lambda": lambda x: 1.0,
        "vmax": DENSITY_VMAX["eros_uniform"],
    },
    "eros_2": {
        "mesh": "eros_lp",
        "mask_generator": lambda x: x[:, 1] < -0.1,
        "mask_scalar": 1.5,
        "vmax": DENSITY_VMAX["eros_2"],
    },
    "eros_3": {
        "mesh": "eros_lp",
        "mask_generator": [
            lambda x: x[:, 0] < -0.3,
            lambda x: x[:, 0] - 0.5 * x[:, 2] > 0.4,
        ],
        "mask_scalar": [1.5, 0.5],
        "vmax": DENSITY_VMAX["eros_3"],
    },
}


pbar = tqdm(
    total=len(ground_truth_configs), desc="Generating mascon models", unit="model"
)
for asteroid, config in ground_truth_configs.items():
    mesh = config["mesh"]
    vmax = config.get("vmax", None)
    if "spot_frequency" in config:
        spot_frequency = config["spot_frequency"]
        mesh_to_gt_random_spots(
            mesh, frequency=spot_frequency, seed=42, output_name=asteroid, vmax=vmax
        )
    elif "lambda" in config:
        lambda_function = config["lambda"]
        mesh_to_gt_function(mesh, lambda_function, output_name=asteroid, vmax=vmax)
    else:
        mask_generator = config["mask_generator"]
        mask_scalar = config["mask_scalar"]
        mesh_to_gt(mesh, mask_generator, mask_scalar, output_name=asteroid, vmax=vmax)
    pbar.update(1)
