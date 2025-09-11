"""Generate internal structure plots for all asteroids with trained models."""

import gc

import torch
from matplotlib import pyplot as plt
from torch import nn

from mascon_cube import geodesynet
from mascon_cube.constants import DENSITY_VMAX, MASS_VMAX, OUTPUT_DIR
from mascon_cube.data.mascon_model import MasconModel
from mascon_cube.visualization import plot_mascon_cube

mascon_cube_folder = OUTPUT_DIR / "mascon_cube"
for asteroid in mascon_cube_folder.iterdir():
    if asteroid.is_dir() and (asteroid / "model.pt").exists():
        cube = torch.load(asteroid / "model.pt", weights_only=False)
        fig = plot_mascon_cube(cube, range=[0, MASS_VMAX[asteroid.name]])
        plt.savefig(asteroid / "mascon_cube.png")
        plt.close()

del cube
geodesynet.enableCUDA()
geodesynet.fixRandomSeeds()
geodesynet_folder = OUTPUT_DIR / "geodesynet"
for asteroid in geodesynet_folder.iterdir():
    if (
        asteroid.is_dir()
        and (asteroid / "model.pt").exists()
        and (asteroid / "c.txt").exists()
    ):
        torch.cuda.empty_cache()
        gc.collect()
        encoding = geodesynet.direct_encoding()
        net = geodesynet.init_network(
            encoding, n_neurons=100, model_type="siren", activation=nn.Tanh()
        ).to("cuda:0")
        net.load_state_dict(torch.load(asteroid / "model.pt"))
        net = net.to("cuda:0")
        uniform_model = MasconModel(asteroid.name, device="cpu", uniform=True)
        uniform_density = uniform_model.get_average_density()
        with open(asteroid / "c.txt", "r") as f:
            c = float(f.read())
        fig = geodesynet.plot_model_vs_mascon_contours(
            net,
            encoding,
            c=c,
            progressbar=False,
            crop_p=1e-8,
            N=10000,
            heatmap=True,
            add_shape_base_value=f"/home/pietrofanti/code/masconCube/data/3dmeshes/{asteroid.name.split('_')[0]}_lp.pk",
            add_const_density=uniform_density,
            range=[0, DENSITY_VMAX[asteroid.name]],
        )
        plt.savefig(asteroid / "mascon_contours.png")
        plt.close()
