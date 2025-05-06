import gc
import os
from collections import deque

import numpy as np
import torch
import torchquad as tquad
from matplotlib import pyplot as plt
from torch import nn
from tqdm import tqdm

from mascon_cube import geodesynet
from mascon_cube.constants import GROUND_TRUTH_DIR, OUTPUT_DIR, VAL_DATASETS_DIR
from mascon_cube.data.mascon_model import MasconModel
from mascon_cube.data.mesh import get_mesh, unpack_triangle_mesh
from mascon_cube.data.sampling import is_outside_torch
from mascon_cube.data.stokes import mascon2stokes
from mascon_cube.training import CubeTrainingConfig, ValidationConfig, training_loop
from mascon_cube.visualization import plot_mascon_cube, stokes_heatmap


def train_geodesynet(asteroid: str):
    mascon_model = MasconModel(asteroid, device=device)
    uniform_model = MasconModel(asteroid, device=device, uniform=True)

    encoding = geodesynet.direct_encoding()
    net = geodesynet.init_network(
        encoding, n_neurons=100, model_type="siren", activation=nn.Tanh()
    ).to("cuda:0")
    # When a new network is created we init empty training logs
    loss_log = []
    weighted_average_log = []
    n_inferences = []
    # .. and we init a loss trend indicators
    weighted_average = deque([], maxlen=20)
    n_quadrature = 300000
    batch_size = 1000
    n_epochs = 20000
    loss_fn = geodesynet.normalized_L1_loss
    mc_method = geodesynet.ACC_trap
    targets_point_sampler = geodesynet.get_target_point_sampler(
        batch_size,
        limit_shape_to_asteroid=GROUND_TRUTH_DIR / asteroid / "mesh.pk",
        method="spherical",
        bounds=[0.0, 1.0],
    )
    # Here we set the optimizer
    learning_rate = 1e-6
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.8, patience=200, min_lr=1e-6, verbose=True
    )
    # And init the best results
    best_loss = np.inf
    best_model_state_dict = net.state_dict()
    mascon_points = mascon_model.coords
    mascon_masses_nu = mascon_model.masses
    mascon_masses_u = uniform_model.masses

    # The main training loop
    for i in range(n_epochs):
        # Each ten epochs we resample the target points
        if i % 10 == 0:
            target_points = targets_point_sampler()
            # We compute the labels whenever the target points are changed
            # These are the difference between a homogeneous and an inhomogenous ground truth
            labels_u = geodesynet.ACC_L(target_points, mascon_points, mascon_masses_u)
            labels_nu = geodesynet.ACC_L(target_points, mascon_points, mascon_masses_nu)
            labels = labels_nu - labels_u

        # We compute the values predicted by the neural density field
        predicted = mc_method(target_points, net, encoding, N=n_quadrature)

        # We learn the scaling constant (k in the paper)
        c = torch.sum(predicted * labels) / torch.sum(predicted * predicted)

        # We compute the loss (note that the contrastive loss needs a different shape for the labels)
        if loss_fn == geodesynet.contrastive_loss:
            loss = loss_fn(predicted, labels)
        else:
            loss = loss_fn(predicted.view(-1), labels.view(-1))

        # We store the model if it has the lowest fitness
        # (this is to avoid losing good results during a run that goes wild)
        if loss < best_loss:
            best_model_state_dict = net.state_dict()
            best_loss = loss
            best_c = c

        # Update the loss trend indicators
        weighted_average.append(loss.item())

        # Update the logs
        weighted_average_log.append(np.mean(weighted_average))
        loss_log.append(loss.item())
        n_inferences.append((n_quadrature * batch_size) // 100000)

        # Zeroes the gradient (necessary because of things)
        optimizer.zero_grad()

        # Backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()

        # Calling the step function on an Optimizer makes an update to its
        # parameters
        optimizer.step()

        # Perform a step in LR scheduler to update LR
        scheduler.step(loss.item())

    # Here we restore the learned parameters of the best model of the run
    for layer in net.state_dict():
        net.state_dict()[layer] = best_model_state_dict[layer]
    return net, best_c


asteroids = [f.name for f in os.scandir(GROUND_TRUTH_DIR) if f.is_dir()]
geodesynet.enableCUDA(device=0)
device = os.environ["TORCH_DEVICE"]
torch.manual_seed(42)
np.random.seed(42)
output_dir = OUTPUT_DIR / "stokes_coefficients_comparison"
output_dir.mkdir(parents=True, exist_ok=True)

results = {}

pbar = tqdm(total=len(asteroids) * 2, unit="training")

for asteroid in asteroids:
    pbar.set_description(f"Training mascon cube on {asteroid}")
    # Load the mascon model
    # Define the training configuration
    config = CubeTrainingConfig(
        asteroid=asteroid, cube_side=100, n_epochs=2000, n_epochs_before_resampling=10
    )
    # Define the validation configuration
    val_dataset = torch.load(VAL_DATASETS_DIR / f"{asteroid}_1000_spherical_0_2.pt").to(
        device
    )
    val_config = ValidationConfig(val_dataset=val_dataset, val_every_n_epochs=50)
    # Run the training loop
    cube = training_loop(config, val_config, device=device)
    _output_dir = output_dir / asteroid
    _output_dir.mkdir(parents=True, exist_ok=True)
    # Save the trained cube
    torch.save(cube, _output_dir / "trained_cube.pt")
    fig = plot_mascon_cube(cube)
    plt.savefig(_output_dir / "trained_cube.png")
    # Save the mascon model
    mascon_model = MasconModel(asteroid, device="cpu")
    fig = plot_mascon_cube(mascon_model.to_cube(100))
    plt.savefig(_output_dir / "mascon_model.png")
    # Compare the Stokes coefficients
    gt_stokes = mascon2stokes(
        mascon_model.coords.to(device),
        mascon_model.masses.to(device),
        r0=1,
        degree=7,
        order=7,
    )
    cube_stokes = mascon2stokes(
        cube.coords.detach(), cube.masses.detach(), r0=1, degree=7, order=7
    )
    fig = stokes_heatmap(
        cube_stokes[0].cpu(),
        gt_stokes[0].cpu(),
        "Cosine stokes coefficients relative error",
    )
    plt.savefig(_output_dir / "cosine_stokes_coefficients.png")
    fig = stokes_heatmap(
        cube_stokes[1].cpu(),
        gt_stokes[1].cpu(),
        "Sine stokes coefficients relative error",
    )
    plt.savefig(_output_dir / "sine_stokes_coefficients.png")
    pbar.update(1)
    pbar.set_description(f"Training geodesynet on {asteroid}")
    # Train geodesynet
    gc.collect()
    torch.cuda.empty_cache()
    net, c = train_geodesynet(asteroid)
    gc.collect()
    torch.cuda.empty_cache()
    # Save the trained geodesynet
    torch.save(net.state_dict(), _output_dir / "trained_geodesynet.pt")
    fig = geodesynet.plot_model_vs_mascon_contours(
        net,
        lambda x: x,
        mascon_model.coords.to(device),
        mascon_model.masses.to(device),
        c=c,
        progressbar=True,
        N=2500,
        heatmap=True,
        add_shape_base_value=GROUND_TRUTH_DIR / asteroid / "mesh.pk",
        add_const_density=mascon_model.get_average_density(),
    )
    plt.savefig(_output_dir / "trained_geodesynet.png")
    # We construct the vecotrized Legendre associated polynomials
    P = geodesynet.legendre_factory_torch(n=16)
    # Declare an integrator
    quad = tquad.Trapezoid()
    uniform_density = mascon_model.get_average_density()

    mesh_vertices, mesh_triangles = get_mesh(asteroid)
    triangles = unpack_triangle_mesh(mesh_vertices, mesh_triangles, "cuda")

    def mass(x):
        result = net(x) * c
        mask = is_outside_torch(x, triangles)
        result[mask] = 0
        mask = torch.bitwise_not(mask)
        result[mask] += uniform_density
        return result

    gc.collect()
    torch.cuda.empty_cache()
    # Compute the function integral
    M = quad.integrate(
        mass, dim=3, N=300000, integration_domain=[[-1, 1], [-1, 1], [-1, 1]]
    )
    stokesC_gann = np.zeros((8, 8))
    for level in range(8):
        for m in range(8):
            if m > level:
                continue
            stokesC_gann[level][m] = quad.integrate(
                lambda x, level=level, m=m, P=P, net=mass, R0=1: geodesynet.Clm(
                    x, net, level, m, R0, P
                ),
                dim=3,
                N=310000,
                integration_domain=[[-1, 1], [-1, 1], [-1, 1]],
            )
            stokesC_gann[level][m] = (
                stokesC_gann[level][m] / M * geodesynet.constant_factors(level, m)
            )
    stokesS_gann = np.zeros((8, 8))
    for level in range(8):
        for m in range(8):
            if m > level:
                continue
            stokesS_gann[level][m] = quad.integrate(
                lambda x, level=level, m=m, P=P, net=mass, R0=1: geodesynet.Slm(
                    x, net, level, m, R0, P
                ),
                dim=3,
                N=300000,
                integration_domain=[[-1, 1], [-1, 1], [-1, 1]],
            )
            stokesS_gann[level][m] = (
                stokesS_gann[level][m] / M * geodesynet.constant_factors(level, m)
            )
    fig = stokes_heatmap(
        torch.Tensor(stokesC_gann).cpu(),
        gt_stokes[0].cpu(),
        "GeodesyNet - Cosine stokes coefficients L1 distance",
    )
    plt.savefig(_output_dir / "geodesynet_cosine_stokes_coefficients.png")
    fig = stokes_heatmap(
        torch.Tensor(stokesS_gann).cpu(),
        gt_stokes[1].cpu(),
        "GeodesyNet - Sine stokes coefficients L1 distance",
    )
    plt.savefig(_output_dir / "geodesynet_sine_stokes_coefficients.png")
    net = net.cpu()
    pbar.update(1)
    cube_stokes_mean_error = (
        torch.mean(torch.abs(cube_stokes[0] - gt_stokes[0]))
        + torch.mean(torch.abs(cube_stokes[1] - gt_stokes[1]))
    ) / 2
    geodesynet_stokes_mean_error = (
        torch.mean(torch.abs(torch.Tensor(stokesC_gann) - gt_stokes[0]))
        + torch.mean(torch.abs(torch.Tensor(stokesS_gann) - gt_stokes[1]))
    ) / 2
    results[asteroid] = {
        "cube_stokes_mean_error": cube_stokes_mean_error.item(),
        "geodesynet_stokes_mean_error": geodesynet_stokes_mean_error.item(),
    }
pbar.close()

# Save the results
with open(output_dir / "results.txt", "w") as f:
    for asteroid, result in results.items():
        f.write(f"{asteroid}: {result}\n")
# Print the results
for asteroid, result in results.items():
    print(f"{asteroid}: {result}")
