# %% [markdown]
# # Stokes coefficients

# %%
# Ensure that changes in imported module (gravann most importantly) are autoreloaded

import gc
import os
from collections import deque

import numpy as np
import torch
import torchquad as tquad
from matplotlib import pyplot as plt
from torch import nn

from mascon_cube import geodesynet
from mascon_cube.constants import GROUND_TRUTH_DIR, OUTPUT_DIR, VAL_DATASETS_DIR
from mascon_cube.data.mascon_model import MasconModel
from mascon_cube.data.mesh import (
    get_mesh,
    is_outside_torch,
    mesh_to_gt,
    unpack_triangle_mesh,
)
from mascon_cube.data.stokes import mascon2stokes
from mascon_cube.training import CubeTrainingConfig, ValidationConfig, training_loop
from mascon_cube.visualization import plot_mascon_cube, stokes_heatmap

# %%
torch.manual_seed(42)
np.random.seed(42)

# %% [markdown]
# ## Generate and visualize ground-truth

# %%
mask_scalar = 2850 / 1750
asteroid = "itokawa_lp"


def mask_generator(mascon_points):
    return mascon_points[:, 0] - 0.5 * mascon_points[:, 2] > 0.4


mesh_to_gt(asteroid, mask_generator, mask_scalar, save_image=True, save_uniform=True)

# %%
img = plt.imread(GROUND_TRUTH_DIR / f"{asteroid}.png")
plt.imshow(img)
plt.show()

# %% [markdown]
# ## Convert thethraedra model to cube model

# %%
model = MasconModel(asteroid, device="cpu")

# %%
cube_gt = model.to_cube(100)

# %%
plot_mascon_cube(cube_gt, range=(1e-5, 3e-5))
plt.show()

# %%
uniform_model = MasconModel(asteroid, device="cpu", uniform=True)

# %% [markdown]
# ## Compute difference in stokes

# %%
gt_stokes = mascon2stokes(model.coords, model.masses, r0=1, degree=7, order=7)

# %%
cube_gt_stokes = mascon2stokes(cube_gt.coords, cube_gt.masses, r0=1, degree=7, order=7)

# %%
stokes_heatmap(
    cube_gt_stokes[0].cpu(),
    gt_stokes[0].cpu(),
    "Cosine stokes coefficients relative error",
)
plt.show()
stokes_heatmap(
    cube_gt_stokes[1].cpu(),
    gt_stokes[1].cpu(),
    "Sine stokes coefficients relative error",
)
plt.show()

# %% [markdown]
# ## Train a MasconCube

# %%
config = CubeTrainingConfig(
    asteroid="itokawa_lp", cube_side=100, n_epochs=1000, n_epochs_before_resampling=10
)

device = "cuda" if torch.cuda.is_available() else "cpu"

val_dataset = torch.load(VAL_DATASETS_DIR / "itokawa_lp_1000_spherical_0_2.pt").to(
    device
)
val_config = ValidationConfig(val_dataset=val_dataset, val_every_n_epochs=50)

# Train the cube
mascon_cube = training_loop(
    config, val_config, log_config=None, device=device, progressbar=True
)

# %%
plot_mascon_cube(mascon_cube, range=(1e-5, 3e-5))
plt.show()

# %%
mascon_cube_stokes = mascon2stokes(
    mascon_cube.coords.detach(), mascon_cube.masses.detach(), r0=1, degree=7, order=7
)

# %%
stokes_heatmap(
    mascon_cube_stokes[0].cpu(),
    gt_stokes[0].cpu(),
    "MasconCube - Cosine stokes coefficients relative error",
)
plt.show()
stokes_heatmap(
    mascon_cube_stokes[1].cpu(),
    gt_stokes[1].cpu(),
    "MasconCube - Sine stokes coefficients relative error",
)
plt.show()

# %% [markdown]
# ## Train a GeodesyNet

# %%

torch.cuda.empty_cache()
gc.collect()
os.environ["TORCH_DEVICE"] = "cuda:0"
torch.manual_seed(42)
np.random.seed(42)

encoding = geodesynet.direct_encoding()
net = geodesynet.init_network(
    encoding, n_neurons=100, model_type="siren", activation=nn.Tanh()
).to("cuda:0")
# When a new network is created we init empty training logs
loss_log = []
weighted_average_log = []
running_loss_log = []
n_inferences = []
# .. and we init a loss trend indicators
weighted_average = deque([], maxlen=20)

# %%
# EXPERIMENTAL SETUP ------------------------------------------------------------------------------------
# Number of points to be used to evaluate numerically the triple integral
# defining the acceleration.
# Use <=30000 to for a quick training ... 300000 was used to produce most of the paper results
n_quadrature = 300000

# Dimension of the batch size, i.e. number of points
# where the ground truth is compared to the predicted acceleration
# at each training epoch.
# Use 100 for a quick training. 1000  was used to produce most of the paper results
batch_size = 1000

# Loss. The normalized L1 loss (kMAE in the paper) was
# found to be one of the best performing choices.
# More are implemented in the module
loss_fn = geodesynet.normalized_L1_loss

# The numerical Integration method.
# Trapezoidal integration is here set over a dataset containing acceleration values,
# (it is possible to also train on values of the gravity potential, results are similar)
mc_method = geodesynet.ACC_trap

# The sampling method to decide what points to consider in each batch.
# In this case we sample points unifromly in a sphere and reject those that are inside the asteroid
targets_point_sampler = geodesynet.get_target_point_sampler(
    batch_size,
    limit_shape_to_asteroid="/home/pietrofanti/code/masconCube/data/3dmeshes/itokawa_lp.pk",
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

mascon_points = model.coords.to("cuda:0")
mascon_masses_nu = model.masses.to("cuda:0")
mascon_masses_u = uniform_model.masses.to("cuda:0")

# %%
# net.load_state_dict(torch.load(OUTPUT_DIR / f"{asteroid}_geodesynet_state_dict.pt"))
# c = geodesynet.compute_c_for_model(net, encoding, mascon_points, mascon_masses_nu, use_acc = True)
# print(c)


# %%
# TRAINING LOOP (differential training, use of any prior shape information)------------------------
# This cell can be stopped and started again without loosing memory of the training nor its logs
torch.cuda.empty_cache()
# The main training loop
for i in range(20000):
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
        print("New Best: ", loss.item())

    # Update the loss trend indicators
    weighted_average.append(loss.item())

    # Update the logs
    weighted_average_log.append(np.mean(weighted_average))
    loss_log.append(loss.item())
    n_inferences.append((n_quadrature * batch_size) // 100000)

    # Print every i iterations
    if i % 25 == 0:
        wa_out = np.mean(weighted_average)
        print(
            f"It={i}\t loss={loss.item():.3e}\t  weighted_average={wa_out:.3e}\t  c={c:.3e}"
        )

    # Zeroes the gradient (necessary because of things)
    optimizer.zero_grad()

    # Backward pass: compute gradient of the loss with respect to model parameters
    loss.backward()

    # Calling the step function on an Optimizer makes an update to its
    # parameters
    optimizer.step()

    # Perform a step in LR scheduler to update LR
    scheduler.step(loss.item())

# %%
# Here we restore the learned parameters of the best model of the run
for layer in net.state_dict():
    net.state_dict()[layer] = best_model_state_dict[layer]
net = net.to("cuda:0")

# %%
# First lets have a look at the training loss history
plt.figure()
abscissa = np.cumsum(n_inferences)
plt.semilogy(abscissa, loss_log)
plt.semilogy(abscissa, weighted_average_log)
plt.xlabel("Thousands of model evaluations")
plt.ylabel("Loss")
plt.legend(["Loss", "Weighted Average Loss"])

# %%
# Then overlaying a heatmap to the mascons
uniform_density = uniform_model.get_average_density()
torch.set_default_tensor_type(torch.cuda.FloatTensor)
geodesynet.plot_model_vs_mascon_contours(
    net,
    encoding,
    mascon_points,
    mascon_masses_nu,
    c=c,
    progressbar=True,
    N=2500,
    heatmap=True,
    add_shape_base_value="/home/pietrofanti/code/masconCube/data/3dmeshes/itokawa_lp.pk",
    add_const_density=uniform_density,
)

# %%
# save state dict
torch.save(net.state_dict(), OUTPUT_DIR / f"{asteroid}_geodesynet_state_dict.pt")

# %%
# We construct the vecotrized Legendre associated polynomials
P = geodesynet.legendre_factory_torch(n=16)
# Declare an integrator
quad = tquad.Trapezoid()

# %%
uniform_density = uniform_model.get_average_density()

mesh_vertices, mesh_triangles = get_mesh(asteroid)
triangles = unpack_triangle_mesh(mesh_vertices, mesh_triangles, "cuda")


def mass(x):
    result = net(x) * c
    mask = is_outside_torch(x, triangles)
    result[mask] = 0
    mask = torch.bitwise_not(mask)
    result[mask] += uniform_density
    return result


# %%
gc.collect()
torch.cuda.empty_cache()
# Compute the function integral
M = quad.integrate(
    mass, dim=3, N=300000, integration_domain=[[-1, 1], [-1, 1], [-1, 1]]
)
print(M)

# %%
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

# %%
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

# %%
stokes_heatmap(
    torch.Tensor(stokesC_gann).cpu(),
    gt_stokes[0].cpu(),
    "GeodesyNet - Cosine stokes coefficients L1 distance",
)
plt.show()
stokes_heatmap(
    torch.Tensor(stokesS_gann).cpu(),
    gt_stokes[1].cpu(),
    "GeodesyNet - Sine stokes coefficients L1 distance",
)
plt.show()
