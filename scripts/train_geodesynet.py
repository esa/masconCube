import os
from argparse import ArgumentParser
from collections import deque

import numpy as np
import torch
from torch import nn

from mascon_cube import geodesynet
from mascon_cube.constants import GROUND_TRUTH_DIR, OUTPUT_DIR
from mascon_cube.data.mascon_model import MasconModel


def train_geodesynet(asteroid: str):
    geodesynet.enableCUDA(device=0)
    device = os.environ["TORCH_DEVICE"]
    torch.manual_seed(42)
    np.random.seed(42)

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
    n_epochs = 10000
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
        optimizer, factor=0.8, patience=200, min_lr=1e-6
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

    # Here we restore the learned parameters of the best model of the run
    for layer in net.state_dict():
        net.state_dict()[layer] = best_model_state_dict[layer]
    return net, best_c


if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    parser = ArgumentParser()
    parser.add_argument(
        "asteroid",
        type=str,
        help="The name of the asteroid to train the model on.",
    )
    args = parser.parse_args()
    net, best_c = train_geodesynet(args.asteroid)
    # Save the model
    asteroid = args.asteroid
    output_path = OUTPUT_DIR / "geodesynet"
    model_path = output_path / asteroid / "model.pt"
    os.makedirs(model_path.parent, exist_ok=True)
    torch.save(net.state_dict(), model_path)
    # Save the best c
    c_path = output_path / asteroid / "c.txt"
    with open(c_path, "w") as f:
        f.write(str(best_c.item()))
