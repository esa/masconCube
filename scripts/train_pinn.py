import os
from argparse import ArgumentParser
from collections import deque

import numpy as np
import torch

from mascon_cube import geodesynet
from mascon_cube.constants import GROUND_TRUTH_DIR, OUTPUT_DIR
from mascon_cube.data.mascon_model import MasconModel
from mascon_cube.pinn_gm import PinnGM, pinn_gm_loss


def train_pinn(asteroid: str) -> PinnGM:
    device = os.environ["TORCH_DEVICE"]
    torch.manual_seed(42)
    np.random.seed(42)

    mascon_model = MasconModel(asteroid, device=device)

    net = PinnGM(hidden_features=32, hidden_layers=8).to(device)

    # When a new network is created we init empty training logs
    loss_log = []
    weighted_average_log = []
    n_inferences = []
    # .. and we init a loss trend indicators
    weighted_average = deque([], maxlen=20)
    batch_size = 1000
    n_epochs = 10000
    loss_fn = pinn_gm_loss
    targets_point_sampler = geodesynet.get_target_point_sampler(
        batch_size,
        limit_shape_to_asteroid=GROUND_TRUTH_DIR / asteroid / "mesh.pk",
        method="spherical",
        bounds=[0.0, 1.0],
    )
    # Here we set the optimizer
    learning_rate = 2**-8
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=1500, min_lr=1e-6, threshold=0.001
    )
    # And init the best results
    best_loss = np.inf
    best_model_state_dict = net.state_dict()
    mascon_points = mascon_model.coords
    mascon_masses = mascon_model.masses

    # The main training loop
    for i in range(n_epochs):
        # Each ten epochs we resample the target points
        if i % 10 == 0:
            target_points = targets_point_sampler()
            # We compute the labels whenever the target points are changed
            labels = geodesynet.ACC_L(target_points, mascon_points, mascon_masses)

        # Require grad
        target_points = target_points.clone().detach().requires_grad_(True)
        # We compute the values predicted by the neural density field
        predicted = net(target_points)

        # scale proxy potential into true potential, see section 3.3 of PINN paper
        r = torch.norm(target_points, dim=1).view(-1, 1)
        n = torch.where(r > 1, r, torch.ones_like(r))
        predicted = predicted / n
        # Optional: Fuse with weighted low-fidelity potential, see section 3.5 of PINN paper
        # TODO: Implement this
        pass
        # enforce boundary conditions, see section 3.4 of PINN paper
        k = 0.5
        r_ref = 3
        w_bc = (1 + torch.tanh(k * (r - r_ref))) / 2
        w_nn = 1 - w_bc
        u_bc = 1 / r  # u_bc = mu / r.    mu = M * G = 1 assuming G = 1 and M = 1
        u_nn = predicted
        predicted = w_bc * u_bc + w_nn * u_nn
        # We differentiate the predicted values to get the acceleration
        predicted = torch.autograd.grad(
            predicted,
            target_points,
            grad_outputs=torch.ones_like(predicted),
            create_graph=True,
        )[0]
        # We compute the loss (note that the contrastive loss needs a different shape for the labels)
        loss = loss_fn(predicted, labels)

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
        n_inferences.append((batch_size) // 1000)  # counted in thousands

        # Print every i iterations
        if i % 25 == 0:
            wa_out = np.mean(weighted_average)
            print(f"It={i}\t loss={loss.item():.3e}\t  weighted_average={wa_out:.3e}\t")

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
    return net


if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    parser = ArgumentParser()
    parser.add_argument(
        "asteroid",
        type=str,
        help="The name of the asteroid to train the model on.",
    )
    args = parser.parse_args()
    net = train_pinn(args.asteroid)
    # Save the model
    asteroid = args.asteroid
    output_path = OUTPUT_DIR / "pinn_gm"
    model_path = output_path / asteroid / "model.pt"
    os.makedirs(model_path.parent, exist_ok=True)
    torch.save(net.state_dict(), model_path)
