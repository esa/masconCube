import torch


def spherical_5d_encoding(x: torch.Tensor, r_threshold: float = 1.0) -> torch.Tensor:
    """5D spherical encoding used by PINN-GM III:
    x = [x, y, z] is encoded as [r_i, r_e, s, t, u]
    """
    r = torch.norm(x, dim=1)
    s = torch.div(x[:, 0], r)
    t = torch.div(x[:, 1], r)
    u = torch.div(x[:, 2], r)
    r_i = torch.where(r < r_threshold, r, torch.ones_like(r))
    r_e = torch.where(r >= r_threshold, 1 / r, torch.ones_like(r))
    return torch.cat(
        (r_i.view(-1, 1), r_e.view(-1, 1), s.view(-1, 1), t.view(-1, 1), u.view(-1, 1)),
        dim=1,
    )


def pinn_gm_loss(predicted, labels):
    """Computes the loss as described in pinn_gm paper
    Args:
        predicted (torch.tensor): model predictions
        labels (torch.tensor): ground truth labels
    Returns:
        [torch.tensor]: loss
    """
    # L1 loss
    l1_loss = torch.norm(predicted - labels, dim=1)
    labels_norm = torch.norm(labels, dim=1)
    return (l1_loss + l1_loss / labels_norm).mean()


def compute_density(coords: torch.Tensor, potential: torch.Tensor, G: float = 1):
    """
    Compute the mass density (rho) from the gravitational potential (Φ) using Poisson's equation.

    Parameters:
    coords (torch.Tensor): Tensor of shape (N, 3) with spatial coordinates (x, y, z).
    potential (torch.Tensor): Tensor of shape (N, 1) with the gravitational potential Φ.
    G (float): Gravitational constant (default is 1 m^3 kg^-1 s^-2).

    Returns:
    torch.Tensor: Tensor of shape (N, 1) with the computed density rho.
    """
    # First derivatives (gradient of Phi)
    grad_potential = torch.autograd.grad(
        outputs=potential,
        inputs=coords,
        grad_outputs=torch.ones_like(potential),
        create_graph=True,
        # retain_graph=True
    )[0]  # Shape: (N, 3)

    # Second derivatives (Laplacian: trace of Hessian matrix)
    laplacian = 0
    for i in range(coords.shape[1]):  # Loop over x, y, z dimensions
        second_derivative = torch.autograd.grad(
            outputs=grad_potential[:, i],
            inputs=coords,
            grad_outputs=torch.ones_like(grad_potential[:, i]),
            create_graph=True,
            # retain_graph=True
        )[0][:, i]  # Take the diagonal elements
        laplacian += second_derivative  # Sum of second partial derivatives

    # Compute density using Poisson's equation
    density = laplacian / (4 * torch.pi * G)
    return density
