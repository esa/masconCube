import torch


def compute_acceleration(
    target_points: torch.Tensor,
    mascon_points: torch.Tensor,
    mascon_masses: torch.Tensor,
):
    """
    Computes the acceleration due to the mascon at the target points. (to be used as Label in the training)

    Args:
        target_points (2-D array-like): an (N, 3) array-like object containing the coordinates of the points where the
            acceleration should be computed.
        mascon_points (2-D array-like): an (N, 3) array-like object containing the points that belong to the mascon
        mascon_masses (1-D array-like): a (N,) array-like object containing the values for the mascon masses.
            Can also be a scalar containing the mass value for all points.

    Returns:
        1-D array-like: a (N, 3) torch tensor containing the acceleration (G=1) at the target points
    """
    device = target_points.device
    mm = mascon_masses.view(-1, 1)
    retval = torch.empty(len(target_points), 3, device=device)
    for i, target_point in enumerate(target_points):
        dr = torch.sub(mascon_points, target_point)
        retval[i] = torch.sum(
            mm / torch.pow(torch.norm(dr, dim=1), 3).view(-1, 1) * dr, dim=0
        )
    return retval
