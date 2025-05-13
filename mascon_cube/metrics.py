from typing import Optional, Union

import torch
from torch import nn

from mascon_cube import geodesynet
from mascon_cube.data.mascon_model import MasconModel
from mascon_cube.models import MasconCube


def compute_acceleration(
    model: Union[MasconCube, MasconModel, nn.Module],
    target_points: torch.Tensor,
    c: Optional[float] = None,
    uniform_model: Optional[MasconModel] = None,
    batch_size: int = 1000,
    N_integration: int = 500000,
) -> torch.Tensor:
    if isinstance(model, MasconCube) or isinstance(model, MasconModel):
        mascon_points = model.coords
        mascon_masses = model.masses
        device = target_points.device
        mm = mascon_masses.view(-1, 1)
        retval = torch.empty(len(target_points), 3, device=device)
        for i, target_point in enumerate(target_points):
            dr = torch.sub(mascon_points, target_point)
            retval[i] = torch.sum(
                mm / torch.pow(torch.norm(dr, dim=1), 3).view(-1, 1) * dr, dim=0
            )
        return retval

    elif isinstance(model, nn.Module):
        is_differential = uniform_model is not None
        if is_differential:
            assert (
                c is not None
            ), "A value for `c` must be provided when computing the acceleration for a differential GeodesyNet"
        torch.cuda.empty_cache()
        pred = []
        integration_grid, h, N_int = geodesynet.compute_integration_grid(N_integration)

        for idx in range((len(target_points) // batch_size) + 1):
            indices = list(
                range(idx * batch_size, min((idx + 1) * batch_size, len(target_points)))
            )
            points = target_points[indices]
            prediction = geodesynet.ACC_trap(
                points, model, N=N_int, h=h, sample_points=integration_grid
            ).detach()
            if is_differential:
                prediction = (
                    geodesynet.ACC_L(points, uniform_model.coords, uniform_model.masses)
                    + c * prediction
                )
            pred.append(prediction)
        pred = torch.cat(pred)
        return pred

    else:
        raise TypeError(
            f"`model` must be of type `MasconCube`, `MasconModel` or `nn.Module`, instead {type(model)} has been passed"
        )


def cosine_distance(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    return 1 - nn.functional.cosine_similarity(pred, gt)


def norm_distance(
    pred: torch.Tensor, gt: torch.Tensor, dim: int = 1, eps: float = 1e-8
) -> torch.Tensor:
    return torch.abs(pred.norm(p=2, dim=1) - gt.norm(p=2, dim=1))


def relative_norm_distance(
    pred: torch.Tensor, gt: torch.Tensor, dim: int = 1, eps: float = 1e-8
) -> torch.Tensor:
    return norm_distance(pred, gt) / gt.norm(p=2, dim=1)
