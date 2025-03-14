import math
from copy import deepcopy

import numpy as np
import scipy
import torch


def mascon2stokes(
    mascon_points: torch.Tensor,
    mascon_masses: torch.Tensor,
    r0: float,
    degree: int,
    order: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Computes the stokes coefficients out of a mascon model or a mascon cube

    Args:
        mascon_points (torch.Tensor): cartesian positions of the mascon points
        mascon_masses (torch.Tensor): masses of the mascons
        r0 (float): characteristic radius (often the mean equatorial) of the body
        degree (int): the degree of the spherical harmonics
        order (int): the order of the spherical harmonics

    Returns:
        tuple[torch.Tensor, torch.Tensor]: the stokes coefficients C and S
    """
    mascon_points = mascon_points.cpu()
    mascon_masses = mascon_masses.cpu()
    stokesS = 0
    stokesC = 0
    for point, mass in zip(mascon_points, mascon_masses):
        tmpC, tmpS = _single_mascon_contribution(point, mass, r0, degree, order)
        stokesS += tmpS
        stokesC += tmpC
    return (stokesC.T, stokesS.T)


def _single_mascon_contribution(
    mascon_point: torch.Tensor,
    mascon_mass: torch.Tensor,
    r0: float,
    degree: int,
    order: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    x, y, z = mascon_point
    r, theta, phi = cart2spherical(x, y, z)
    stokesC = mascon_mass
    stokesC *= scipy.special.lpmn(order, degree, np.cos(theta))[0]
    stokesS = deepcopy(stokesC)

    for _order in range(order + 1):
        for _degree in range(degree + 1):
            if _degree < _order:
                continue
            if _order == 0:
                delta = 1
            else:
                delta = 0
            coeff1 = (r / r0) ** _degree
            coeff2C = np.cos(_order * phi)
            coeff2S = np.sin(_order * phi)
            coeff3 = (
                (2 - delta)
                * math.factorial(_degree - _order)
                / math.factorial(_degree + _order)
            )
            normalized = np.sqrt(
                math.factorial(_degree + _order)
                / (2 - delta)
                / (2 * _degree + 1)
                / math.factorial(_degree - _order)
            )
            stokesS[_order, _degree] *= coeff1 * coeff2S * coeff3 * normalized
            stokesC[_order, _degree] *= coeff1 * coeff2C * coeff3 * normalized
    return (stokesC, stokesS)


def cart2spherical(x: float, y: float, z: float) -> tuple[float, float, float]:
    """Converts Cartesian to spherical coordinates defined as
     - r (radius)
     - phi (longitude, in [0, 2pi])
     - theta (colatitude in [0, pi])

    Args:
        x (float): x Cartesian coordinate
        y (float): y Cartesian coordinate
        z (float): z Cartesian coordinate

    Returns:
        tuple: the spherical coordinates r, theta, phi
    """
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)
    if phi < 0:
        phi = phi + 2 * np.pi
    return r, theta, phi
