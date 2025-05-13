import math

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
    """Computes the Stokes coefficients from a mascon model or cube.

    Args:
        mascon_points (torch.Tensor): Cartesian positions of the mascon points.
        mascon_masses (torch.Tensor): Masses of the mascons.
        r0 (float): Characteristic radius (often mean equatorial radius) of the body.
        degree (int): Degree of spherical harmonics.
        order (int): Order of spherical harmonics.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Stokes coefficients C and S.
    """
    # Move data to CPU for compatibility with SciPy functions
    mascon_points = mascon_points.detach().cpu().numpy()
    mascon_masses = mascon_masses.detach().cpu().numpy()

    # Preallocate Stokes coefficients
    stokesC = np.zeros((order + 1, degree + 1))
    stokesS = np.zeros((order + 1, degree + 1))

    # Compute contributions for all mascons
    for point, mass in zip(mascon_points, mascon_masses):
        tmpC, tmpS = _single_mascon_contribution(point, mass, r0, degree, order)
        stokesC += tmpC
        stokesS += tmpS

    # Convert results back to tensors
    return torch.tensor(stokesC.T), torch.tensor(stokesS.T)


def _single_mascon_contribution(
    mascon_point: np.ndarray,
    mascon_mass: float,
    r0: float,
    degree: int,
    order: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Computes the contribution to Stokes coefficients from a single mascon."""
    x, y, z = mascon_point
    r, theta, phi = cart2spherical(x, y, z)

    # Precompute Legendre polynomials for efficiency
    legendre_polynomials = scipy.special.lpmn(order, degree, np.cos(theta))[0]

    # Initialize Stokes coefficients
    stokesC = np.zeros((order + 1, degree + 1))
    stokesS = np.zeros((order + 1, degree + 1))

    for _order in range(order + 1):
        for _degree in range(_order, degree + 1):
            delta = 1 if _order == 0 else 0

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

            # Update Stokes coefficients
            stokesC[_order, _degree] += (
                mascon_mass
                * legendre_polynomials[_order, _degree]
                * coeff1
                * coeff2C
                * coeff3
                * normalized
            )
            stokesS[_order, _degree] += (
                mascon_mass
                * legendre_polynomials[_order, _degree]
                * coeff1
                * coeff2S
                * coeff3
                * normalized
            )

    return stokesC, stokesS


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

    # Ensure phi is in [0, 2π]
    if phi < 0:
        phi += 2 * np.pi

    return r, theta, phi


def combine_stokes_coefficients(
    stokesC: np.ndarray,
    stokesS: np.ndarray,
) -> np.ndarray:
    """Combines Stokes coefficients into a single array.

    Args:
        stokesC (np.ndarray): Stokes C coefficients.
        stokesS (np.ndarray): Stokes S coefficients.

    Returns:
        np.ndarray: Combined Stokes coefficients.
    """
    if isinstance(stokesC, torch.Tensor):
        stokesC = stokesC.cpu().numpy()
    if isinstance(stokesS, torch.Tensor):
        stokesS = stokesS.cpu().numpy()
    stokesS = np.concatenate((stokesS.T[1:], np.zeros((1, len(stokesS)))))
    return stokesC + stokesS
