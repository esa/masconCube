# ruff: noqa: E741 : Ignore naming problems from darioizzo/geodesyNets

import numpy as np
import torch
import torchquad as tquad

from .models import GeodesyNet


def geodesynet2stokes(
    net: GeodesyNet,
    n_quadrature: int,
    r0: float,
    degree: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Computes the Stokes coefficients from a GeodesyNet.

    Args:
        net (GeodesyNet): Input GeodesyNet.
        n_quadrature (int): Number of points to use for the evaluation.
        r0 (float): Characteristic radius (often mean equatorial radius) of the body.
        degree (int): Degree and order of spherical harmonics.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Stokes coefficients C and S.
    """

    # Integrand to compute the mass
    def mass(x):
        return net(x)

    # We construct the vecotrized Legendre associated polynomials
    P = legendre_factory_torch(n=degree)
    # Declare an integrator
    quad = tquad.Trapezoid()
    M = quad.integrate(
        mass, dim=3, N=n_quadrature, integration_domain=[[-1, 1], [-1, 1], [-1, 1]]
    )
    stokesC_gann = np.zeros((degree + 1, degree + 1))
    for l in range(degree + 1):
        for m in range(degree + 1):
            if m > l:
                continue
            stokesC_gann[l][m] = quad.integrate(
                lambda x, l=l, m=m, P=P, model=net, R0=r0: Clm(x, model, l, m, R0, P),
                dim=3,
                N=n_quadrature,
                integration_domain=[[-1, 1], [-1, 1], [-1, 1]],
            )
            stokesC_gann[l][m] = stokesC_gann[l][m] / M * constant_factors(l, m)
    stokesS_gann = np.zeros((degree + 1, degree + 1))
    for l in range(degree + 1):
        for m in range(degree + 1):
            if m > l:
                continue
            stokesS_gann[l][m] = quad.integrate(
                lambda x, l=l, m=m, P=P, model=net, R0=r0: Slm(x, model, l, m, R0, P),
                dim=3,
                N=n_quadrature,
                integration_domain=[[-1, 1], [-1, 1], [-1, 1]],
            )
            stokesS_gann[l][m] = stokesS_gann[l][m] / M * constant_factors(l, m)

    # Convert results back to tensors
    return torch.tensor(stokesC_gann), torch.tensor(stokesS_gann)


def legendre_factory_torch(n=7):
    """From darioizzo/geodesyNets.
    Generates a dictionary of callables: the associated Legendre Polynomials.
    All the callables will be vectorized so that they can work on torch tensors


    Args:
        n (int, optional): maximum degree and order of the Legendre associated. Defaults to 16.

    Returns:
        dict: a dictionary with callables P[l][m](torch.tensor (N,1)) -> torch.tensor (N,1)
    """
    P = dict()
    for i in range(n + 1):
        P[i] = dict()
    P[0][0] = lambda x: torch.ones(len(x), 1)
    # First we compute all the associated legendre polynomials with l=m. (0,0), (1,1), (2,2), ....
    for l in range(n):
        P[l + 1][l + 1] = (
            lambda x, l=l: -(2.0 * l + 1.0) * torch.sqrt(1.0 - x**2) * P[l][l](x)
        )
    # Then we compute the ones with l+1,l. (1,0), (2,1), (3,2), ....
    for l in range(n):
        P[l + 1][l] = lambda x, l=l: (2.0 * l + 1.0) * x * P[l][l](x)
    # Then all the rest
    for m in range(n + 1):
        for l in range(m + 1, n):
            P[l + 1][m] = lambda x, l=l, m=m: (
                (2.0 * l + 1.0) * x * P[l][m](x) - (l + m) * P[l - 1][m](x)
            ) / (l - m + 1.0)
    return P


def Clm(x, model, l, m, R0, P):
    """From darioizzo/geodesyNets.
    Integrand to compute the cos Stoke coefficient from a density. Lacks normalization and constant factors.

    Args:
        x (torch.tensor (N,3)): Cartesian coordinates where to evaluate te integrand.
        model (callable): callable returning the mass density at points.
        l (int): degree.
        m (int): order.
        P (dict): vectorized Legendre associated polynomials constructed at the correct order/degree.

    Returns:
        torch.tensor (N,1): the itegrand evaluated at points
    """
    sph = cart2spherical_torch(x)
    retval = model(x)
    retval = retval * (sph[:, 0].view(-1, 1) / R0) ** l
    retval = retval * P[l][m](torch.cos(sph[:, 1].view(-1, 1)))
    retval = retval * torch.cos(m * sph[:, 2].view(-1, 1))
    return retval


def Slm(x, model, l, m, R0, P):
    """From darioizzo/geodesyNets.
    Integrand to compute the sin Stoke coefficient from a density. Lacks normalization and constant factors.

    Args:
        x (torch.tensor (N,3)): Cartesian coordinates where to evaluate te integrand.
        model (callable): callable returning the mass density at points.
        l (int): degree.
        m (int): order.
        P (dict): vectorized Legendre associated polynomials constructed at the correct order/degree.

    Returns:
        torch.tensor (N,1): the itegrand evaluated at points.
    """
    sph = cart2spherical_torch(x)
    retval = model(x)
    retval = retval * (sph[:, 0].view(-1, 1) / R0) ** l
    retval = retval * P[l][m](torch.cos(sph[:, 1].view(-1, 1)))
    retval = retval * torch.sin(m * sph[:, 2].view(-1, 1))
    return retval


def cart2spherical_torch(x):
    """From darioizzo/geodesyNets.
    Converts Cartesian to spherical coordinates defined as
     - r (radius)
     - theta (colatitude in [0, pi])
     - phi (longitude, in [0, 2pi])

     and works with torch tensors so that multiple conversions can be made at once.

    Args:
        x (torch.tensor (N, 3)): Cartesian points

    Returns:
        torch.tensor (N, 3): Corresponding spherical coordinates (r, theta, phi)
    """
    r = torch.norm(x, dim=1).view(-1, 1)
    theta = torch.arccos(x[:, 2].view(-1, 1) / r)
    phi = torch.atan2(x[:, 1], x[:, 0]).view(-1, 1)
    phi[phi < 0] = phi[phi < 0] + 2 * torch.pi
    return torch.concat((r, theta, phi), dim=1)


def constant_factors(l, m):
    """From darioizzo/geodesyNets.
    Computes the missing constant factors to compute Stokes coefficients

    Args:
        l (int): degree.
        m (int): order.

    Returns:
        float: the factor
    """
    if m == 0:
        delta = 1
    else:
        delta = 0
    retval = (2 - delta) * np.math.factorial(l - m) / np.math.factorial(l + m)
    retval = retval * np.sqrt(
        1
        / (2 - delta)
        / np.math.factorial(l - m)
        / (2 * l + 1)
        * np.math.factorial(l + m)
    )
    return retval
