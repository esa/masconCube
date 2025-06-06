from copy import deepcopy
import numpy as np
import math
import scipy
import torch


def cart2spherical(x: float, y: float, z: float):
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


def spherical2cart(r, theta, phi):
    """Converts spherical to Cartesian coordinates, inverting the
    cart2spherical function.

    Args:
        r (float): radius
        theta (float): colatitude
        phi (float): longitude

    Returns:
        tuple: the Cartesian coordinates x,y,z
    """
    x = r*np.sin(theta)*np.cos(phi)
    y = r*np.sin(theta)*np.sin(phi)
    z = r*np.cos(theta)
    return x, y, z


def _single_mascon_contribution(mascon_point, mascon_mass, R0, l, m):
    x, y, z = mascon_point
    r, theta, phi = cart2spherical(x, y, z)
    stokesC = mascon_mass
    stokesC *= scipy.special.lpmn(m, l, np.cos(theta))[0]
    stokesS = deepcopy(stokesC)

    for mm in range(m+1):
        for ll in range(l+1):
            if ll < mm:
                continue
            if mm == 0:
                delta = 1
            else:
                delta = 0
            coeff1 = (r/R0)**ll
            coeff2C = np.cos(mm*phi)
            coeff2S = np.sin(mm*phi)
            coeff3 = (2-delta)*math.factorial(ll-mm) / \
                math.factorial(ll+mm)
            normalized = np.sqrt(math.factorial(
                ll+mm) / (2-delta)/(2*ll+1)/math.factorial(ll-mm))
            stokesS[mm, ll] *= coeff1*coeff2S*coeff3*normalized
            stokesC[mm, ll] *= coeff1*coeff2C*coeff3*normalized
    return (stokesC, stokesS)


def mascon2stokes(
    mascon_points: np.ndarray,
    mascon_masses: np.ndarray,
    r0: float,
    l: int,
    m: int,
):
    """Computes the Stokes coefficients from a mascon model or cube.

    Args:
        mascon_points (np.ndarray): Cartesian positions of the mascon points.
        mascon_masses (np.ndarray): Masses of the mascons.
        r0 (float): Characteristic radius (often mean equatorial radius) of the body.
        l (int): Degree of spherical harmonics.
        m (int): Order of spherical harmonics.

    Returns:
        tuple[np.ndarray, np.ndarray]: Stokes coefficients C and S.
    """
    # Preallocate Stokes coefficients
    stokesC = np.zeros((m + 1, l + 1))
    stokesS = np.zeros((m + 1, l + 1))

    # Compute contributions for all mascons
    for point, mass in zip(mascon_points, mascon_masses):
        tmpC, tmpS = _single_mascon_contribution(point, mass, r0, l, m)
        stokesC += tmpC
        stokesS += tmpS

    # Convert results back to tensors
    return stokesC.T, stokesS.T


def _single_mascon_contribution(
    mascon_point: np.ndarray,
    mascon_mass: float,
    r0: float,
    l: int,
    m: int,
):
    """Computes the contribution to Stokes coefficients from a single mascon."""
    x, y, z = mascon_point
    r, theta, phi = cart2spherical(x, y, z)

    # Precompute Legendre polynomials for efficiency
    legendre_polynomials = scipy.special.lpmn(m, l, np.cos(theta))[0]

    # Initialize Stokes coefficients
    stokesC = np.zeros((m + 1, l + 1))
    stokesS = np.zeros((m + 1, l + 1))

    for _order in range(m + 1):
        for _degree in range(_order, l + 1):
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



# Vectorized version of cartesian to spherical coordinates (radius, colatitude (0,pi), longitude  (0,2pi))
def cart2spherical_torch(x):
    """Converts Cartesian to spherical coordinates defined as
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
    theta = torch.arccos(x[:, 2].view(-1, 1)/r)
    phi = torch.atan2(x[:, 1], x[:, 0]).view(-1, 1)
    phi[phi < 0] = phi[phi < 0] + 2*torch.pi
    return torch.concat((r, theta, phi), dim=1)


# Integrand to compute the C Stokes coefficents (normalization and other factors not included)
def Clm(x, model, l, m, R0, P):
    """Integrand to compute the cos Stoke coefficient from a density. Lacks normalization and constant factors.

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
    retval = retval * (sph[:, 0].view(-1, 1)/R0)**l
    retval = retval * P[l][m](torch.cos(sph[:, 1].view(-1, 1)))
    retval = retval * torch.cos(m*sph[:, 2].view(-1, 1))
    return retval

# Integrand to compute the S Stokes coefficents (normalization and other factors not included)


def Slm(x, model, l, m, R0, P):
    """Integrand to compute the sin Stoke coefficient from a density. Lacks normalization and constant factors.

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
    retval = retval * (sph[:, 0].view(-1, 1)/R0)**l
    retval = retval * P[l][m](torch.cos(sph[:, 1].view(-1, 1)))
    retval = retval * torch.sin(m*sph[:, 2].view(-1, 1))
    return retval

# Normalization and factors to get the normalized Stokes


def constant_factors(l, m):
    """Computes the missing constant factors to compute Stokes coefficients

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
    retval = (2-delta)*math.factorial(l-m)/math.factorial(l+m)
    retval = retval * \
        np.sqrt(1/(2-delta)/math.factorial(l-m) /
                (2*l+1)*math.factorial(l+m))
    return retval

# Constructs the vectorized Legendre associated polynomials as lambda funcions


def legendre_factory_torch(n=16):
    """Generates a dictionary of callables: the associated Legendre Polynomials.
    All the callables will be vectorized so that they can work on torch tensors 


    Args:
        n (int, optional): maximum degree and order of the Legendre associated. Defaults to 16.

    Returns:
        dict: a dictionary with callables P[l][m](torch.tensor (N,1)) -> torch.tensor (N,1)
    """
    P = dict()
    for i in range(n+1):
        P[i] = dict()
    P[0][0] = lambda x: torch.ones(len(x), 1)
    # First we compute all the associated legendre polynomials with l=m. (0,0), (1,1), (2,2), ....
    for l in range(n):
        P[l+1][l+1] = lambda x, l=l: - \
            (2.*l+1.) * torch.sqrt(1.-x**2) * P[l][l](x)
    # Then we compute the ones with l+1,l. (1,0), (2,1), (3,2), ....
    for l in range(n):
        P[l+1][l] = lambda x, l=l: (2.*l+1.) * x * P[l][l](x)
    # Then all the rest
    for m in range(16+1):
        for l in range(m+1, 16):
            P[l+1][m] = lambda x, l=l, m=m: ((2.*l+1.) *
                                             x * P[l][m](x) - (l+m)*P[l-1][m](x))/(l-m+1.)
    return P
