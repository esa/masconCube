"""
Asteroid trajectory simulation and plotting module.
Most of the code is adapted from [collision-free-swarm](https://gitlab.com/EuropeanSpaceAgency/collision-free-swarm)
"""

from copy import deepcopy
from typing import Tuple

import cascade as csc
import heyoka as hy
import numpy as np
import pykep as pk
from scipy.spatial.transform import Rotation as rot

# === Constants ===
ROTATION_CONVERSION = (
    2 * np.pi * 24 / pk.DAY2SEC
)  # Convert from hours/rounds to rad/sec
GRAVITATIONAL_CONSTANT = 6.67430e-11  # m^3 kg^-1 s^-2
CUBESAT_AREA = 1.0  # m^2
CUBESAT_MASS = 12.0  # kg
SOLAR_CONSTANT = 1360  # W/m^2 at 1 AU
SPEED_OF_LIGHT = 3e8  # m/s
SUN_DIRECTION = np.array([0.5, 0.5, 0.0]) / np.linalg.norm([0.5, 0.5, 0.0])

ASTEROIDS_DATABASE = {
    "itokawa": {
        "M": 3.51e10,
        "W": ROTATION_CONVERSION / 12.1,
        "ra": 1.69 * pk.AU,
        "rp": 0.95 * pk.AU,
        "diameter": 535,
    },
    "bennu": {
        "M": 7.329e10,
        "W": ROTATION_CONVERSION / 4.3,
        "ra": 1.35 * pk.AU,
        "rp": 0.89 * pk.AU,
        "diameter": 565,
    },
    "eros": {
        "M": 6.69e15,
        "W": ROTATION_CONVERSION / 5.27,
        "ra": 1.78 * pk.AU,
        "rp": 1.13 * pk.AU,
        "diameter": 34400,
    },
    "planetesimal": {
        "M": 1e11,
        "W": ROTATION_CONVERSION / 10.0,
        "ra": 1.5 * pk.AU,
        "rp": 1.0 * pk.AU,
        "diameter": 1000,
    },
}


# === Utility Functions ===


def solar_radiation_force(
    distance_au: float, area: float, reflectivity: float = 1.0, shadow: float = 1.0
) -> float:
    """
    Computes the Solar Radiation Force.
    """
    if distance_au <= 0 or area <= 0:
        raise ValueError("Distance and area must be positive.")
    solar_irradiation = SOLAR_CONSTANT / distance_au**2
    return shadow * solar_irradiation / SPEED_OF_LIGHT * reflectivity * area


def build_initial_conditions(
    keplerian_params: np.ndarray,
    angular_velocity: np.ndarray,
    gravitational_param: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Builds initial conditions in the rotating frame.
    """
    pos_inertial, vel_inertial = pk.par2ic(keplerian_params, gravitational_param)
    vel_inertial = np.array(vel_inertial)
    vel_rot = np.cross(angular_velocity, np.array(pos_inertial))
    vel_rotating_frame = vel_inertial - vel_rot
    return np.array(pos_inertial), vel_rotating_frame


def _rotate_states(tgrid, states, w_vec):
    """Rotate position and velocity vectors by rotation vector over time."""
    rotated = deepcopy(states)
    for t, item in zip(tgrid, rotated):
        R = rot.from_rotvec(t * w_vec).as_matrix()
        item[:3] = R @ item[:3]
        item[3:6] = R @ item[3:6]
    return rotated


# === Simulation Functions ===


def simulate_trajectory(
    asteroid_name: str,
    mascon_points: np.ndarray,
    mascon_masses: np.ndarray,
    safety_coefficient: float = 1.4,
    exit_radius: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulates the trajectory of a spacecraft around an asteroid.

    Returns:
        Tuple of (body frame trajectory, inertial frame trajectory).
    """
    # Input validation
    if asteroid_name not in ASTEROIDS_DATABASE:
        raise ValueError(f"Asteroid '{asteroid_name}' not found in database.")
    if mascon_points.shape[1] != 3:
        raise ValueError("mascon_points must have shape (N, 3).")
    if mascon_points.shape[0] != mascon_masses.shape[0]:
        raise ValueError("mascon_points and mascon_masses must have the same length.")

    asteroid = ASTEROIDS_DATABASE[asteroid_name]
    diameter = asteroid["diameter"]
    mass = asteroid["M"]
    angular_velocity = asteroid["W"]
    perihelion_distance = asteroid["rp"]

    unit_length = diameter / 1.6  # Brillouin sphere radius
    unit_time = np.sqrt(unit_length**3 / GRAVITATIONAL_CONSTANT / mass)
    angular_velocity_scaled = angular_velocity * unit_time
    rotation_vector = np.array([0.0, 0.0, angular_velocity_scaled])

    a = (
        (np.max(mascon_points[:, 0]) - np.min(mascon_points[:, 0]))
        / 2
        * safety_coefficient
    )
    b = (
        (np.max(mascon_points[:, 1]) - np.min(mascon_points[:, 1]))
        / 2
        * safety_coefficient
    )
    c = (
        (np.max(mascon_points[:, 2]) - np.min(mascon_points[:, 2]))
        / 2
        * safety_coefficient
    )

    x, y, z = hy.make_vars("x", "y", "z")
    vx, vy, vz = hy.make_vars("vx", "vy", "vz")

    dynamics = csc.dynamics.mascon_asteroid(
        Gconst=1.0, masses=mascon_masses, points=mascon_points, omega=rotation_vector
    )

    # Solar radiation pressure acceleration
    solar_accel = (
        (
            solar_radiation_force(perihelion_distance / pk.AU, CUBESAT_AREA)
            / CUBESAT_MASS
        )
        / unit_length
        * unit_time**2
    )

    # Update dynamics with solar radiation pressure
    dynamics[3] = (vx, dynamics[3][1] - solar_accel * SUN_DIRECTION[0])
    dynamics[4] = (vy, dynamics[4][1] - solar_accel * SUN_DIRECTION[1])
    dynamics[5] = (vz, dynamics[5][1] - solar_accel * SUN_DIRECTION[2])

    delta_v_total = [0.0]  # Use list for closure

    def apply_delta_v(taylor_adaptive, _):
        pos = deepcopy(taylor_adaptive.state[:3])
        vel = deepcopy(taylor_adaptive.state[3:6])
        rot_vel = np.cross(rotation_vector, pos)
        inertial_vel = vel - rot_vel
        radial_unit = pos / np.linalg.norm(pos)
        delta_v = -2 * np.dot(radial_unit, inertial_vel) * radial_unit
        taylor_adaptive.state[3:6] = vel + delta_v
        delta_v_total[0] += np.linalg.norm(delta_v)
        return True

    ellipsoid_entry = hy.t_event(
        (x / a) ** 2 + (y / b) ** 2 + (z / c) ** 2 - 1, callback=apply_delta_v
    )
    sphere_exit = hy.t_event(
        x**2 + y**2 + z**2 - exit_radius**2, callback=apply_delta_v
    )

    taylor_integrator = hy.taylor_adaptive(
        sys=dynamics,
        state=[0.0] * 6,
        compact_mode=True,
        t_events=[ellipsoid_entry, sphere_exit],
    )

    deployment_sma = 1.5
    keplerian_params = [deployment_sma, 0.0, np.pi / 2, 0.0, 0.0, np.pi / 2]
    init_pos, init_vel = build_initial_conditions(keplerian_params, rotation_vector)
    taylor_integrator.state[:3] = init_pos
    taylor_integrator.state[3:6] = init_vel
    taylor_integrator.time = 0.0

    propagation_days = 1.0
    time_grid = np.linspace(0.0, propagation_days * pk.DAY2SEC / unit_time, 1000)
    trajectory = taylor_integrator.propagate_grid(time_grid)

    return trajectory[5], _rotate_states(time_grid, trajectory[5], rotation_vector)
