"""
================================================================================
Drona Aviation Swarm - Trajectory Generation Engine
================================================================================
Implements $C^2$ continuous minimum-jerk trajectory interpolation using quintic
(5th-order) polynomial time scaling. Ensures smooth zero-velocity and zero-acceleration
boundary transitions for single drones and synchronized multi-UAV swarms.

Mathematical Formulation:
-------------------------
Normalized time parameter: tau = (t - t_start) / duration in [0, 1]

Position scaling:
    s(tau) = 10 * tau^3 - 15 * tau^4 + 6 * tau^5
Boundary invariants:
    s(0) = 0,   s(1) = 1

Velocity scaling (ds/dt):
    ds/dtau = 30 * tau^2 - 60 * tau^3 + 30 * tau^4
    s_dot(tau) = (1 / duration) * (ds/dtau)
Boundary invariants:
    s_dot(0) = 0,   s_dot(1) = 0   (Smooth zero-velocity start and arrival)

Acceleration scaling (d^2s/dt^2):
    d^2s/dtau^2 = 60 * tau - 180 * tau^2 + 120 * tau^3
    s_ddot(tau) = (1 / duration^2) * (d^2s/dtau^2)
Boundary invariants:
    s_ddot(0) = 0,   s_ddot(1) = 0  (Zero-jerk onset, eliminates rotor shock)

Spatial coordinate evolution:
    p(t) = p_start + (p_end - p_start) * s(tau)
    v(t) = (p_end - p_start) * s_dot(tau)
    a(t) = (p_end - p_start) * s_ddot(tau)
================================================================================
"""

from typing import Tuple, Union
import numpy as np


def quintic_polynomial_scaling(
    t: float,
    duration: float
) -> Tuple[float, float, float]:
    """
    Compute quintic polynomial scaling parameter s(tau), s_dot(tau), and s_ddot(tau).

    Parameters:
    -----------
    t : float
        Elapsed time from maneuver start in seconds.
    duration : float
        Total duration of the maneuver in seconds (must be > 0).

    Returns:
    --------
    s : float
        Position scaling factor in [0.0, 1.0].
    s_dot : float
        Velocity scaling factor (1/s).
    s_ddot : float
        Acceleration scaling factor (1/s^2).
    """
    if duration <= 0.0:
        raise ValueError(f"Trajectory duration must be positive, received: {duration}")

    tau = np.clip(t / duration, 0.0, 1.0)

    # Position scaling
    s = 10.0 * (tau ** 3) - 15.0 * (tau ** 4) + 6.0 * (tau ** 5)

    # Velocity scaling: ds/dt = (1/T) * ds/dtau
    if 0.0 < tau < 1.0:
        ds_dtau = 30.0 * (tau ** 2) - 60.0 * (tau ** 3) + 30.0 * (tau ** 4)
        s_dot = ds_dtau / duration

        # Acceleration scaling: d2s/dt2 = (1/T^2) * d2s/dtau2
        d2s_dtau2 = 60.0 * tau - 180.0 * (tau ** 2) + 120.0 * (tau ** 3)
        s_ddot = d2s_dtau2 / (duration ** 2)
    else:
        s_dot = 0.0
        s_ddot = 0.0

    return s, s_dot, s_ddot


def generate_point_to_point_trajectory(
    start_pos: np.ndarray,
    end_pos: np.ndarray,
    t: float,
    duration: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate instantaneous 3D position, velocity, and acceleration for a single UAV.

    Parameters:
    -----------
    start_pos : np.ndarray of shape (3,)
        Starting 3D coordinates [x, y, z].
    end_pos : np.ndarray of shape (3,)
        Target 3D coordinates [x, y, z].
    t : float
        Elapsed time since trajectory initiation in seconds.
    duration : float
        Total transition duration in seconds.

    Returns:
    --------
    pos : np.ndarray of shape (3,)
        Desired position [x, y, z] at time t.
    vel : np.ndarray of shape (3,)
        Desired velocity vector [vx, vy, vz] at time t.
    acc : np.ndarray of shape (3,)
        Desired acceleration vector [ax, ay, az] at time t.
    """
    start = np.asarray(start_pos, dtype=np.float64)
    end = np.asarray(end_pos, dtype=np.float64)
    delta = end - start

    s, s_dot, s_ddot = quintic_polynomial_scaling(t, duration)

    pos = start + delta * s
    vel = delta * s_dot
    acc = delta * s_ddot

    return pos, vel, acc


class SwarmTrajectoryPlanner:
    """
    Synchronized trajectory coordinator for the entire 4-drone swarm.
    Translates all drones simultaneously along a shared timeline.
    """

    def __init__(
        self,
        start_positions: np.ndarray,
        target_positions: np.ndarray,
        duration: float,
        start_time: float = 0.0
    ):
        """
        Initialize a synchronized swarm transition maneuver.

        Parameters:
        -----------
        start_positions : np.ndarray of shape (N, 3)
            Starting coordinates of all drones in the swarm.
        target_positions : np.ndarray of shape (N, 3)
            Destination coordinates of all drones.
        duration : float
            Total transition duration in seconds.
        start_time : float
            Global epoch start time in seconds.
        """
        self.start_positions = np.asarray(start_positions, dtype=np.float64)
        self.target_positions = np.asarray(target_positions, dtype=np.float64)
        self.duration = float(duration)
        self.start_time = float(start_time)

        if self.start_positions.shape != self.target_positions.shape:
            raise ValueError(
                f"Shape mismatch: start {self.start_positions.shape} vs target {self.target_positions.shape}"
            )
        self.num_drones = self.start_positions.shape[0]

    def get_swarm_state(self, current_time: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute positions, velocities, and accelerations for all N drones at current_time.

        Returns:
        --------
        positions : np.ndarray of shape (N, 3)
        velocities : np.ndarray of shape (N, 3)
        accelerations : np.ndarray of shape (N, 3)
        """
        elapsed = current_time - self.start_time
        s, s_dot, s_ddot = quintic_polynomial_scaling(elapsed, self.duration)

        deltas = self.target_positions - self.start_positions

        positions = self.start_positions + deltas * s
        velocities = deltas * s_dot
        accelerations = deltas * s_ddot

        return positions, velocities, accelerations

    def is_completed(self, current_time: float) -> bool:
        """Return True if current_time >= start_time + duration."""
        return (current_time - self.start_time) >= self.duration

