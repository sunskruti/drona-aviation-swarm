"""
================================================================================
Drona Aviation Swarm - Traveling Wave Trajectory Engine
================================================================================
Implements synchronized phase-shifted wave dynamics across the multi-UAV swarm.
Supports both Task 1 (Line Formation with Wave) and Task 2 (Circular Formation 
with Wave), with smooth entry/exit envelope modulation to prevent motor shock.

Mathematical Formulation:
-------------------------
For Drone i in {0, 1, ..., N-1}:
Phase offset:
    phi_i = i * (2 * pi / N)

Harmonic vertical oscillation:
    z_i(t) = z_0,i + E(t) * A * sin(2 * pi * f * t - phi_i)

Where:
    A       : Wave amplitude in meters (default: 0.25 m)
    f       : Frequency in Hz (default: 0.25 Hz -> T = 4.0 s)
    z_0,i   : Nominal base altitude (1.5 m)
    E(t)    : C^2 Smooth envelope function in [0, 1] fading in at onset and fading out at completion.

Analytical Kinematic Derivatives:
    v_z,i(t) = E(t) * A * omega * cos(omega * t - phi_i) + dE/dt * A * sin(...)
    a_z,i(t) = -E(t) * A * omega^2 * sin(omega * t - phi_i) + ...

Preservation of Formation Geometry:
    The lateral horizontal positions (x_i, y_i) remain rigidly locked to the base 
    geometry (collinear line or perimeter circle) throughout the wave execution.
================================================================================
"""

from typing import List, Tuple, Union
import numpy as np


def generate_wave_state(
    base_pos: np.ndarray,
    t: float,
    amplitude: float = 0.25,
    frequency: float = 0.25,
    phase_offset: float = 0.0,
    envelope: float = 1.0,
    envelope_dot: float = 0.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute instantaneous position, velocity, and acceleration for a single drone executing a wave.

    Parameters:
    -----------
    base_pos : np.ndarray of shape (3,)
        Nominal base coordinate [x_0, y_0, z_0].
    t : float
        Elapsed time in seconds.
    amplitude : float
        Vertical oscillation amplitude in meters.
    frequency : float
        Oscillation frequency in Hz.
    phase_offset : float
        Phase delay angle in radians.
    envelope : float
        Current amplitude scaling factor in [0, 1].
    envelope_dot : float
        Time derivative of the envelope factor.

    Returns:
    --------
    pos : np.ndarray of shape (3,)
    vel : np.ndarray of shape (3,)
    acc : np.ndarray of shape (3,)
    """
    omega = 2.0 * np.pi * frequency
    theta = omega * t - phase_offset

    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    # Position
    pos = np.copy(base_pos)
    pos[2] = base_pos[2] + envelope * amplitude * sin_theta

    # Velocity
    vel = np.zeros(3, dtype=np.float64)
    vel[2] = (envelope_dot * amplitude * sin_theta) + (envelope * amplitude * omega * cos_theta)

    # Acceleration
    acc = np.zeros(3, dtype=np.float64)
    acc[2] = -envelope * amplitude * (omega ** 2) * sin_theta

    return pos, vel, acc


def compute_smooth_envelope(
    t: float,
    total_duration: float,
    ramp_duration: float = 1.5
) -> Tuple[float, float]:
    """
    Compute a C^2 smooth window envelope to smoothly ramp wave amplitude up from 0 
    and ramp back down to 0 at trajectory termination.
    """
    if t <= 0.0 or t >= total_duration:
        return 0.0, 0.0

    # Ramp up phase
    if t < ramp_duration:
        tau = t / ramp_duration
        s = 10.0 * (tau ** 3) - 15.0 * (tau ** 4) + 6.0 * (tau ** 5)
        ds = (30.0 * (tau ** 2) - 60.0 * (tau ** 3) + 30.0 * (tau ** 4)) / ramp_duration
        return s, ds

    # Ramp down phase
    elif t > (total_duration - ramp_duration):
        tau = (total_duration - t) / ramp_duration
        s = 10.0 * (tau ** 3) - 15.0 * (tau ** 4) + 6.0 * (tau ** 5)
        ds = -(30.0 * (tau ** 2) - 60.0 * (tau ** 3) + 30.0 * (tau ** 4)) / ramp_duration
        return s, ds

    # Steady state full amplitude
    else:
        return 1.0, 0.0


class LineWaveTrajectory:
    """
    Task 1: Orchestrates sequential traveling wave execution for drones in straight-line formation.
    """

    def __init__(
        self,
        base_positions: np.ndarray,
        amplitude: float = 0.25,
        frequency: float = 0.25,
        num_cycles: int = 3,
        ramp_duration: float = 1.5,
        start_time: float = 0.0,
    ):
        self.base_positions = np.asarray(base_positions, dtype=np.float64)
        self.num_drones = len(self.base_positions)
        self.amplitude = float(amplitude)
        self.frequency = float(frequency)
        self.num_cycles = int(num_cycles)
        self.period = 1.0 / self.frequency if self.frequency > 0 else 0.0
        self.total_duration = self.num_cycles * self.period + 2 * ramp_duration
        self.ramp_duration = float(ramp_duration)
        self.start_time = float(start_time)

        # Uniform progressive phase offsets: [0, pi/2, pi, 3pi/2] for N=4
        self.phase_offsets = np.array([
            i * (2.0 * np.pi / self.num_drones) for i in range(self.num_drones)
        ], dtype=np.float64)

    def get_swarm_state(self, current_time: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute 3D position, velocity, and acceleration for all N drones."""
        elapsed = current_time - self.start_time
        env, env_dot = compute_smooth_envelope(elapsed, self.total_duration, self.ramp_duration)

        positions = np.zeros_like(self.base_positions)
        velocities = np.zeros_like(self.base_positions)
        accelerations = np.zeros_like(self.base_positions)

        for i in range(self.num_drones):
            p, v, a = generate_wave_state(
                base_pos=self.base_positions[i],
                t=elapsed,
                amplitude=self.amplitude,
                frequency=self.frequency,
                phase_offset=self.phase_offsets[i],
                envelope=env,
                envelope_dot=env_dot
            )
            positions[i] = p
            velocities[i] = v
            accelerations[i] = a

        return positions, velocities, accelerations

    def is_completed(self, current_time: float) -> bool:
        return (current_time - self.start_time) >= self.total_duration


class CircularWaveTrajectory:
    """
    Task 2: Orchestrates traveling wave execution for drones in circular formation.
    """

    def __init__(
        self,
        base_positions: np.ndarray,
        amplitude: float = 0.25,
        frequency: float = 0.25,
        num_cycles: int = 3,
        ramp_duration: float = 1.5,
        start_time: float = 0.0,
    ):
        self.base_positions = np.asarray(base_positions, dtype=np.float64)
        self.num_drones = len(self.base_positions)
        self.amplitude = float(amplitude)
        self.frequency = float(frequency)
        self.num_cycles = int(num_cycles)
        self.period = 1.0 / self.frequency if self.frequency > 0 else 0.0
        self.total_duration = self.num_cycles * self.period + 2 * ramp_duration
        self.ramp_duration = float(ramp_duration)
        self.start_time = float(start_time)

        # Distribute phase around the circle
        self.phase_offsets = np.array([
            i * (2.0 * np.pi / self.num_drones) for i in range(self.num_drones)
        ], dtype=np.float64)

    def get_swarm_state(self, current_time: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute 3D position, velocity, and acceleration for all N drones."""
        elapsed = current_time - self.start_time
        env, env_dot = compute_smooth_envelope(elapsed, self.total_duration, self.ramp_duration)

        positions = np.zeros_like(self.base_positions)
        velocities = np.zeros_like(self.base_positions)
        accelerations = np.zeros_like(self.base_positions)

        for i in range(self.num_drones):
            p, v, a = generate_wave_state(
                base_pos=self.base_positions[i],
                t=elapsed,
                amplitude=self.amplitude,
                frequency=self.frequency,
                phase_offset=self.phase_offsets[i],
                envelope=env,
                envelope_dot=env_dot
            )
            positions[i] = p
            velocities[i] = v
            accelerations[i] = a

        return positions, velocities, accelerations

    def is_completed(self, current_time: float) -> bool:
        return (current_time - self.start_time) >= self.total_duration

