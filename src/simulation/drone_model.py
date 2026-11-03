"""
================================================================================
Drona Aviation Swarm - Simulated Pluto Drone Dynamic Model
================================================================================
Simulates individual nano-drone flight dynamics with:
1. First-order actuator velocity lag response:
       dv/dt = (1 / tau) * (u_cmd - v)
2. Constant aerodynamical drift (rotor downwash & ground effect bias):
       p(t + dt) = p(t) + v(t) * dt + d_drift * dt
3. Optical tracking sensor noise injection:
       z_meas = p_true + N(0, sigma_noise^2)
================================================================================
"""

from typing import Tuple
import numpy as np


class SimulatedPlutoDrone:
    """
    Physical dynamics model of a single Pluto nano-quadrotor.
    """

    def __init__(
        self,
        drone_id: int,
        initial_position: np.ndarray,
        time_constant: float = 0.15,      # Motor/attitude velocity response lag (seconds)
        drift_velocity: np.ndarray = None, # Constant uncompensated physical drift (m/s)
        sensor_noise_std: float = 0.015   # 1.5 cm standard deviation optical noise
    ):
        self.drone_id = int(drone_id)
        self.position = np.asarray(initial_position, dtype=np.float64).copy()
        self.velocity = np.zeros(3, dtype=np.float64)
        self.tau = float(time_constant)

        if drift_velocity is None:
            # Subtle random aerodynamic bias on each drone: ~2-5 cm/s drift
            np.random.seed(self.drone_id * 42)
            self.drift = np.random.uniform(-0.04, 0.04, size=3)
            self.drift[2] = np.random.uniform(-0.02, 0.02)  # Vertical drift bias
        else:
            self.drift = np.asarray(drift_velocity, dtype=np.float64)

        self.sensor_noise_std = float(sensor_noise_std)

    def step(self, command_velocity: np.ndarray, dt: float) -> np.ndarray:
        """
        Advance drone physical simulation by dt seconds given commanded velocity u_cmd.
        Returns true ground-truth position.
        """
        u_cmd = np.asarray(command_velocity, dtype=np.float64)

        # 1. First-order velocity response: dv/dt = (1/tau) * (u_cmd - v)
        accel = (u_cmd - self.velocity) / self.tau
        self.velocity += accel * dt

        # 2. Position integration including environmental drift
        effective_vel = self.velocity + self.drift
        self.position += effective_vel * dt

        return np.copy(self.position)

    def get_measured_position(self) -> np.ndarray:
        """
        Emulate external vision camera observation corrupted by Gaussian noise.
        """
        noise = np.random.normal(0.0, self.sensor_noise_std, size=3)
        return self.position + noise

