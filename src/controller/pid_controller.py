"""
================================================================================
Drona Aviation Swarm - Closed-Loop 3D PID Controller with Drift Correction
================================================================================
Implements a 2-DOF (Degree-of-Freedom) Feedback + Feedforward Controller:
    u_command = v_feedforward + u_PID(error)

Features:
- Separate gain scheduling for horizontal plane (XY) and vertical axis (Z).
- Anti-windup conditional integration to eliminate integral saturation.
- Low-pass filtered derivative term to attenuate camera quantization noise.
- Strict velocity saturation clamping to ensure flight stability on Pluto nano-drones.
================================================================================
"""

from typing import Tuple, Union
import numpy as np


class PIDController3D:
    """
    Decoupled 3D Cartesian PID controller with anti-windup and derivative filtering.
    """

    def __init__(
        self,
        kp_xy: float = 1.2,
        ki_xy: float = 0.05,
        kd_xy: float = 0.35,
        kp_z: float = 1.5,
        ki_z: float = 0.10,
        kd_z: float = 0.40,
        max_vel_xy: float = 0.80,   # Max horizontal speed in m/s
        max_vel_z: float = 0.50,    # Max vertical speed in m/s
        max_integral_xy: float = 0.30,
        max_integral_z: float = 0.20,
        d_filter_alpha: float = 0.70  # Low-pass filter coefficient for derivative
    ):
        self.Kp = np.array([kp_xy, kp_xy, kp_z], dtype=np.float64)
        self.Ki = np.array([ki_xy, ki_xy, ki_z], dtype=np.float64)
        self.Kd = np.array([kd_xy, kd_xy, kd_z], dtype=np.float64)

        self.max_vel = np.array([max_vel_xy, max_vel_xy, max_vel_z], dtype=np.float64)
        self.max_integral = np.array([max_integral_xy, max_integral_xy, max_integral_z], dtype=np.float64)
        self.d_filter_alpha = float(d_filter_alpha)

        # Internal state
        self.integral = np.zeros(3, dtype=np.float64)
        self.prev_error = np.zeros(3, dtype=np.float64)
        self.filtered_derivative = np.zeros(3, dtype=np.float64)
        self.last_time = None

    def reset(self):
        """Reset internal controller accumulators."""
        self.integral = np.zeros(3, dtype=np.float64)
        self.prev_error = np.zeros(3, dtype=np.float64)
        self.filtered_derivative = np.zeros(3, dtype=np.float64)
        self.last_time = None

    def compute(
        self,
        target_pos: np.ndarray,
        actual_pos: np.ndarray,
        feedforward_vel: np.ndarray = None,
        current_time: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute commanded 3D velocity correction vector.

        Parameters:
        -----------
        target_pos : np.ndarray (3,)
            Desired position setpoint from trajectory generator.
        actual_pos : np.ndarray (3,)
            Measured position from vision localization system.
        feedforward_vel : np.ndarray (3,)
            Analytic target velocity from trajectory generator (optional).
        current_time : float
            Current timestamp in seconds.

        Returns:
        --------
        u_cmd : np.ndarray (3,)
            Clamped 3D velocity command [vx, vy, vz] for Pluto flight stack.
        error : np.ndarray (3,)
            Current position error [ex, ey, ez].
        """
        target = np.asarray(target_pos, dtype=np.float64)
        actual = np.asarray(actual_pos, dtype=np.float64)
        error = target - actual

        if feedforward_vel is None:
            v_ff = np.zeros(3, dtype=np.float64)
        else:
            v_ff = np.asarray(feedforward_vel, dtype=np.float64)

        if self.last_time is None:
            dt = 0.02  # Default 50Hz dt on first sample
        else:
            dt = current_time - self.last_time

        if dt <= 0.0 or dt > 0.5:
            dt = 0.02  # Sanity clamp against pause or timing jumps

        # 1. Proportional term
        P_term = self.Kp * error

        # 2. Integral term with anti-windup clamping
        self.integral += error * dt
        self.integral = np.clip(self.integral, -self.max_integral, self.max_integral)
        I_term = self.Ki * self.integral

        # 3. Filtered derivative term
        raw_derivative = (error - self.prev_error) / dt
        self.filtered_derivative = (
            self.d_filter_alpha * self.filtered_derivative
            + (1.0 - self.d_filter_alpha) * raw_derivative
        )
        D_term = self.Kd * self.filtered_derivative

        # 4. Total control law: Feedforward + Feedback
        raw_command = v_ff + P_term + I_term + D_term

        # 5. Output velocity saturation limits
        u_cmd = np.clip(raw_command, -self.max_vel, self.max_vel)

        # Update historical state
        self.prev_error = np.copy(error)
        self.last_time = float(current_time)

        return u_cmd, error

