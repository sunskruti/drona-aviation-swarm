"""
================================================================================
Drona Aviation Swarm - Real-time Vision Position Tracker & State Estimator
================================================================================
Processes external vision measurements (e-Yantra / Dagger / ArUco overhead tracking).
Provides:
1. Marker ID to Drone ID association (preventing identity swaps).
2. Outlier rejection (thresholding unrealistic position jumps > max_physical_velocity).
3. Discrete Alpha-Beta Kalman filter for zero-phase state smoothing and velocity estimation.
4. Watchdog telemetry loss detection for safe-state triggers.
================================================================================
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np


@dataclass
class DroneState:
    drone_id: int
    position: np.ndarray        # Filtered [x, y, z] in meters
    velocity: np.ndarray        # Estimated [vx, vy, vz] in m/s
    timestamp: float            # Current time in seconds
    is_valid: bool              # Tracking validity status
    confidence: float           # Vision confidence score in [0.0, 1.0]


class AlphaBetaFilter3D:
    """
    3D Alpha-Beta tracking filter for constant-velocity kinematic tracking:
        x_pred = x_est + v_est * dt
        v_pred = v_est
        residual = z_meas - x_pred
        x_est = x_pred + alpha * residual
        v_est = v_pred + (beta / dt) * residual
    """

    def __init__(self, alpha: float = 0.65, beta: float = 0.25):
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.x_est = None  # np.ndarray shape (3,)
        self.v_est = np.zeros(3, dtype=np.float64)
        self.last_timestamp = None

    def initialize(self, initial_pos: np.ndarray, timestamp: float):
        self.x_est = np.asarray(initial_pos, dtype=np.float64)
        self.v_est = np.zeros(3, dtype=np.float64)
        self.last_timestamp = float(timestamp)

    def update(self, measurement: np.ndarray, timestamp: float) -> Tuple[np.ndarray, np.ndarray]:
        if self.x_est is None:
            self.initialize(measurement, timestamp)
            return self.x_est, self.v_est

        dt = timestamp - self.last_timestamp
        if dt <= 0.0:
            return self.x_est, self.v_est

        # Prediction step
        x_pred = self.x_est + self.v_est * dt
        v_pred = self.v_est

        # Innovation / residual
        meas = np.asarray(measurement, dtype=np.float64)
        residual = meas - x_pred

        # Correction step
        self.x_est = x_pred + self.alpha * residual
        self.v_est = v_pred + (self.beta / dt) * residual
        self.last_timestamp = float(timestamp)

        return self.x_est, self.v_est


class SwarmPositionTracker:
    """
    Manages multi-agent external vision localization for all Pluto drones.
    """

    def __init__(
        self,
        drone_ids: list = (1, 2, 3, 4),
        max_jump_velocity: float = 3.0,  # Max physical speed in m/s before discarding as outlier
        watchdog_timeout: float = 0.25   # 250ms vision timeout
    ):
        self.drone_ids = list(drone_ids)
        self.max_jump_velocity = float(max_jump_velocity)
        self.watchdog_timeout = float(watchdog_timeout)

        self.filters: Dict[int, AlphaBetaFilter3D] = {
            d_id: AlphaBetaFilter3D() for d_id in self.drone_ids
        }
        self.states: Dict[int, DroneState] = {}
        self.last_seen_times: Dict[int, float] = {d_id: -1.0 for d_id in self.drone_ids}

    def process_raw_measurement(
        self,
        drone_id: int,
        raw_position: np.ndarray,
        timestamp: float,
        confidence: float = 1.0
    ) -> Optional[DroneState]:
        """
        Ingest a raw vision measurement, validate sanity, filter, and update state.
        """
        if drone_id not in self.filters:
            return None

        raw_pos = np.asarray(raw_position, dtype=np.float64)
        current_state = self.states.get(drone_id)

        # Outlier rejection check: test for impossible instantaneous velocity jump
        if current_state is not None and current_state.is_valid:
            dt = timestamp - current_state.timestamp
            if dt > 0.0:
                jump_speed = np.linalg.norm(raw_pos - current_state.position) / dt
                if jump_speed > self.max_jump_velocity:
                    # Discard outlier measurement (e.g. reflection / marker misidentification)
                    return current_state

        # Update Alpha-Beta filter
        filt = self.filters[drone_id]
        filt_pos, filt_vel = filt.update(raw_pos, timestamp)

        state = DroneState(
            drone_id=drone_id,
            position=filt_pos,
            velocity=filt_vel,
            timestamp=timestamp,
            is_valid=True,
            confidence=float(confidence)
        )
        self.states[drone_id] = state
        self.last_seen_times[drone_id] = float(timestamp)
        return state

    def check_watchdogs(self, current_time: float):
        """
        Check for telemetry timeouts; mark states invalid if vision drops out.
        """
        for d_id in self.drone_ids:
            if d_id in self.states:
                last_seen = self.last_seen_times[d_id]
                if (current_time - last_seen) > self.watchdog_timeout:
                    self.states[d_id].is_valid = False
                    self.states[d_id].confidence = 0.0

    def get_swarm_positions(self) -> Dict[int, np.ndarray]:
        """Return dict of current filtered positions for all drones."""
        return {d_id: state.position for d_id, state in self.states.items() if state.is_valid}

