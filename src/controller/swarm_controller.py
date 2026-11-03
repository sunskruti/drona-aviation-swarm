"""
================================================================================
Drona Aviation Swarm - Swarm Controller Manager
================================================================================
Coordinates closed-loop tracking for all 4 Pluto drones simultaneously.
Calculates individual velocity commands, formation-level error statistics,
and handles integration of feedforward trajectories.
================================================================================
"""

from typing import Dict, List, Tuple
import numpy as np
from .pid_controller import PIDController3D


class SwarmController:
    """
    Multi-agent closed-loop controller coordinator.
    """

    def __init__(self, drone_ids: List[int] = (1, 2, 3, 4)):
        self.drone_ids = list(drone_ids)
        self.controllers: Dict[int, PIDController3D] = {
            d_id: PIDController3D() for d_id in self.drone_ids
        }

    def reset(self):
        for ctrl in self.controllers.values():
            ctrl.reset()

    def update(
        self,
        target_positions: np.ndarray,
        actual_positions: Dict[int, np.ndarray],
        target_velocities: np.ndarray = None,
        current_time: float = 0.0
    ) -> Tuple[Dict[int, np.ndarray], float, float]:
        """
        Compute control commands and formation tracking metrics.

        Parameters:
        -----------
        target_positions : np.ndarray of shape (N, 3)
            Desired positions ordered by drone index 0..N-1.
        actual_positions : Dict[int, np.ndarray]
            Current actual positions keyed by drone_id.
        target_velocities : np.ndarray of shape (N, 3), optional
            Feedforward target velocities.
        current_time : float
            Timestamp in seconds.

        Returns:
        --------
        commands : Dict[int, np.ndarray]
            Commanded velocity vectors for each active drone.
        mean_formation_error : float
            Mean Euclidean position tracking error across all active drones.
        max_error : float
            Maximum individual tracking error.
        """
        commands = {}
        errors = []

        for idx, d_id in enumerate(self.drone_ids):
            if d_id not in actual_positions:
                continue

            target_p = target_positions[idx]
            actual_p = actual_positions[d_id]
            target_v = target_velocities[idx] if target_velocities is not None else None

            u_cmd, err = self.controllers[d_id].compute(
                target_pos=target_p,
                actual_pos=actual_p,
                feedforward_vel=target_v,
                current_time=current_time
            )
            commands[d_id] = u_cmd
            errors.append(np.linalg.norm(err))

        if errors:
            mean_error = float(np.mean(errors))
            max_err = float(np.max(errors))
        else:
            mean_error = 0.0
            max_err = 0.0

        return commands, mean_error, max_err

