"""
================================================================================
Drona Aviation Swarm - Real-time Collision Avoidance & Safety Monitor
================================================================================
Implements rigorous multi-drone separation monitoring, arena geofencing,
and fail-safe triggers to satisfy competition safety requirements.
================================================================================
"""

from typing import Dict, List, NamedTuple, Tuple
import numpy as np


class SeparationViolation(NamedTuple):
    drone_a_id: int
    drone_b_id: int
    distance: float
    is_critical: bool  # True if < min_separation (0.4m), False if warning (< 0.55m)


class GeofenceViolation(NamedTuple):
    drone_id: int
    coordinate: np.ndarray
    axis: str
    bound_violated: float


class SwarmSafetyMonitor:
    """
    Continuous real-time safety and collision surveillance monitor.
    """

    def __init__(
        self,
        min_separation_distance: float = 0.40,
        warning_separation_distance: float = 0.55,
        arena_bounds: dict = None
    ):
        self.min_separation = float(min_separation_distance)
        self.warning_separation = float(warning_separation_distance)

        if arena_bounds is None:
            self.bounds = {
                "x_min": -2.5, "x_max": 2.5,
                "y_min": -2.5, "y_max": 2.5,
                "z_min":  0.2, "z_max": 2.5
            }
        else:
            self.bounds = arena_bounds

    def check_swarm_separation(
        self,
        positions: Dict[int, np.ndarray]
    ) -> List[SeparationViolation]:
        """
        Evaluate pairwise distances for all unique drone pairs.
        """
        violations = []
        d_ids = list(positions.keys())
        n = len(d_ids)

        for i in range(n):
            for j in range(i + 1, n):
                id_a, id_b = d_ids[i], d_ids[j]
                dist = float(np.linalg.norm(positions[id_a] - positions[id_b]))

                if dist < self.min_separation:
                    violations.append(SeparationViolation(id_a, id_b, dist, is_critical=True))
                elif dist < self.warning_separation:
                    violations.append(SeparationViolation(id_a, id_b, dist, is_critical=False))

        return violations

    def check_geofence(
        self,
        positions: Dict[int, np.ndarray]
    ) -> List[GeofenceViolation]:
        """
        Verify all drones are strictly contained inside the physical flying arena boundaries.
        """
        violations = []
        for d_id, pos in positions.items():
            x, y, z = pos[0], pos[1], pos[2]

            if x < self.bounds["x_min"]:
                violations.append(GeofenceViolation(d_id, pos, "X", self.bounds["x_min"]))
            elif x > self.bounds["x_max"]:
                violations.append(GeofenceViolation(d_id, pos, "X", self.bounds["x_max"]))

            if y < self.bounds["y_min"]:
                violations.append(GeofenceViolation(d_id, pos, "Y", self.bounds["y_min"]))
            elif y > self.bounds["y_max"]:
                violations.append(GeofenceViolation(d_id, pos, "Y", self.bounds["y_max"]))

            if z < self.bounds["z_min"]:
                violations.append(GeofenceViolation(d_id, pos, "Z", self.bounds["z_min"]))
            elif z > self.bounds["z_max"]:
                violations.append(GeofenceViolation(d_id, pos, "Z", self.bounds["z_max"]))

        return violations

    def validate_trajectory_preflight(
        self,
        trajectory_eval_func,
        duration: float,
        dt: float = 0.05
    ) -> Tuple[bool, str]:
        """
        Pre-flight scan: evaluates proposed swarm trajectory at discrete time intervals
        before executing on physical drones. Rejects trajectory if collision or boundary breach occurs.
        """
        steps = int(duration / dt)
        for step in range(steps + 1):
            t = step * dt
            pos_arr, _, _ = trajectory_eval_func(t)
            pos_dict = {i + 1: pos_arr[i] for i in range(len(pos_arr))}

            # Separation check
            sep_viol = self.check_swarm_separation(pos_dict)
            critical = [v for v in sep_viol if v.is_critical]
            if critical:
                v = critical[0]
                return False, f"Collision risk between Drone {v.drone_a_id} and {v.drone_b_id} at t={t:.2f}s (dist={v.distance:.2f}m < {self.min_separation}m)"

            # Geofence check
            geo_viol = self.check_geofence(pos_dict)
            if geo_viol:
                g = geo_viol[0]
                return False, f"Geofence breach by Drone {g.drone_id} on axis {g.axis} at t={t:.2f}s (pos={g.coordinate})"

        return True, "Pre-flight trajectory validation successful. Safe to execute."

