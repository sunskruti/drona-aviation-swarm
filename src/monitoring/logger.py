"""
================================================================================
Drona Aviation Swarm - Flight Telemetry Logger & Performance Reporter
================================================================================
Records high-frequency multi-drone flight metrics at 50 Hz.
Calculates competition evaluation metrics:
- Mean Formation Error (E_f)
- Maximum Error (E_max)
- Root-Mean-Square Error (RMSE) across X, Y, Z
- Inter-drone minimum separation
- Synchronization milestone completion times
Exports data to CSV for the mandatory 10-page Performance Report.
================================================================================
"""

import csv
import os
from typing import Dict, List
import numpy as np


class FlightDataLogger:
    """
    In-memory and file-based flight data logging facility.
    """

    def __init__(self, log_filepath: str = "flight_data.csv"):
        self.log_filepath = log_filepath
        self.records: List[dict] = []

    def log_step(
        self,
        timestamp: float,
        drone_id: int,
        target_pos: np.ndarray,
        actual_pos: np.ndarray,
        cmd_vel: np.ndarray,
        min_separation: float,
        safety_status: str = "NORMAL"
    ):
        """Record telemetry for a single drone at the current control step."""
        err = target_pos - actual_pos
        err_norm = float(np.linalg.norm(err))

        record = {
            "timestamp": round(float(timestamp), 4),
            "drone_id": int(drone_id),
            "target_x": round(float(target_pos[0]), 4),
            "target_y": round(float(target_pos[1]), 4),
            "target_z": round(float(target_pos[2]), 4),
            "actual_x": round(float(actual_pos[0]), 4),
            "actual_y": round(float(actual_pos[1]), 4),
            "actual_z": round(float(actual_pos[2]), 4),
            "error_x": round(float(err[0]), 4),
            "error_y": round(float(err[1]), 4),
            "error_z": round(float(err[2]), 4),
            "error_norm": round(err_norm, 4),
            "cmd_vx": round(float(cmd_vel[0]), 4),
            "cmd_vy": round(float(cmd_vel[1]), 4),
            "cmd_vz": round(float(cmd_vel[2]), 4),
            "min_separation": round(float(min_separation), 4),
            "safety_status": safety_status
        }
        self.records.append(record)

    def export_csv(self, filepath: str = None):
        """Export all logged rows to CSV file."""
        target_path = filepath if filepath is not None else self.log_filepath
        os.makedirs(os.path.dirname(target_path) or ".", exist_ok=True)

        if not self.records:
            return

        headers = list(self.records[0].keys())
        with open(target_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(self.records)

    def compute_summary_statistics(self) -> dict:
        """Compute aggregate statistics for the technical report."""
        if not self.records:
            return {}

        errors = np.array([r["error_norm"] for r in self.records])
        err_x = np.array([r["error_x"] for r in self.records])
        err_y = np.array([r["error_y"] for r in self.records])
        err_z = np.array([r["error_z"] for r in self.records])
        separations = np.array([r["min_separation"] for r in self.records])

        return {
            "total_samples": len(self.records),
            "mean_formation_error_m": float(np.mean(errors)),
            "max_formation_error_m": float(np.max(errors)),
            "rmse_x_m": float(np.sqrt(np.mean(err_x ** 2))),
            "rmse_y_m": float(np.sqrt(np.mean(err_y ** 2))),
            "rmse_z_m": float(np.sqrt(np.mean(err_z ** 2))),
            "rmse_total_m": float(np.sqrt(np.mean(errors ** 2))),
            "min_observed_separation_m": float(np.min(separations)),
        }

