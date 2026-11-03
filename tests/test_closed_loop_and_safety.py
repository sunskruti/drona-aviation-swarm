"""
================================================================================
Unit Tests for Closed-Loop Control, Localization, and Swarm Safety
================================================================================
"""

import sys
import os
import pytest
import numpy as np

# Ensure src path is accessible
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from localization.position_tracker import AlphaBetaFilter3D, SwarmPositionTracker
from localization.coordinate_transform import CoordinateTransformer, euler_to_rotation_matrix
from controller.pid_controller import PIDController3D
from safety.collision_detection import SwarmSafetyMonitor
from simulation.swarm_simulator import run_closed_loop_simulation


class TestLocalizationAndTracking:
    def test_alpha_beta_filter_smoothing(self):
        """Alpha-beta filter should reduce noise standard deviation."""
        filt = AlphaBetaFilter3D(alpha=0.6, beta=0.2)
        true_pos = np.array([1.0, 2.0, 1.5])
        noisy_samples = [true_pos + np.random.normal(0, 0.05, 3) for _ in range(50)]

        filtered_pos = []
        for i, sample in enumerate(noisy_samples):
            p_est, _ = filt.update(sample, timestamp=i * 0.02)
            if i > 20:  # Allow convergence
                filtered_pos.append(p_est)

        sample_err = np.std(np.linalg.norm(noisy_samples[20:] - true_pos, axis=1))
        filt_err = np.std(np.linalg.norm(filtered_pos - true_pos, axis=1))
        assert filt_err < sample_err  # Filter must smooth variance

    def test_outlier_rejection(self):
        """Sudden unrealistic jump (e.g. 10m in 20ms) should be rejected."""
        tracker = SwarmPositionTracker(drone_ids=[1], max_jump_velocity=3.0)
        p0 = np.array([0.0, 0.0, 1.5])
        tracker.process_raw_measurement(1, p0, timestamp=0.0)

        # Unrealistic jump: 10m away
        outlier_p = np.array([10.0, 0.0, 1.5])
        state = tracker.process_raw_measurement(1, outlier_p, timestamp=0.02)
        # Position should not have jumped to 10.0
        assert state.position[0] < 1.0


class TestPIDController:
    def test_error_correction_direction(self):
        """Positive position error should produce positive control velocity."""
        controller = PIDController3D(kp_xy=1.5, ki_xy=0.0, kd_xy=0.0)
        target = np.array([1.0, 0.0, 1.5])
        actual = np.array([0.5, 0.0, 1.5])  # Error is +0.5m along X
        u_cmd, err = controller.compute(target, actual, current_time=0.02)
        assert u_cmd[0] > 0.0  # Must drive forward

    def test_velocity_clamping(self):
        """Large step error must be clamped to max_vel_xy (0.8m/s)."""
        controller = PIDController3D(kp_xy=10.0, max_vel_xy=0.80)
        target = np.array([10.0, 0.0, 1.5])
        actual = np.array([0.0, 0.0, 1.5])  # Error is 10.0m
        u_cmd, _ = controller.compute(target, actual, current_time=0.02)
        assert np.isclose(u_cmd[0], 0.80)


class TestSafetyMonitor:
    def test_separation_detection(self):
        """Detect critical collision risk when distance < 0.40m."""
        monitor = SwarmSafetyMonitor(min_separation_distance=0.40, warning_separation_distance=0.55)
        safe_positions = {
            1: np.array([0.0, 0.0, 1.5]),
            2: np.array([1.0, 0.0, 1.5])
        }
        assert len(monitor.check_swarm_separation(safe_positions)) == 0

        unsafe_positions = {
            1: np.array([0.0, 0.0, 1.5]),
            2: np.array([0.30, 0.0, 1.5])  # 0.30m < 0.40m!
        }
        violations = monitor.check_swarm_separation(unsafe_positions)
        assert len(violations) == 1
        assert violations[0].is_critical is True

    def test_geofence_detection(self):
        """Detect boundary breach if drone exceeds arena boundary."""
        monitor = SwarmSafetyMonitor()
        out_of_bounds = {1: np.array([3.0, 0.0, 1.5])}  # 3.0m > x_max (2.5m)
        violations = monitor.check_geofence(out_of_bounds)
        assert len(violations) == 1
        assert violations[0].axis == "X"


class TestFullClosedLoopSystem:
    def test_closed_loop_drift_correction(self):
        """Full simulation of line wave must achieve mean tracking error < 5cm despite drift!"""
        res = run_closed_loop_simulation(task="line", duration=8.0)
        assert res["mean_error"] < 0.05  # Mean error under 5 cm!
        assert np.min(res["min_separations"]) >= 0.40  # Zero collision violations!

