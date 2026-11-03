"""
================================================================================
Unit Tests for Trajectory Generation and Wave Dynamics
================================================================================
"""

import sys
import os
import pytest
import numpy as np

# Ensure src path is accessible
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from formation.line import generate_line
from formation.circle import generate_circle
from trajectory.trajectory_generator import (
    quintic_polynomial_scaling,
    generate_point_to_point_trajectory,
    SwarmTrajectoryPlanner,
)
from trajectory.wave_generator import (
    generate_wave_state,
    LineWaveTrajectory,
    CircularWaveTrajectory,
)


class TestQuinticPolynomial:
    def test_boundary_conditions(self):
        """Test s(0)=0, s(T)=1, s_dot(0)=0, s_dot(T)=0, s_ddot(0)=0, s_ddot(T)=0."""
        duration = 5.0
        s_0, s_dot_0, s_ddot_0 = quintic_polynomial_scaling(0.0, duration)
        assert np.isclose(s_0, 0.0)
        assert np.isclose(s_dot_0, 0.0)
        assert np.isclose(s_ddot_0, 0.0)

        s_T, s_dot_T, s_ddot_T = quintic_polynomial_scaling(duration, duration)
        assert np.isclose(s_T, 1.0)
        assert np.isclose(s_dot_T, 0.0)
        assert np.isclose(s_ddot_T, 0.0)

    def test_numerical_derivative_consistency(self):
        """Verify analytical velocity matches finite difference of position."""
        duration = 4.0
        dt = 1e-5
        times = np.linspace(0.5, 3.5, 10)
        for t in times:
            s_plus, _, _ = quintic_polynomial_scaling(t + dt, duration)
            s_minus, _, _ = quintic_polynomial_scaling(t - dt, duration)
            numerical_s_dot = (s_plus - s_minus) / (2.0 * dt)

            _, analytical_s_dot, _ = quintic_polynomial_scaling(t, duration)
            assert np.isclose(numerical_s_dot, analytical_s_dot, atol=1e-4)


class TestPointToPointTrajectory:
    def test_spatial_endpoints(self):
        """Drone should start at p0 and arrive precisely at pf."""
        p0 = np.array([0.0, 0.0, 0.0])
        pf = np.array([1.5, -2.0, 1.8])
        duration = 6.0

        pos_0, vel_0, acc_0 = generate_point_to_point_trajectory(p0, pf, 0.0, duration)
        np.testing.assert_allclose(pos_0, p0, atol=1e-7)
        np.testing.assert_allclose(vel_0, np.zeros(3), atol=1e-7)

        pos_T, vel_T, acc_T = generate_point_to_point_trajectory(p0, pf, duration, duration)
        np.testing.assert_allclose(pos_T, pf, atol=1e-7)
        np.testing.assert_allclose(vel_T, np.zeros(3), atol=1e-7)


class TestSwarmWaveDynamics:
    def test_line_wave_horizontal_invariance(self):
        """Horizontal coordinates (X, Y) must remain completely stationary during the wave."""
        line_bases = generate_line(n=4, spacing=1.0, center=[0.0, 0.0, 1.5])
        wave = LineWaveTrajectory(base_positions=line_bases, amplitude=0.25, frequency=0.25)

        for t in np.linspace(0, wave.total_duration, 20):
            pos, _, _ = wave.get_swarm_state(t)
            # Compare X and Y components with base positions
            np.testing.assert_allclose(pos[:, :2], line_bases[:, :2], atol=1e-7)

    def test_wave_phase_delays(self):
        """For N=4, phase delay between consecutive drones must be pi/2 (90 degrees)."""
        line_bases = generate_line(n=4, spacing=1.0, center=[0.0, 0.0, 1.5])
        wave = LineWaveTrajectory(base_positions=line_bases, amplitude=0.25, frequency=0.25)
        expected_phases = np.array([0.0, np.pi / 2, np.pi, 3.0 * np.pi / 2])
        np.testing.assert_allclose(wave.phase_offsets, expected_phases, atol=1e-7)

    def test_separation_safety_during_wave(self):
        """Minimum distance between any pair of drones must strictly remain >= 0.40m."""
        line_bases = generate_line(n=4, spacing=0.8, center=[0.0, 0.0, 1.5])
        circle_bases = generate_circle(n=4, radius=1.0, center=[0.0, 0.0, 1.5])

        line_wave = LineWaveTrajectory(base_positions=line_bases, amplitude=0.30, frequency=0.30)
        circle_wave = CircularWaveTrajectory(base_positions=circle_bases, amplitude=0.30, frequency=0.30)

        for t in np.linspace(0, line_wave.total_duration, 50):
            # Line check
            pos_l, _, _ = line_wave.get_swarm_state(t)
            diff_l = pos_l[:, None, :] - pos_l[None, :, :]
            dist_l = np.linalg.norm(diff_l, axis=-1)
            np.fill_diagonal(dist_l, np.inf)
            assert np.min(dist_l) >= 0.40

            # Circle check
            pos_c, _, _ = circle_wave.get_swarm_state(t)
            diff_c = pos_c[:, None, :] - pos_c[None, :, :]
            dist_c = np.linalg.norm(diff_c, axis=-1)
            np.fill_diagonal(dist_c, np.inf)
            assert np.min(dist_c) >= 0.40

