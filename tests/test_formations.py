"""
================================================================================
Unit Tests for Multi-Drone Formation Generators
================================================================================
Covers:
- Line formation coordinates, centering, spacing, collinearity, and rotations.
- Circular formation coordinates, radial constancy, chord distances, and symmetry.
- Boundary conditions, error handling, and metric matrix symmetry.
================================================================================
"""

import sys
import os
import pytest
import numpy as np

# Ensure src path is accessible
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from formation.line import (
    generate_line,
    compute_inter_drone_distances,
    verify_line_formation,
)
from formation.circle import (
    generate_circle,
    compute_radial_distances,
    verify_circle_formation,
)


class TestLineFormation:
    def test_default_four_drones(self):
        """Verify the standard 4-drone line formation specified in the problem statement."""
        positions = generate_line(n=4, spacing=1.0, center=[0.0, 0.0, 1.5])
        expected = np.array([
            [-1.5, 0.0, 1.5],
            [-0.5, 0.0, 1.5],
            [ 0.5, 0.0, 1.5],
            [ 1.5, 0.0, 1.5],
        ])
        np.testing.assert_allclose(positions, expected, atol=1e-7)

    def test_inter_drone_spacing_and_matrix(self):
        """Verify adjacent spacing is uniform and distance matrix is symmetric."""
        spacing = 0.8
        positions = generate_line(n=4, spacing=spacing, center=[1.0, -2.0, 1.2])
        dist_mat, adj_dists = compute_inter_drone_distances(positions)

        # All adjacent distances must strictly equal spacing
        np.testing.assert_allclose(adj_dists, [spacing, spacing, spacing], atol=1e-7)

        # Distance matrix must be symmetric with zero diagonal
        np.testing.assert_allclose(dist_mat, dist_mat.T, atol=1e-7)
        np.testing.assert_allclose(np.diag(dist_mat), np.zeros(4), atol=1e-7)

        # Verify distance between D1 and D4 is 3 * spacing
        assert np.isclose(dist_mat[0, 3], 3 * spacing, atol=1e-7)

    def test_centroid_preservation(self):
        """Geometric centroid must identically match the requested center."""
        center = [2.5, -1.8, 1.75]
        for n in [2, 3, 4, 5, 8]:
            positions = generate_line(n=n, spacing=0.6, center=center)
            actual_centroid = np.mean(positions, axis=0)
            np.testing.assert_allclose(actual_centroid, center, atol=1e-7)

    def test_yaw_rotation(self):
        """Verify 90-degree yaw rotation aligns along the Y-axis."""
        positions = generate_line(n=4, spacing=1.0, center=[0.0, 0.0, 1.5], yaw_angle_deg=90.0)
        expected = np.array([
            [0.0, -1.5, 1.5],
            [0.0, -0.5, 1.5],
            [0.0,  0.5, 1.5],
            [0.0,  1.5, 1.5],
        ])
        np.testing.assert_allclose(positions, expected, atol=1e-7)

    def test_invalid_parameters(self):
        """Ensure invalid inputs raise ValueError."""
        with pytest.raises(ValueError):
            generate_line(n=0)
        with pytest.raises(ValueError):
            generate_line(spacing=-0.5)
        with pytest.raises(ValueError):
            generate_line(axis="z")


class TestCircleFormation:
    def test_default_four_drones(self):
        """Verify cardinal positions for 4 drones on a 1m radius circle."""
        positions = generate_circle(n=4, radius=1.0, center=[0.0, 0.0, 1.5], start_angle_deg=0.0)
        expected = np.array([
            [ 1.0,  0.0, 1.5],  # D1: 0 deg
            [ 0.0,  1.0, 1.5],  # D2: 90 deg
            [-1.0,  0.0, 1.5],  # D3: 180 deg
            [ 0.0, -1.0, 1.5],  # D4: 270 deg
        ])
        np.testing.assert_allclose(positions, expected, atol=1e-7)

    def test_radial_distance_constancy(self):
        """Every drone must be exactly radius distance from center in XY."""
        r = 1.25
        center = [0.5, -0.5, 1.4]
        positions = generate_circle(n=4, radius=r, center=center)
        radii, dist_mat, adj_chords = compute_radial_distances(positions, center=center)

        np.testing.assert_allclose(radii, np.full(4, r), atol=1e-7)

        # Theoretical adjacent chord length for N=4: 2 * r * sin(pi/4) = r * sqrt(2)
        expected_chord = r * np.sqrt(2.0)
        np.testing.assert_allclose(adj_chords, np.full(4, expected_chord), atol=1e-7)

        # Opposite drones separation: 2 * r (diameter)
        assert np.isclose(dist_mat[0, 2], 2.0 * r, atol=1e-7)
        assert np.isclose(dist_mat[1, 3], 2.0 * r, atol=1e-7)

    def test_circle_centroid_preservation(self):
        """Circular swarm centroid must match the center."""
        center = [-1.0, 2.0, 1.6]
        for n in [3, 4, 6, 8]:
            positions = generate_circle(n=n, radius=0.9, center=center)
            actual_centroid = np.mean(positions, axis=0)
            np.testing.assert_allclose(actual_centroid, center, atol=1e-7)

    def test_circle_invariants_verifier(self):
        """Test the automated geometric validator."""
        positions = generate_circle(n=4, radius=1.0, center=[0.0, 0.0, 1.5])
        assert verify_circle_formation(positions, expected_radius=1.0, expected_center=[0.0, 0.0, 1.5])

    def test_invalid_parameters(self):
        """Ensure invalid inputs raise ValueError."""
        with pytest.raises(ValueError):
            generate_circle(n=-1)
        with pytest.raises(ValueError):
            generate_circle(radius=0.0)

