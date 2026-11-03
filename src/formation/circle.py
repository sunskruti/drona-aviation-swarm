"""
================================================================================
Drona Aviation Swarm - Circular Formation Generator
================================================================================
This module implements deterministic coordinate generation for planar circular 
multi-drone swarm formations. It supports arbitrary swarm sizes (N >= 1), 
configurable circle radius (r), arbitrary 3D center coordinates, starting 
angular offsets (theta_0), and clockwise/counter-clockwise ordering.

Mathematical Formulation:
-------------------------
For a swarm of N drones indexed i in {0, 1, ..., N-1}:
Uniform angular subdivision around 2*pi radians:
    delta_theta = (2 * pi) / N
    theta_i = theta_0 + s * i * delta_theta
    where s = -1 if clockwise else +1

Cartesian coordinates in horizontal plane:
    x_i = x_c + r * cos(theta_i)
    y_i = y_c + r * sin(theta_i)
    z_i = z_c

Theoretical Chord Length (inter-drone separation for adjacent drones):
    d_adjacent = 2 * r * sin(pi / N)
    For N = 4 and r = 1.0 m:
    d_adjacent = 2 * 1.0 * sin(pi / 4) = 2 * (sqrt(2) / 2) = sqrt(2) approx 1.4142 m.

Centroid Invariance:
    For any N >= 2, by symmetry of roots of unity:
        sum_{i=0}^{N-1} cos(theta_i) = 0
        sum_{i=0}^{N-1} sin(theta_i) = 0
    Therefore, the center of mass of the drone circle coincides exactly with [x_c, y_c, z_c]!
================================================================================
"""

from typing import List, Tuple, Union
import numpy as np


def generate_circle(
    n: int = 4,
    radius: float = 1.0,
    center: Union[List[float], Tuple[float, float, float], np.ndarray] = (0.0, 0.0, 1.5),
    start_angle_deg: float = 0.0,
    clockwise: bool = False,
) -> np.ndarray:
    """
    Generate target 3D Cartesian coordinates for N drones uniformly distributed 
    on a horizontal circle of specified radius and center altitude.

    Parameters:
    -----------
    n : int
        Number of drones in the circular swarm (N >= 1).
    radius : float
        Radius of the formation circle in meters (r > 0).
    center : array-like of shape (3,)
        World coordinate [x_c, y_c, z_c] defining the center of the circle.
    start_angle_deg : float
        Starting angle for drone 1 (ID 1) in degrees relative to the positive X-axis.
    clockwise : bool
        If True, drones are sequenced clockwise (decreasing angle); if False, counter-clockwise.

    Returns:
    --------
    positions : np.ndarray of shape (n, 3)
        Target coordinates for each drone in the swarm.

    Raises:
    -------
    ValueError: If n < 1 or radius <= 0.
    """
    if n < 1:
        raise ValueError(f"Swarm size n must be >= 1, received: {n}")
    if radius <= 0.0:
        raise ValueError(f"Circle radius must be positive, received: {radius}")

    center_arr = np.asarray(center, dtype=np.float64)
    if center_arr.shape != (3,):
        raise ValueError(f"Center must be a 3D coordinate [x, y, z], received shape {center_arr.shape}")

    theta_0 = np.deg2rad(start_angle_deg)
    direction = -1.0 if clockwise else 1.0
    delta_theta = (2.0 * np.pi) / n

    positions = np.zeros((n, 3), dtype=np.float64)

    for i in range(n):
        angle = theta_0 + direction * (i * delta_theta)
        positions[i, 0] = center_arr[0] + radius * np.cos(angle)
        positions[i, 1] = center_arr[1] + radius * np.sin(angle)
        positions[i, 2] = center_arr[2]

    return positions


def compute_radial_distances(
    positions: np.ndarray,
    center: Union[List[float], Tuple[float, float, float], np.ndarray] = (0.0, 0.0, 1.5)
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute radial distance from center for each drone, the full inter-drone distance matrix,
    and adjacent circular chord lengths (including wraparound from D_N to D_1).

    Parameters:
    -----------
    positions : np.ndarray of shape (n, 3)
        Coordinates of all drones.
    center : array-like of shape (3,)
        Center of the circle.

    Returns:
    --------
    radial_distances : np.ndarray of shape (n,)
        Radial distance of each drone from center in the XY plane.
    dist_matrix : np.ndarray of shape (n, n)
        Full pairwise Euclidean distance matrix.
    circular_adjacent_distances : np.ndarray of shape (n,)
        Distances between circular neighbors: ||p_{i+1} - p_i|| with wraparound.
    """
    positions = np.asarray(positions, dtype=np.float64)
    center_arr = np.asarray(center, dtype=np.float64)
    n = len(positions)

    if n < 1:
        return np.empty((0,)), np.empty((0, 0)), np.empty((0,))

    # In-plane radial distances
    xy_deltas = positions[:, :2] - center_arr[:2]
    radial_distances = np.linalg.norm(xy_deltas, axis=1)

    # Full pairwise distance matrix
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist_matrix = np.linalg.norm(diff, axis=-1)

    # Adjacent chord lengths with circular loop closing
    if n > 1:
        next_indices = np.roll(np.arange(n), -1)  # e.g., [1, 2, 3, 0]
        circular_adjacent_distances = np.linalg.norm(positions[next_indices] - positions, axis=1)
    else:
        circular_adjacent_distances = np.array([0.0], dtype=np.float64)

    return radial_distances, dist_matrix, circular_adjacent_distances


def verify_circle_formation(
    positions: np.ndarray,
    expected_radius: float,
    expected_center: Union[List[float], Tuple[float, float, float], np.ndarray],
    tol: float = 1e-6
) -> bool:
    """
    Validate that the generated circular formation satisfies:
    1. Every drone's horizontal distance from center equals expected_radius within tolerance.
    2. Centroid equals expected_center.
    3. All adjacent chord lengths are equal to 2 * r * sin(pi / N).

    Returns:
    --------
    is_valid : bool
        True if all invariants pass.
    """
    positions = np.asarray(positions, dtype=np.float64)
    expected_center = np.asarray(expected_center, dtype=np.float64)
    n = len(positions)

    if n < 2:
        return True

    rad_dist, _, adj_chords = compute_radial_distances(positions, expected_center)

    # 1. Radius check for all drones
    max_radius_err = np.max(np.abs(rad_dist - expected_radius))
    if max_radius_err > tol:
        print(f"[FAIL] Max radial error {max_radius_err:.2e} exceeds tolerance {tol}")
        return False

    # 2. Centroid check
    actual_centroid = np.mean(positions, axis=0)
    centroid_err = np.linalg.norm(actual_centroid - expected_center)
    if centroid_err > tol:
        print(f"[FAIL] Centroid error {centroid_err:.2e} exceeds tolerance {tol}")
        return False

    # 3. Chord distance check
    expected_chord = 2.0 * expected_radius * np.sin(np.pi / n)
    chord_err = np.max(np.abs(adj_chords - expected_chord))
    if chord_err > tol:
        print(f"[FAIL] Max adjacent chord error {chord_err:.2e} exceeds tolerance {tol}")
        return False

    return True


if __name__ == "__main__":
    print("=" * 70)
    print("DRONA AVIATION SWARM - TESTING CIRCULAR FORMATION GENERATOR")
    print("=" * 70)

    n_drones = 4
    r_circle = 1.0
    c_point = [0.0, 0.0, 1.5]

    coords = generate_circle(n=n_drones, radius=r_circle, center=c_point)
    print(f"\nGenerated circular coordinates for N={n_drones}, radius={r_circle}m, center={c_point}:")
    for idx, pos in enumerate(coords):
        angle_deg = idx * (360.0 / n_drones)
        print(f"  Drone D{idx + 1} (theta = {angle_deg:5.1f} deg): X = {pos[0]:+6.2f} m, Y = {pos[1]:+6.2f} m, Z = {pos[2]:+6.2f} m")

    radii, matrix, chords = compute_radial_distances(coords, c_point)
    expected_chord_len = 2.0 * r_circle * np.sin(np.pi / n_drones)
    print(f"\nMeasured In-Plane Radii: {radii} m")
    print(f"Theoretical Chord Length: {expected_chord_len:.4f} m (sqrt(2) for N=4, r=1m)")
    print(f"Measured Adjacent Chords: {chords} m")

    print("\nPairwise Inter-Drone Euclidean Distance Matrix (meters):")
    print("      D1     D2     D3     D4")
    for r in range(n_drones):
        row_str = f"D{r+1}: " + "  ".join([f"{matrix[r, c]:6.2f}" for c in range(n_drones)])
        print(row_str)

    valid = verify_circle_formation(coords, r_circle, c_point)
    print(f"\nCircular Formation Geometric Invariants Check: {'[PASS]' if valid else '[FAIL]'}")
    print("=" * 70)

