"""
================================================================================
Drona Aviation Swarm - Line Formation Generator
================================================================================
This module implements deterministic coordinate generation for collinear 
multi-drone swarm formations. It supports arbitrary swarm sizes (N >= 1), 
configurable inter-drone spacing (d), arbitrary 3D formation center coordinates, 
and 2D in-plane yaw orientation rotations.

Mathematical Formulation:
-------------------------
For a swarm of N drones indexed i in {0, 1, ..., N-1}:
Centering offset: delta_i = i - (N - 1) / 2
Unrotated displacement along primary axis (X-axis):
    x_local,i = delta_i * d
    y_local,i = 0
    z_local,i = 0

Applying in-plane planar rotation by angle psi (yaw):
    [x_rot,i]   [cos(psi)  -sin(psi)] [x_local,i]
    [y_rot,i] = [sin(psi)   cos(psi)] [y_local,i]

Translating to world center coordinate [x_c, y_c, z_c]:
    p_i = [x_c + x_rot,i,  y_c + y_rot,i,  z_c]

Centroid Invariance:
    (1/N) * sum_{i=0}^{N-1} delta_i = (1/N) * [ sum_{i=0}^{N-1} i - N*(N-1)/2 ]
                                    = (1/N) * [ N*(N-1)/2 - N*(N-1)/2 ] = 0
Therefore, the geometric centroid of generated coordinates identically equals center!
================================================================================
"""

from typing import List, Tuple, Union
import numpy as np


def generate_line(
    n: int = 4,
    spacing: float = 1.0,
    center: Union[List[float], Tuple[float, float, float], np.ndarray] = (0.0, 0.0, 1.5),
    yaw_angle_deg: float = 0.0,
    axis: str = "x",
) -> np.ndarray:
    """
    Generate target 3D Cartesian coordinates for N drones arranged in a straight line.

    Parameters:
    -----------
    n : int
        Number of drones in the swarm (N >= 1).
    spacing : float
        Inter-drone distance in meters between consecutive drones along the line (d > 0).
    center : array-like of shape (3,)
        World coordinate [x_c, y_c, z_c] defining the geometric center of the formation.
    yaw_angle_deg : float
        Counter-clockwise planar rotation angle (yaw) of the line in degrees.
    axis : str
        Base unrotated alignment axis ('x' for lateral alignment, 'y' for longitudinal).

    Returns:
    --------
    positions : np.ndarray of shape (n, 3)
        Array where each row i contains [x_i, y_i, z_i] target world coordinates for Drone i+1.

    Raises:
    -------
    ValueError: If n < 1 or spacing <= 0 or axis not in ('x', 'y').
    """
    if n < 1:
        raise ValueError(f"Swarm size n must be >= 1, received: {n}")
    if spacing <= 0.0:
        raise ValueError(f"Inter-drone spacing must be positive, received: {spacing}")
    if axis not in ("x", "y"):
        raise ValueError(f"Axis must be 'x' or 'y', received: {axis}")

    center_arr = np.asarray(center, dtype=np.float64)
    if center_arr.shape != (3,):
        raise ValueError(f"Center must be a 3D coordinate [x, y, z], received shape {center_arr.shape}")

    # Convert yaw angle from degrees to radians
    yaw_rad = np.deg2rad(yaw_angle_deg)
    cos_yaw = np.cos(yaw_rad)
    sin_yaw = np.sin(yaw_rad)

    # 2D in-plane rotation matrix
    r_matrix = np.array([
        [cos_yaw, -sin_yaw],
        [sin_yaw,  cos_yaw]
    ], dtype=np.float64)

    positions = np.zeros((n, 3), dtype=np.float64)

    for i in range(n):
        # Symmetrical centering index offset: for n=4 -> [-1.5, -0.5, +0.5, +1.5]
        offset = (i - (n - 1) / 2.0) * spacing

        if axis == "x":
            local_xy = np.array([offset, 0.0], dtype=np.float64)
        else:  # axis == 'y'
            local_xy = np.array([0.0, offset], dtype=np.float64)

        # Apply planar rotation
        rot_xy = r_matrix @ local_xy

        # Translate to global world frame
        positions[i, 0] = center_arr[0] + rot_xy[0]
        positions[i, 1] = center_arr[1] + rot_xy[1]
        positions[i, 2] = center_arr[2]

    return positions


def compute_inter_drone_distances(positions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute full Euclidean distance matrix between all pairs of drones, 
    and distance vector between adjacent drones along the line sequence.

    Parameters:
    -----------
    positions : np.ndarray of shape (n, 3)
        Coordinates of all drones in the swarm.

    Returns:
    --------
    dist_matrix : np.ndarray of shape (n, n)
        Symmetric Euclidean distance matrix where dist_matrix[i, j] = ||p_i - p_j||.
    adjacent_distances : np.ndarray of shape (n - 1,)
        Distances between adjacent drones: ||p_{i+1} - p_i|| for i = 0 ... n-2.
    """
    positions = np.asarray(positions, dtype=np.float64)
    n = len(positions)
    if n < 1:
        return np.empty((0, 0)), np.empty((0,))

    # Efficient pairwise vector subtraction using broadcasting: (N, 1, 3) - (1, N, 3)
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist_matrix = np.linalg.norm(diff, axis=-1)

    if n > 1:
        adjacent_distances = np.linalg.norm(positions[1:] - positions[:-1], axis=-1)
    else:
        adjacent_distances = np.array([], dtype=np.float64)

    return dist_matrix, adjacent_distances


def verify_line_formation(
    positions: np.ndarray,
    expected_spacing: float,
    expected_center: Union[List[float], Tuple[float, float, float], np.ndarray],
    tol: float = 1e-6
) -> bool:
    """
    Validate that the generated line formation strictly satisfies:
    1. Uniform spacing between all consecutive drones equal to expected_spacing.
    2. Mean centroid exactly matches expected_center.
    3. Perfect collinearity (rank-1 planar span).

    Returns:
    --------
    is_valid : bool
        True if all geometric invariants hold within tolerance.
    """
    positions = np.asarray(positions, dtype=np.float64)
    expected_center = np.asarray(expected_center, dtype=np.float64)
    n = len(positions)

    if n < 2:
        return True

    # 1. Centroid check
    actual_centroid = np.mean(positions, axis=0)
    centroid_err = np.linalg.norm(actual_centroid - expected_center)
    if centroid_err > tol:
        print(f"[FAIL] Centroid error {centroid_err:.2e} exceeds tolerance {tol}")
        return False

    # 2. Adjacent spacing check
    _, adj_dist = compute_inter_drone_distances(positions)
    spacing_err = np.max(np.abs(adj_dist - expected_spacing))
    if spacing_err > tol:
        print(f"[FAIL] Max adjacent spacing error {spacing_err:.2e} exceeds tolerance {tol}")
        return False

    # 3. Collinearity check: difference vectors between adjacent points must be parallel
    deltas = positions[1:] - positions[:-1]
    unit_direction = deltas[0] / np.linalg.norm(deltas[0])
    for k in range(1, len(deltas)):
        proj = np.dot(deltas[k], unit_direction)
        if not np.isclose(proj, expected_spacing, atol=tol):
            print(f"[FAIL] Collinearity violation at drone pair {k}->{k+1}")
            return False

    return True


if __name__ == "__main__":
    print("=" * 70)
    print("DRONA AVIATION SWARM - TESTING LINE FORMATION GENERATOR")
    print("=" * 70)

    n_drones = 4
    d_spacing = 1.0
    c_point = [0.0, 0.0, 1.5]

    coords = generate_line(n=n_drones, spacing=d_spacing, center=c_point)
    print(f"\nGenerated line coordinates for N={n_drones}, spacing={d_spacing}m, center={c_point}:")
    for idx, pos in enumerate(coords):
        print(f"  Drone D{idx + 1} (ID: {idx + 1}): X = {pos[0]:+6.2f} m, Y = {pos[1]:+6.2f} m, Z = {pos[2]:+6.2f} m")

    matrix, adj_d = compute_inter_drone_distances(coords)
    print(f"\nAdjacent drone distances (d_{{i, i+1}}): {adj_d} m")
    print("\nPairwise Inter-Drone Euclidean Distance Matrix (meters):")
    print("      D1     D2     D3     D4")
    for r in range(n_drones):
        row_str = f"D{r+1}: " + "  ".join([f"{matrix[r, c]:6.2f}" for c in range(n_drones)])
        print(row_str)

    valid = verify_line_formation(coords, d_spacing, c_point)
    print(f"\nFormation Geometric Invariants Check: {'[PASS]' if valid else '[FAIL]'}")
    print("=" * 70)

