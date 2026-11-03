"""
================================================================================
Drona Aviation Swarm - Coordinate Transformations & Calibration
================================================================================
Implements SE(3) Euclidean rigid body coordinate transformations between:
- Camera Frame {C} (Overhead vision sensor coordinate system)
- World Frame {W} (Arena origin: ENU - East, North, Up)
- Drone Body Frame {B} (Center of mass of individual Pluto UAV)

Mathematical Formulation:
-------------------------
Homogeneous transformation matrix T in SE(3):
    T_C^W = [ R_C^W    t_C^W ]
            [  0^T       1   ]

Transforming a 3D point measured in camera frame p^C to world coordinates p^W:
    p^W = R_C^W * p^C + t_C^W

Euler Angle Rotation Matrix (Yaw-Pitch-Roll in Z-Y-X convention):
    R = R_z(psi) * R_y(theta) * R_x(phi)
================================================================================
"""

from typing import Tuple, Union
import numpy as np


def euler_to_rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    Compute 3x3 rotation matrix from Euler angles (radians, ZYX convention).
    """
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    R_x = np.array([
        [1.0, 0.0, 0.0],
        [0.0, cr, -sr],
        [0.0, sr,  cr]
    ])

    R_y = np.array([
        [cp,  0.0, sp],
        [0.0, 1.0, 0.0],
        [-sp, 0.0, cp]
    ])

    R_z = np.array([
        [cy, -sy, 0.0],
        [sy,  cy, 0.0],
        [0.0, 0.0, 1.0]
    ])

    return R_z @ R_y @ R_x


class CoordinateTransformer:
    """
    Transforms coordinates between external vision camera frame and arena world frame.
    """

    def __init__(
        self,
        rotation_matrix: np.ndarray = None,
        translation_vector: np.ndarray = None
    ):
        """
        Initialize with extrinsic rotation R_C^W and translation t_C^W.
        Default is identity rotation with overhead camera mounted at [0, 0, 3.0] looking downward.
        """
        if rotation_matrix is None:
            # Typical overhead camera looking down:
            # Camera Z points down (-World Z), Camera X points right (+World X), Camera Y points down-arena (-World Y)
            self.R_cam_to_world = np.array([
                [1.0,  0.0,  0.0],
                [0.0, -1.0,  0.0],
                [0.0,  0.0, -1.0]
            ], dtype=np.float64)
        else:
            self.R_cam_to_world = np.asarray(rotation_matrix, dtype=np.float64)

        if translation_vector is None:
            # Camera optical center located at 3.0m ceiling above arena origin
            self.t_cam_to_world = np.array([0.0, 0.0, 3.0], dtype=np.float64)
        else:
            self.t_cam_to_world = np.asarray(translation_vector, dtype=np.float64)

        # Precompute inverse transformation
        self.R_world_to_cam = self.R_cam_to_world.T
        self.t_world_to_cam = -self.R_world_to_cam @ self.t_cam_to_world

    def camera_to_world(self, p_cam: np.ndarray) -> np.ndarray:
        """Transform 3D point from Camera Frame to World Frame."""
        p_c = np.asarray(p_cam, dtype=np.float64)
        return self.R_cam_to_world @ p_c + self.t_cam_to_world

    def world_to_camera(self, p_world: np.ndarray) -> np.ndarray:
        """Transform 3D point from World Frame to Camera Frame."""
        p_w = np.asarray(p_world, dtype=np.float64)
        return self.R_world_to_cam @ p_w + self.t_world_to_cam

