"""
Trajectory generation module for synchronized multi-drone formations.
Includes minimum-jerk polynomial interpolation and phase-shifted traveling wave dynamics.
"""
from .trajectory_generator import (
    quintic_polynomial_scaling,
    generate_point_to_point_trajectory,
    SwarmTrajectoryPlanner,
)
from .wave_generator import (
    generate_wave_state,
    LineWaveTrajectory,
    CircularWaveTrajectory,
)

__all__ = [
    "quintic_polynomial_scaling",
    "generate_point_to_point_trajectory",
    "SwarmTrajectoryPlanner",
    "generate_wave_state",
    "LineWaveTrajectory",
    "CircularWaveTrajectory",
]

