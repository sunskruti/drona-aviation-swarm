"""
Formation generation module for straight-line and circular multi-drone geometries.
Provides coordinate transformations, distance matrix validation, and centering.
"""
from .line import generate_line, compute_inter_drone_distances
from .circle import generate_circle, compute_radial_distances

__all__ = [
    "generate_line",
    "compute_inter_drone_distances",
    "generate_circle",
    "compute_radial_distances",
]

