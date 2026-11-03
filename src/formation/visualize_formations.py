"""
================================================================================
Drona Aviation Swarm - Formation Visualizer
================================================================================
Generates publication-quality 2D overhead and 3D spatial plots of:
1. Four-Drone Straight-Line Formation (Task 1 base geometry)
2. Four-Drone Circular Formation (Task 2 base geometry)
Saves resulting figures to plots/formation_geometry.png.
================================================================================
"""

import os
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from line import generate_line, compute_inter_drone_distances
from circle import generate_circle, compute_radial_distances


def plot_swarm_formations(output_filepath: str = "plots/formation_geometry.png"):
    """
    Generate side-by-side 2D and 3D visualizations for Line and Circle formations.
    """
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)

    # 1. Compute Formations
    n = 4
    line_spacing = 1.0
    circle_radius = 1.0
    center = [0.0, 0.0, 1.5]

    line_pos = generate_line(n=n, spacing=line_spacing, center=center)
    circle_pos = generate_circle(n=n, radius=circle_radius, center=center)

    fig = plt.figure(figsize=(15, 12), dpi=200)
    fig.patch.set_facecolor("#0F172A")  # Dark slate background

    colors = ["#38BDF8", "#34D399", "#FBBF24", "#F43F5E"]  # D1 to D4 distinct hex
    drone_labels = [f"D{i+1}" for i in range(n)]

    # --------------------------------------------------------------------------
    # Subplot 1: 2D Line Formation (Overhead XY)
    # --------------------------------------------------------------------------
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.set_facecolor("#1E293B")
    ax1.grid(True, linestyle="--", alpha=0.3, color="#94A3B8")

    # Connect line
    ax1.plot(line_pos[:, 0], line_pos[:, 1], color="#64748B", linestyle="-", linewidth=2.5, zorder=1)

    # Plot drones
    for i in range(n):
        ax1.scatter(line_pos[i, 0], line_pos[i, 1], s=250, color=colors[i], edgecolors="white", linewidth=2, zorder=3)
        ax1.text(line_pos[i, 0], line_pos[i, 1] + 0.12, drone_labels[i], color="white", fontsize=11,
                 fontweight="bold", ha="center", va="bottom")

    # Center marker
    ax1.scatter(center[0], center[1], color="#EF4444", marker="x", s=100, linewidth=2.5, label="Swarm Centroid")

    ax1.set_title(f"Task 1: Line Formation (2D Overhead, d = {line_spacing} m)", color="white", fontsize=13, pad=12)
    ax1.set_xlabel("X World Position (meters)", color="#94A3B8")
    ax1.set_ylabel("Y World Position (meters)", color="#94A3B8")
    ax1.tick_params(colors="#94A3B8")
    ax1.axis("equal")
    ax1.set_xlim(-2.2, 2.2)
    ax1.set_ylim(-1.5, 1.5)
    ax1.legend(loc="upper right", facecolor="#334155", edgecolor="none", labelcolor="white")

    # --------------------------------------------------------------------------
    # Subplot 2: 2D Circular Formation (Overhead XY)
    # --------------------------------------------------------------------------
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.set_facecolor("#1E293B")
    ax2.grid(True, linestyle="--", alpha=0.3, color="#94A3B8")

    # Perimeter circle trace
    theta_sweep = np.linspace(0, 2 * np.pi, 200)
    ax2.plot(center[0] + circle_radius * np.cos(theta_sweep),
             center[1] + circle_radius * np.sin(theta_sweep),
             color="#64748B", linestyle=":", linewidth=2, label=f"Circle (r={circle_radius}m)")

    # Chord lines connecting consecutive drones
    closed_circle_pos = np.vstack([circle_pos, circle_pos[0]])
    ax2.plot(closed_circle_pos[:, 0], closed_circle_pos[:, 1], color="#475569", linestyle="--", linewidth=1.5, label="Inter-drone chords")

    for i in range(n):
        ax2.scatter(circle_pos[i, 0], circle_pos[i, 1], s=250, color=colors[i], edgecolors="white", linewidth=2, zorder=3)
        ax2.text(circle_pos[i, 0] * 1.18, circle_pos[i, 1] * 1.18, drone_labels[i], color="white",
                 fontsize=11, fontweight="bold", ha="center", va="center")

    ax2.scatter(center[0], center[1], color="#EF4444", marker="x", s=100, linewidth=2.5, label="Swarm Centroid")

    ax2.set_title(f"Task 2: Circular Formation (2D Overhead, r = {circle_radius} m)", color="white", fontsize=13, pad=12)
    ax2.set_xlabel("X World Position (meters)", color="#94A3B8")
    ax2.set_ylabel("Y World Position (meters)", color="#94A3B8")
    ax2.tick_params(colors="#94A3B8")
    ax2.axis("equal")
    ax2.set_xlim(-1.8, 1.8)
    ax2.set_ylim(-1.8, 1.8)
    ax2.legend(loc="upper right", facecolor="#334155", edgecolor="none", labelcolor="white")

    # --------------------------------------------------------------------------
    # Subplot 3: 3D Spatial Line Formation
    # --------------------------------------------------------------------------
    ax3 = fig.add_subplot(2, 2, 3, projection="3d")
    ax3.set_facecolor("#1E293B")

    ax3.plot(line_pos[:, 0], line_pos[:, 1], line_pos[:, 2], color="#38BDF8", linewidth=2.5)
    for i in range(n):
        ax3.scatter(line_pos[i, 0], line_pos[i, 1], line_pos[i, 2], s=120, color=colors[i], depthshade=False)
        ax3.text(line_pos[i, 0], line_pos[i, 1], line_pos[i, 2] + 0.1, drone_labels[i], color="white", fontweight="bold")

    ax3.set_title("3D Perspective - Line Formation", color="white", fontsize=12, pad=10)
    ax3.set_xlabel("X (m)", color="#94A3B8")
    ax3.set_ylabel("Y (m)", color="#94A3B8")
    ax3.set_zlabel("Z (Altitude m)", color="#94A3B8")
    ax3.tick_params(colors="#94A3B8")
    ax3.set_zlim(0.0, 2.5)

    # --------------------------------------------------------------------------
    # Subplot 4: 3D Spatial Circular Formation
    # --------------------------------------------------------------------------
    ax4 = fig.add_subplot(2, 2, 4, projection="3d")
    ax4.set_facecolor("#1E293B")

    ax4.plot(closed_circle_pos[:, 0], closed_circle_pos[:, 1], closed_circle_pos[:, 2], color="#34D399", linewidth=2.5)
    for i in range(n):
        ax4.scatter(circle_pos[i, 0], circle_pos[i, 1], circle_pos[i, 2], s=120, color=colors[i], depthshade=False)
        ax4.text(circle_pos[i, 0], circle_pos[i, 1], circle_pos[i, 2] + 0.1, drone_labels[i], color="white", fontweight="bold")

    ax4.set_title("3D Perspective - Circular Formation", color="white", fontsize=12, pad=10)
    ax4.set_xlabel("X (m)", color="#94A3B8")
    ax4.set_ylabel("Y (m)", color="#94A3B8")
    ax4.set_zlabel("Z (Altitude m)", color="#94A3B8")
    ax4.tick_params(colors="#94A3B8")
    ax4.set_zlim(0.0, 2.5)

    plt.suptitle("Drona Aviation Swarm - Multi-UAV Geometry Verification (Inter IIT 15.0)",
                 color="white", fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig(output_filepath, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"[SUCCESS] High-resolution visualization saved to: {output_filepath}")


if __name__ == "__main__":
    plot_swarm_formations()

