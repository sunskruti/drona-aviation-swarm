"""
================================================================================
Drona Aviation Swarm - Trajectory & Traveling Wave Visualizer
================================================================================
Generates publication plots verifying:
1. Z-axis altitude vs time for all 4 drones showing progressive phase delays.
2. 3D spatial trajectory ribbons showing smooth synchronized flight path.
3. Dynamic inter-drone separation distance over the entire execution timeline.
Saves figure to plots/trajectory_wave_verification.png.
================================================================================
"""

import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Ensure src path is accessible
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from formation.line import generate_line
from formation.circle import generate_circle
from trajectory.wave_generator import LineWaveTrajectory, CircularWaveTrajectory


def plot_wave_trajectories(output_filepath: str = "plots/trajectory_wave_verification.png"):
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)

    # Base formations
    line_bases = generate_line(n=4, spacing=1.0, center=[0.0, 0.0, 1.5])
    circle_bases = generate_circle(n=4, radius=1.0, center=[0.0, 0.0, 1.5])

    line_wave = LineWaveTrajectory(base_positions=line_bases, amplitude=0.25, frequency=0.25, num_cycles=2)
    circle_wave = CircularWaveTrajectory(base_positions=circle_bases, amplitude=0.25, frequency=0.25, num_cycles=2)

    total_time = line_wave.total_duration
    time_steps = np.linspace(0, total_time, 400)

    # Collect trajectory histories
    line_history = []
    circle_history = []
    for t in time_steps:
        pos_l, _, _ = line_wave.get_swarm_state(t)
        pos_c, _, _ = circle_wave.get_swarm_state(t)
        line_history.append(pos_l)
        circle_history.append(pos_c)

    line_history = np.array(line_history)     # (T, 4, 3)
    circle_history = np.array(circle_history) # (T, 4, 3)

    fig = plt.figure(figsize=(16, 12), dpi=200)
    fig.patch.set_facecolor("#0F172A")
    colors = ["#38BDF8", "#34D399", "#FBBF24", "#F43F5E"]
    labels = ["Drone 1 (0°)", "Drone 2 (90°)", "Drone 3 (180°)", "Drone 4 (270°)"]

    # --------------------------------------------------------------------------
    # Subplot 1: Line Wave Altitude vs Time (Task 1 Phase Progression)
    # --------------------------------------------------------------------------
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.set_facecolor("#1E293B")
    ax1.grid(True, linestyle="--", alpha=0.3, color="#94A3B8")

    for i in range(4):
        ax1.plot(time_steps, line_history[:, i, 2], color=colors[i], linewidth=2.5, label=labels[i])

    ax1.set_title("Task 1: Line Formation - Vertical Traveling Wave (Z vs Time)", color="white", fontsize=12, pad=10)
    ax1.set_xlabel("Time (seconds)", color="#94A3B8")
    ax1.set_ylabel("Altitude Z (meters)", color="#94A3B8")
    ax1.tick_params(colors="#94A3B8")
    ax1.set_ylim(1.15, 1.85)
    ax1.legend(loc="upper right", facecolor="#334155", edgecolor="none", labelcolor="white")

    # --------------------------------------------------------------------------
    # Subplot 2: Circular Wave Altitude vs Time (Task 2 Phase Progression)
    # --------------------------------------------------------------------------
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.set_facecolor("#1E293B")
    ax2.grid(True, linestyle="--", alpha=0.3, color="#94A3B8")

    for i in range(4):
        ax2.plot(time_steps, circle_history[:, i, 2], color=colors[i], linewidth=2.5, label=labels[i])

    ax2.set_title("Task 2: Circular Formation - Traveling Wave (Z vs Time)", color="white", fontsize=12, pad=10)
    ax2.set_xlabel("Time (seconds)", color="#94A3B8")
    ax2.set_ylabel("Altitude Z (meters)", color="#94A3B8")
    ax2.tick_params(colors="#94A3B8")
    ax2.set_ylim(1.15, 1.85)
    ax2.legend(loc="upper right", facecolor="#334155", edgecolor="none", labelcolor="white")

    # --------------------------------------------------------------------------
    # Subplot 3: 3D Spatial Trajectory Ribbon - Line Wave
    # --------------------------------------------------------------------------
    ax3 = fig.add_subplot(2, 2, 3, projection="3d")
    ax3.set_facecolor("#1E293B")

    for i in range(4):
        ax3.plot(line_history[:, i, 0], line_history[:, i, 1], line_history[:, i, 2],
                 color=colors[i], linewidth=2.5, label=f"D{i+1}")
        # Scatter current endpoint
        ax3.scatter(line_history[-1, i, 0], line_history[-1, i, 1], line_history[-1, i, 2],
                    color=colors[i], s=80)

    ax3.set_title("3D Space-Time Trajectory: Line Formation Wave", color="white", fontsize=12, pad=10)
    ax3.set_xlabel("X (m)", color="#94A3B8")
    ax3.set_ylabel("Y (m)", color="#94A3B8")
    ax3.set_zlabel("Altitude Z (m)", color="#94A3B8")
    ax3.tick_params(colors="#94A3B8")
    ax3.set_zlim(1.0, 2.0)

    # --------------------------------------------------------------------------
    # Subplot 4: Dynamic Minimum Separation Over Time (Collision Validation)
    # --------------------------------------------------------------------------
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.set_facecolor("#1E293B")
    ax4.grid(True, linestyle="--", alpha=0.3, color="#94A3B8")

    # Compute min pairwise separation at each time step
    min_dist_line = []
    min_dist_circle = []
    for t_idx in range(len(time_steps)):
        # Line pairwise distances
        diff_l = line_history[t_idx][:, None, :] - line_history[t_idx][None, :, :]
        dist_l = np.linalg.norm(diff_l, axis=-1)
        np.fill_diagonal(dist_l, np.inf)
        min_dist_line.append(np.min(dist_l))

        # Circle pairwise distances
        diff_c = circle_history[t_idx][:, None, :] - circle_history[t_idx][None, :, :]
        dist_c = np.linalg.norm(diff_c, axis=-1)
        np.fill_diagonal(dist_c, np.inf)
        min_dist_circle.append(np.min(dist_c))

    ax4.plot(time_steps, min_dist_line, color="#38BDF8", linewidth=2.2, label="Line Wave Min Separation")
    ax4.plot(time_steps, min_dist_circle, color="#34D399", linewidth=2.2, label="Circular Wave Min Separation")
    ax4.axhline(0.40, color="#EF4444", linestyle="--", linewidth=2, label="Safety Threshold (d_min = 0.4m)")

    ax4.set_title("Safety Verification: Minimum Swarm Separation", color="white", fontsize=12, pad=10)
    ax4.set_xlabel("Time (seconds)", color="#94A3B8")
    ax4.set_ylabel("Minimum Distance (meters)", color="#94A3B8")
    ax4.tick_params(colors="#94A3B8")
    ax4.set_ylim(0.0, 2.0)
    ax4.legend(loc="lower right", facecolor="#334155", edgecolor="none", labelcolor="white")

    plt.suptitle("Drona Aviation Swarm - Synchronized 4-Drone Wave Execution Verification",
                 color="white", fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig(output_filepath, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"[SUCCESS] Trajectory verification figure saved to: {output_filepath}")


if __name__ == "__main__":
    plot_wave_trajectories()

