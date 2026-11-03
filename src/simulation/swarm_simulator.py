"""
================================================================================
Drona Aviation Swarm - Multi-UAV Closed-Loop Simulator
================================================================================
Simulates complete 4-drone swarm execution under:
- Task 1: Line formation + Traveling wave
- Task 2: Circular formation + Traveling wave
- Uncompensated physical drift (without control -> drift diverges)
- External vision tracking with Gaussian sensor noise
- Closed-loop PID drift correction running at 50 Hz
- Continuous separation and safety monitoring
================================================================================
"""

import os
import sys
from typing import Dict, List
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Ensure src path is accessible
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from formation.line import generate_line
from formation.circle import generate_circle
from trajectory.wave_generator import LineWaveTrajectory, CircularWaveTrajectory
from localization.position_tracker import SwarmPositionTracker
from controller.swarm_controller import SwarmController
from safety.collision_detection import SwarmSafetyMonitor
from simulation.drone_model import SimulatedPlutoDrone


def run_closed_loop_simulation(task: str = "line", duration: float = 12.0, dt: float = 0.02):
    """
    Run closed-loop swarm simulation for either 'line' or 'circle' task.
    """
    drone_ids = [1, 2, 3, 4]

    # 1. Base formation and trajectory
    if task == "line":
        base_positions = generate_line(n=4, spacing=1.0, center=[0.0, 0.0, 1.5])
        trajectory = LineWaveTrajectory(base_positions=base_positions, amplitude=0.25, frequency=0.25)
    else:
        base_positions = generate_circle(n=4, radius=1.0, center=[0.0, 0.0, 1.5])
        trajectory = CircularWaveTrajectory(base_positions=base_positions, amplitude=0.25, frequency=0.25)

    # 2. Instantiate simulated physical drones (with deliberate drift and noise)
    drones = {
        d_id: SimulatedPlutoDrone(
            drone_id=d_id,
            initial_position=base_positions[idx],
            sensor_noise_std=0.012  # 1.2 cm camera noise
        )
        for idx, d_id in enumerate(drone_ids)
    }

    # 3. Instantiate tracker, controller, and safety monitor
    tracker = SwarmPositionTracker(drone_ids=drone_ids)
    controller = SwarmController(drone_ids=drone_ids)
    safety_monitor = SwarmSafetyMonitor()

    # Data logging arrays
    timestamps = []
    tracking_errors = []
    min_separations = []
    actual_trajectories = {d_id: [] for d_id in drone_ids}
    target_trajectories = {d_id: [] for d_id in drone_ids}

    steps = int(duration / dt)
    current_time = 0.0

    for step in range(steps):
        # A. Trajectory generation (Target state)
        target_pos, target_vel, _ = trajectory.get_swarm_state(current_time)

        # B. Vision measurement and tracking (with noise)
        for idx, d_id in enumerate(drone_ids):
            raw_meas = drones[d_id].get_measured_position()
            tracker.process_raw_measurement(d_id, raw_meas, current_time)

        filtered_positions = tracker.get_swarm_positions()

        # C. Closed-Loop Controller update (PID + Feedforward)
        commands, mean_err, max_err = controller.update(
            target_positions=target_pos,
            actual_positions=filtered_positions,
            target_velocities=target_vel,
            current_time=current_time
        )

        # D. Advance physical drone models
        for d_id in drone_ids:
            u_cmd = commands.get(d_id, np.zeros(3))
            true_pos = drones[d_id].step(u_cmd, dt)
            actual_trajectories[d_id].append(true_pos.copy())

        # E. Safety check
        violations = safety_monitor.check_swarm_separation(filtered_positions)
        geofence_viol = safety_monitor.check_geofence(filtered_positions)

        # Compute minimum pairwise separation
        pos_list = np.array([filtered_positions[d_id] for d_id in drone_ids])
        diff = pos_list[:, None, :] - pos_list[None, :, :]
        dist = np.linalg.norm(diff, axis=-1)
        np.fill_diagonal(dist, np.inf)
        min_separations.append(np.min(dist))

        # Log
        timestamps.append(current_time)
        tracking_errors.append(mean_err)
        for idx, d_id in enumerate(drone_ids):
            target_trajectories[d_id].append(target_pos[idx].copy())

        current_time += dt

    return {
        "timestamps": np.array(timestamps),
        "tracking_errors": np.array(tracking_errors),
        "min_separations": np.array(min_separations),
        "actual_trajectories": {d: np.array(actual_trajectories[d]) for d in drone_ids},
        "target_trajectories": {d: np.array(target_trajectories[d]) for d in drone_ids},
        "mean_error": float(np.mean(tracking_errors)),
        "max_error": float(np.max(tracking_errors)),
    }


def plot_simulation_results(output_filepath: str = "plots/closed_loop_simulation_verification.png"):
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)

    print("[INFO] Running closed-loop simulation for Task 1 (Line Formation)...")
    res_line = run_closed_loop_simulation("line", duration=12.0)

    print("[INFO] Running closed-loop simulation for Task 2 (Circular Formation)...")
    res_circle = run_closed_loop_simulation("circle", duration=12.0)

    fig = plt.figure(figsize=(16, 12), dpi=200)
    fig.patch.set_facecolor("#0F172A")
    colors = ["#38BDF8", "#34D399", "#FBBF24", "#F43F5E"]

    # 1. Line Wave Tracking: Target vs Actual Altitude
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.set_facecolor("#1E293B")
    ax1.grid(True, linestyle="--", alpha=0.3, color="#94A3B8")

    t = res_line["timestamps"]
    for i, d_id in enumerate([1, 2, 3, 4]):
        target_z = res_line["target_trajectories"][d_id][:, 2]
        actual_z = res_line["actual_trajectories"][d_id][:, 2]
        ax1.plot(t, target_z, linestyle="--", color=colors[i], alpha=0.6, label=f"D{d_id} Target" if i == 0 else "")
        ax1.plot(t, actual_z, linestyle="-", color=colors[i], linewidth=2.0, label=f"D{d_id} Actual")

    ax1.set_title("Task 1: Line Wave Altitude Closed-Loop Tracking (Z vs Time)", color="white", fontsize=12, pad=10)
    ax1.set_xlabel("Time (s)", color="#94A3B8")
    ax1.set_ylabel("Altitude Z (m)", color="#94A3B8")
    ax1.tick_params(colors="#94A3B8")
    ax1.legend(loc="upper right", facecolor="#334155", edgecolor="none", labelcolor="white")

    # 2. Tracking Error Over Time (Drift Correction Performance)
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.set_facecolor("#1E293B")
    ax2.grid(True, linestyle="--", alpha=0.3, color="#94A3B8")

    ax2.plot(t, res_line["tracking_errors"] * 100, color="#38BDF8", linewidth=2.0,
             label=f"Line Wave Mean Error (Avg: {res_line['mean_error']*100:.1f} cm)")
    ax2.plot(t, res_circle["tracking_errors"] * 100, color="#34D399", linewidth=2.0,
             label=f"Circular Wave Mean Error (Avg: {res_circle['mean_error']*100:.1f} cm)")
    ax2.axhline(5.0, color="#EF4444", linestyle=":", linewidth=2, label="High-Precision Target (5 cm)")

    ax2.set_title("Drift Correction Performance: Position Tracking Error (cm)", color="white", fontsize=12, pad=10)
    ax2.set_xlabel("Time (s)", color="#94A3B8")
    ax2.set_ylabel("Mean Euclidean Error (cm)", color="#94A3B8")
    ax2.tick_params(colors="#94A3B8")
    ax2.set_ylim(0.0, 15.0)
    ax2.legend(loc="upper right", facecolor="#334155", edgecolor="none", labelcolor="white")

    # 3. 3D Spatial Trajectory - Actual vs Target (Circular Wave)
    ax3 = fig.add_subplot(2, 2, 3, projection="3d")
    ax3.set_facecolor("#1E293B")

    for i, d_id in enumerate([1, 2, 3, 4]):
        act = res_circle["actual_trajectories"][d_id]
        tgt = res_circle["target_trajectories"][d_id]
        ax3.plot(tgt[:, 0], tgt[:, 1], tgt[:, 2], linestyle=":", color=colors[i], alpha=0.5)
        ax3.plot(act[:, 0], act[:, 1], act[:, 2], linestyle="-", color=colors[i], linewidth=2.2, label=f"D{d_id}")

    ax3.set_title("Task 2: Circular Wave 3D Closed-Loop Execution", color="white", fontsize=12, pad=10)
    ax3.set_xlabel("X (m)", color="#94A3B8")
    ax3.set_ylabel("Y (m)", color="#94A3B8")
    ax3.set_zlabel("Z (m)", color="#94A3B8")
    ax3.tick_params(colors="#94A3B8")
    ax3.set_zlim(1.0, 2.0)

    # 4. Inter-Drone Separation Distance (Safety Check)
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.set_facecolor("#1E293B")
    ax4.grid(True, linestyle="--", alpha=0.3, color="#94A3B8")

    ax4.plot(t, res_line["min_separations"], color="#38BDF8", linewidth=2.0, label="Line Wave Min Separation")
    ax4.plot(t, res_circle["min_separations"], color="#34D399", linewidth=2.0, label="Circular Wave Min Separation")
    ax4.axhline(0.40, color="#EF4444", linestyle="--", linewidth=2, label="Safety Limit (d_min = 0.40m)")

    ax4.set_title("Swarm Collision Avoidance Verification", color="white", fontsize=12, pad=10)
    ax4.set_xlabel("Time (s)", color="#94A3B8")
    ax4.set_ylabel("Minimum Distance (m)", color="#94A3B8")
    ax4.tick_params(colors="#94A3B8")
    ax4.set_ylim(0.0, 2.0)
    ax4.legend(loc="lower right", facecolor="#334155", edgecolor="none", labelcolor="white")

    plt.suptitle("Drona Aviation Swarm - Closed-Loop Control, Drift Correction & Safety Verification",
                 color="white", fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig(output_filepath, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"[SUCCESS] Closed-loop simulation verification figure saved to: {output_filepath}")


if __name__ == "__main__":
    plot_simulation_results()

