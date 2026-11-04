"""
================================================================================
Drona Aviation Swarm - Interactive 3D Real-time Flight Simulator & Animator
================================================================================
Provides a live visual simulation window for the 4-drone Pluto swarm:
- 3D interactive arena with geofence bounding box and arena floor grid
- 4 color-coded quadrotor representations with active rotor orientation
- Trajectory ribbons visualizing the flight paths over time
- Live HUD displaying flight time, mission phase, tracking error, and separation
- Can run interactively in a GUI window or export to an animated GIF

Run interactively:
    python src/simulation/animate_swarm.py

Export animated GIF:
    python src/simulation/animate_swarm.py --save-gif
================================================================================
"""

import sys
import os
import argparse
import numpy as np

# Ensure src path is accessible
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from swarm_manager import SwarmMissionManager, SwarmMissionState
from simulation.drone_model import SimulatedPlutoDrone


def run_precomputed_mission(duration: float = 34.0, dt: float = 0.05):
    """
    Run the full closed-loop mission and record spatial histories for smooth animation.
    """
    manager = SwarmMissionManager(
        drone_ids=[1, 2, 3, 4],
        center=[0.0, 0.0, 1.5],
        line_spacing=1.0,
        circle_radius=1.0,
        wave_amplitude=0.25,
        wave_frequency=0.25
    )

    sim_drones = {
        d_id: SimulatedPlutoDrone(
            drone_id=d_id,
            initial_position=np.array([float(d_id - 2.5), 0.0, 0.0]),
            sensor_noise_std=0.012
        )
        for d_id in manager.drone_ids
    }

    steps = int(duration / dt)
    time_points = []
    actual_pos_history = []  # List of (4, 3)
    target_pos_history = []  # List of (4, 3)
    states_history = []
    errors_history = []
    separations_history = []

    sim_time = 0.0
    for _ in range(steps):
        # 1. Vision feedback
        for d_id in manager.drone_ids:
            meas = sim_drones[d_id].get_measured_position()
            manager.tracker.process_raw_measurement(d_id, meas, sim_time)

        # 2. Manager update
        state = manager.update(sim_time)
        active_pos = manager.tracker.get_swarm_positions()

        # 3. Step dynamics
        if len(active_pos) == manager.num_drones and manager.current_trajectory is not None:
            tgt_pos, tgt_vel, _ = manager.current_trajectory.get_swarm_state(sim_time)
            cmds, mean_err, _ = manager.controller.update(
                target_positions=tgt_pos,
                actual_positions=active_pos,
                target_velocities=tgt_vel,
                current_time=sim_time
            )
            for d_id in manager.drone_ids:
                u_cmd = cmds.get(d_id, np.zeros(3))
                sim_drones[d_id].step(u_cmd, dt)

            # Record
            pos_now = np.array([sim_drones[d].position.copy() for d in manager.drone_ids])
            diff = pos_now[:, None, :] - pos_now[None, :, :]
            dist = np.linalg.norm(diff, axis=-1)
            np.fill_diagonal(dist, np.inf)

            time_points.append(sim_time)
            actual_pos_history.append(pos_now)
            target_pos_history.append(tgt_pos.copy())
            states_history.append(state.name)
            errors_history.append(mean_err)
            separations_history.append(np.min(dist))

        sim_time += dt

    return {
        "times": np.array(time_points),
        "actual": np.array(actual_pos_history),    # (Frames, 4, 3)
        "target": np.array(target_pos_history),    # (Frames, 4, 3)
        "states": states_history,
        "errors": np.array(errors_history),
        "separations": np.array(separations_history)
    }


def launch_animation(data, save_gif_path: str = None):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from mpl_toolkits.mplot3d import Axes3D

    times = data["times"]
    actual = data["actual"]
    target = data["target"]
    states = data["states"]
    errors = data["errors"]
    separations = data["separations"]

    num_frames = len(times)
    num_drones = 4
    colors = ["#38BDF8", "#34D399", "#FBBF24", "#F43F5E"]
    labels = ["Pluto 1", "Pluto 2", "Pluto 3", "Pluto 4"]

    fig = plt.figure(figsize=(14, 9), dpi=120)
    fig.patch.set_facecolor("#0F172A")

    ax = fig.add_subplot(1, 1, 1, projection="3d")
    ax.set_facecolor("#1E293B")

    # Fixed arena limits
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_zlim(0.0, 2.5)

    ax.set_xlabel("X World Position (m)", color="#94A3B8", labelpad=10)
    ax.set_ylabel("Y World Position (m)", color="#94A3B8", labelpad=10)
    ax.set_zlabel("Altitude Z (m)", color="#94A3B8", labelpad=10)
    ax.tick_params(colors="#94A3B8")

    # Floor grid outline
    floor_x = [-2.5, 2.5, 2.5, -2.5, -2.5]
    floor_y = [-2.5, -2.5, 2.5, 2.5, -2.5]
    ax.plot(floor_x, floor_y, [0, 0, 0, 0, 0], color="#475569", linestyle="--", linewidth=1.2)

    # Graphical elements to update
    drone_scatters = [
        ax.scatter([], [], [], s=140, color=colors[i], edgecolors="white", linewidth=1.5, label=labels[i])
        for i in range(num_drones)
    ]
    target_scatters = [
        ax.scatter([], [], [], s=40, color=colors[i], marker="x", alpha=0.6)
        for i in range(num_drones)
    ]
    trail_lines = [
        ax.plot([], [], [], color=colors[i], linewidth=1.8, alpha=0.7)[0]
        for i in range(num_drones)
    ]

    # Live HUD text box
    hud_text = fig.text(
        0.03, 0.95, "", color="white", fontsize=11, family="monospace",
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#020617", edgecolor="#38BDF8", alpha=0.85)
    )

    ax.legend(loc="upper right", facecolor="#334155", edgecolor="none", labelcolor="white", fontsize=10)

    # Frame trail memory: show last 30 frames of trail
    trail_len = 35

    def update_frame(frame_idx):
        t_curr = times[frame_idx]
        pos_curr = actual[frame_idx]
        tgt_curr = target[frame_idx]
        state_curr = states[frame_idx]
        err_curr = errors[frame_idx] * 100.0
        sep_curr = separations[frame_idx]

        for i in range(num_drones):
            # Update drone position
            drone_scatters[i]._offsets3d = ([pos_curr[i, 0]], [pos_curr[i, 1]], [pos_curr[i, 2]])
            target_scatters[i]._offsets3d = ([tgt_curr[i, 0]], [tgt_curr[i, 1]], [tgt_curr[i, 2]])

            # Update trail
            start_k = max(0, frame_idx - trail_len)
            trail_x = actual[start_k:frame_idx + 1, i, 0]
            trail_y = actual[start_k:frame_idx + 1, i, 1]
            trail_z = actual[start_k:frame_idx + 1, i, 2]
            trail_lines[i].set_data(trail_x, trail_y)
            trail_lines[i].set_3d_properties(trail_z)

        # Update HUD
        status_color = "SAFE" if sep_curr >= 0.55 else ("WARNING" if sep_curr >= 0.40 else "CRITICAL")
        hud_str = (
            f"DRONA AVIATION 4-DRONE SWARM SIMULATOR\n"
            f"────────────────────────────────────────\n"
            f"Flight Time      : {t_curr:5.2f} s / {times[-1]:5.2f} s\n"
            f"Mission Phase    : {state_curr}\n"
            f"Formation Error  : {err_curr:5.2f} cm\n"
            f"Min Separation   : {sep_curr:5.2f} m (Limit: 0.40m)\n"
            f"Safety Status    : [{status_color}]"
        )
        hud_text.set_text(hud_str)

        # Subtle 3D camera pan around the arena for cinematic perspective
        ax.view_init(elev=28.0, azim=45.0 + (frame_idx * 0.15))
        return drone_scatters + target_scatters + trail_lines + [hud_text]

    anim = FuncAnimation(fig, update_frame, frames=num_frames, interval=50, blit=False)

    if save_gif_path:
        print(f"[INFO] Exporting animated simulation GIF to: {save_gif_path} (this may take a few moments)...")
        # Sample every 2nd frame for optimal file size and smooth 25 fps
        anim_gif = FuncAnimation(fig, update_frame, frames=range(0, num_frames, 2), interval=40, blit=False)
        anim_gif.save(save_gif_path, writer="pillow", fps=25)
        print(f"[SUCCESS] Animated simulation GIF saved to: {save_gif_path}")
        plt.close(fig)
    else:
        plt.title("Drona Aviation Swarm - Autonomous Multi-UAV 3D Flight Simulation",
                  color="white", fontsize=14, fontweight="bold", pad=20)
        print("[INFO] Launching 3D Interactive Simulation GUI... Use mouse to rotate/zoom!")
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Drone Swarm 3D Flight Visualizer")
    parser.add_argument("--save-gif", type=str, default="", help="Path to save animated GIF")
    args = parser.parse_args()

    print("[INFO] Computing 4-drone closed-loop mission trajectory dynamics...")
    flight_data = run_precomputed_mission()

    if args.save_gif:
        launch_animation(flight_data, save_gif_path=args.save_gif)
    else:
        launch_animation(flight_data)

