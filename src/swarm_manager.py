"""
================================================================================
Drona Aviation Swarm - Master Mission Orchestrator
================================================================================
Controls the autonomous mission execution state machine:
  INITIALIZING
      │
      ▼
  PREFLIGHT_CHECK ──────(Safety Failure)───────┐
      │                                        │
      ▼                                        │
  TAKEOFF_AND_HOVER                            │
      │                                        │
      ▼                                        │
  TASK_1_LINE_FORMATION                        │
      │                                        │
      ▼                                        │
  TASK_1_LINE_WAVE                             ▼
      │                               EMERGENCY_SAFE_STATE
      ▼                               (Hover / Controlled Land)
  TRANSITION_TO_CIRCLE                         ▲
      │                                        │
      ▼                                        │
  TASK_2_CIRCLE_FORMATION                      │
      │                                        │
      ▼                                        │
  TASK_2_CIRCLE_WAVE ──────────────────────────┘
      │
      ▼
  COORDINATED_LANDING
      │
      ▼
  MISSION_COMPLETE
================================================================================
"""

from enum import Enum, auto
import time
from typing import Dict, List, Optional
import numpy as np
from scipy.optimize import linear_sum_assignment

from formation.line import generate_line
from formation.circle import generate_circle
from trajectory.trajectory_generator import SwarmTrajectoryPlanner
from trajectory.wave_generator import LineWaveTrajectory, CircularWaveTrajectory
from localization.position_tracker import SwarmPositionTracker
from controller.swarm_controller import SwarmController
from safety.collision_detection import SwarmSafetyMonitor
from hardware.pluto_interface import PlutoControlInterface, PlutoSwarmManagerHAL
from monitoring.logger import FlightDataLogger


class SwarmMissionState(Enum):
    INITIALIZING = auto()
    PREFLIGHT_CHECK = auto()
    TAKEOFF_AND_HOVER = auto()
    TASK_1_LINE_FORMATION = auto()
    TASK_1_LINE_WAVE = auto()
    TRANSITION_TO_CIRCLE = auto()
    TASK_2_CIRCLE_FORMATION = auto()
    TASK_2_CIRCLE_WAVE = auto()
    COORDINATED_LANDING = auto()
    MISSION_COMPLETE = auto()
    EMERGENCY_SAFE_STATE = auto()


class SwarmMissionManager:
    """
    Master coordinator executing the full multi-UAV competition workflow.
    """

    def __init__(
        self,
        drone_ids: List[int] = (1, 2, 3, 4),
        center: List[float] = (0.0, 0.0, 1.5),
        line_spacing: float = 1.0,
        circle_radius: float = 1.0,
        wave_amplitude: float = 0.25,
        wave_frequency: float = 0.25,
        hal: Optional[PlutoSwarmManagerHAL] = None
    ):
        self.drone_ids = list(drone_ids)
        self.num_drones = len(self.drone_ids)
        self.center = np.asarray(center, dtype=np.float64)
        self.line_spacing = float(line_spacing)
        self.circle_radius = float(circle_radius)
        self.wave_amplitude = float(wave_amplitude)
        self.wave_frequency = float(wave_frequency)

        # Core subsystem instances
        self.tracker = SwarmPositionTracker(drone_ids=self.drone_ids)
        self.controller = SwarmController(drone_ids=self.drone_ids)
        self.safety = SwarmSafetyMonitor()
        self.logger = FlightDataLogger()

        if hal is None:
            self.hal = PlutoSwarmManagerHAL(PlutoControlInterface(), drone_ids=self.drone_ids)
        else:
            self.hal = hal

        # State machine variables
        self.state = SwarmMissionState.INITIALIZING
        self.state_start_time = 0.0
        self.current_trajectory = None

        # Precompute base geometry setpoints with optimal non-intersecting assignment
        self.line_positions = generate_line(self.num_drones, self.line_spacing, self.center)
        raw_circle = generate_circle(self.num_drones, self.circle_radius, self.center)
        cost_matrix = np.linalg.norm(self.line_positions[:, None, :] - raw_circle[None, :, :], axis=-1)
        _, optimal_cols = linear_sum_assignment(cost_matrix)
        self.circle_positions = raw_circle[optimal_cols]

    def transition_to(self, new_state: SwarmMissionState, current_time: float):
        """Perform state transition and initialize phase-specific trajectory."""
        print(f"[MISSION] {self.state.name} -> {new_state.name} at t = {current_time:.2f}s")
        self.state = new_state
        self.state_start_time = current_time

        if new_state == SwarmMissionState.TASK_1_LINE_FORMATION:
            current_positions = np.array([self.tracker.states[d].position for d in self.drone_ids])
            self.current_trajectory = SwarmTrajectoryPlanner(
                start_positions=current_positions,
                target_positions=self.line_positions,
                duration=4.0,
                start_time=current_time
            )

        elif new_state == SwarmMissionState.TASK_1_LINE_WAVE:
            self.current_trajectory = LineWaveTrajectory(
                base_positions=self.line_positions,
                amplitude=self.wave_amplitude,
                frequency=self.wave_frequency,
                num_cycles=2,
                start_time=current_time
            )

        elif new_state == SwarmMissionState.TRANSITION_TO_CIRCLE:
            self.current_trajectory = SwarmTrajectoryPlanner(
                start_positions=self.line_positions,
                target_positions=self.circle_positions,
                duration=5.0,
                start_time=current_time
            )

        elif new_state == SwarmMissionState.TASK_2_CIRCLE_WAVE:
            self.current_trajectory = CircularWaveTrajectory(
                base_positions=self.circle_positions,
                amplitude=self.wave_amplitude,
                frequency=self.wave_frequency,
                num_cycles=2,
                start_time=current_time
            )

        elif new_state == SwarmMissionState.COORDINATED_LANDING:
            current_positions = np.array([self.tracker.states[d].position for d in self.drone_ids])
            landing_positions = current_positions.copy()
            landing_positions[:, 2] = 0.05  # Lower smoothly to ground
            self.current_trajectory = SwarmTrajectoryPlanner(
                start_positions=current_positions,
                target_positions=landing_positions,
                duration=3.5,
                start_time=current_time
            )

        elif new_state == SwarmMissionState.EMERGENCY_SAFE_STATE:
            self.hal.emergency_stop_all()

    def update(self, current_time: float) -> SwarmMissionState:
        """
        Main 50 Hz control loop step called every 20ms.
        """
        # 1. Update vision watchdogs
        self.tracker.check_watchdogs(current_time)
        positions = self.tracker.get_swarm_positions()

        # Check if any drone lost tracking
        if len(positions) < self.num_drones and self.state not in (
            SwarmMissionState.INITIALIZING,
            SwarmMissionState.PREFLIGHT_CHECK,
            SwarmMissionState.MISSION_COMPLETE,
            SwarmMissionState.EMERGENCY_SAFE_STATE
        ):
            print("[CRITICAL] Vision tracking lost for one or more UAVs! Aborting to Safe State.")
            self.transition_to(SwarmMissionState.EMERGENCY_SAFE_STATE, current_time)
            return self.state

        # 2. Check collision avoidance & geofencing
        if len(positions) == self.num_drones:
            sep_viol = self.safety.check_swarm_separation(positions)
            critical = [v for v in sep_viol if v.is_critical]
            if critical and self.state != SwarmMissionState.EMERGENCY_SAFE_STATE:
                print(f"[CRITICAL] Immediate collision breach! D{critical[0].drone_a_id} - D{critical[0].drone_b_id} (dist={critical[0].distance:.2f}m)")
                self.transition_to(SwarmMissionState.EMERGENCY_SAFE_STATE, current_time)
                return self.state

        # 3. State Machine Progression
        if self.state == SwarmMissionState.INITIALIZING:
            if self.hal.connect_all():
                self.transition_to(SwarmMissionState.PREFLIGHT_CHECK, current_time)

        elif self.state == SwarmMissionState.PREFLIGHT_CHECK:
            if self.hal.arm_all():
                self.transition_to(SwarmMissionState.TASK_1_LINE_FORMATION, current_time)

        elif self.state in (
            SwarmMissionState.TASK_1_LINE_FORMATION,
            SwarmMissionState.TASK_1_LINE_WAVE,
            SwarmMissionState.TRANSITION_TO_CIRCLE,
            SwarmMissionState.TASK_2_CIRCLE_WAVE,
            SwarmMissionState.COORDINATED_LANDING
        ):
            # Compute desired state from active trajectory
            target_pos, target_vel, _ = self.current_trajectory.get_swarm_state(current_time)

            # Compute PID velocity control
            commands, mean_err, max_err = self.controller.update(
                target_positions=target_pos,
                actual_positions=positions,
                target_velocities=target_vel,
                current_time=current_time
            )

            # Dispatch commands to hardware
            self.hal.dispatch_swarm_velocities(commands)

            # Compute min separation for logging
            pos_arr = np.array([positions[d] for d in self.drone_ids])
            diff = pos_arr[:, None, :] - pos_arr[None, :, :]
            dist = np.linalg.norm(diff, axis=-1)
            np.fill_diagonal(dist, np.inf)
            min_sep = float(np.min(dist))

            # Log flight data
            for idx, d_id in enumerate(self.drone_ids):
                self.logger.log_step(
                    timestamp=current_time,
                    drone_id=d_id,
                    target_pos=target_pos[idx],
                    actual_pos=positions[d_id],
                    cmd_vel=commands.get(d_id, np.zeros(3)),
                    min_separation=min_sep
                )

            # Check phase completion
            if self.current_trajectory.is_completed(current_time):
                if self.state == SwarmMissionState.TASK_1_LINE_FORMATION:
                    self.transition_to(SwarmMissionState.TASK_1_LINE_WAVE, current_time)
                elif self.state == SwarmMissionState.TASK_1_LINE_WAVE:
                    self.transition_to(SwarmMissionState.TRANSITION_TO_CIRCLE, current_time)
                elif self.state == SwarmMissionState.TRANSITION_TO_CIRCLE:
                    self.transition_to(SwarmMissionState.TASK_2_CIRCLE_WAVE, current_time)
                elif self.state == SwarmMissionState.TASK_2_CIRCLE_WAVE:
                    self.transition_to(SwarmMissionState.COORDINATED_LANDING, current_time)
                elif self.state == SwarmMissionState.COORDINATED_LANDING:
                    self.hal.land_all()
                    self.transition_to(SwarmMissionState.MISSION_COMPLETE, current_time)

        return self.state


if __name__ == "__main__":
    print("=" * 75)
    print("DRONA AVIATION SWARM - AUTONOMOUS MISSION FLIGHT ORCHESTRATION")
    print("=" * 75)

    from simulation.drone_model import SimulatedPlutoDrone

    # Initialize manager
    manager = SwarmMissionManager(
        drone_ids=[1, 2, 3, 4],
        center=[0.0, 0.0, 1.5],
        line_spacing=1.0,
        circle_radius=1.0,
        wave_amplitude=0.25,
        wave_frequency=0.25
    )

    # Instantiate simulated physical drones with realistic ground-effect drift & sensor noise
    sim_drones = {
        d_id: SimulatedPlutoDrone(
            drone_id=d_id,
            initial_position=np.array([float(d_id - 2.5), 0.0, 0.0]),
            sensor_noise_std=0.012
        )
        for d_id in manager.drone_ids
    }

    # Simulation parameters
    dt = 0.02  # 50 Hz loop rate
    sim_time = 0.0
    max_mission_duration = 35.0  # Max seconds before timeout

    print(f"[STATUS] Initial state: {manager.state.name}")
    print("[STATUS] Beginning 50 Hz closed-loop mission execution...\n")

    last_print_time = -1.0

    while sim_time < max_mission_duration and manager.state != SwarmMissionState.MISSION_COMPLETE:
        # Feed vision measurements from drones to manager tracker
        for d_id in manager.drone_ids:
            meas_pos = sim_drones[d_id].get_measured_position()
            manager.tracker.process_raw_measurement(d_id, meas_pos, sim_time)

        # Execute manager control step
        current_state = manager.update(sim_time)

        # Advance simulated drone dynamics using commands from controller
        active_positions = manager.tracker.get_swarm_positions()
        if len(active_positions) == manager.num_drones and manager.current_trajectory is not None:
            tgt_pos, tgt_vel, _ = manager.current_trajectory.get_swarm_state(sim_time)
            cmds, mean_err, max_err = manager.controller.update(
                target_positions=tgt_pos,
                actual_positions=active_positions,
                target_velocities=tgt_vel,
                current_time=sim_time
            )
            for d_id in manager.drone_ids:
                u_cmd = cmds.get(d_id, np.zeros(3))
                sim_drones[d_id].step(u_cmd, dt)

            # Print telemetry at 2 Hz (every 0.5s)
            if sim_time - last_print_time >= 0.5:
                pos_arr = np.array([active_positions[d] for d in manager.drone_ids])
                diff = pos_arr[:, None, :] - pos_arr[None, :, :]
                dist = np.linalg.norm(diff, axis=-1)
                np.fill_diagonal(dist, np.inf)
                min_sep = np.min(dist)

                print(
                    f"t = {sim_time:5.2f}s | State: {current_state.name:<24} | "
                    f"Mean Err: {mean_err * 100:4.1f}cm | Max Err: {max_err * 100:4.1f}cm | "
                    f"Min Sep: {min_sep:4.2f}m"
                )
                last_print_time = sim_time

        sim_time += dt

    # Final summary and export
    manager.logger.export_csv("flight_data.csv")
    stats = manager.logger.compute_summary_statistics()

    print("\n" + "=" * 75)
    print("MISSION EXECUTION SUMMARY & PERFORMANCE REPORT METRICS")
    print("=" * 75)
    print(f"Final State             : {manager.state.name}")
    print(f"Total Flight Samples    : {stats.get('total_samples', 0)} (at 50 Hz)")
    print(f"Mean Formation Error    : {stats.get('mean_formation_error_m', 0.0) * 100:5.2f} cm")
    print(f"Max Formation Error     : {stats.get('max_formation_error_m', 0.0) * 100:5.2f} cm")
    print(f"RMSE (X axis)           : {stats.get('rmse_x_m', 0.0) * 100:5.2f} cm")
    print(f"RMSE (Y axis)           : {stats.get('rmse_y_m', 0.0) * 100:5.2f} cm")
    print(f"RMSE (Z altitude)       : {stats.get('rmse_z_m', 0.0) * 100:5.2f} cm")
    print(f"Total Euclidean RMSE    : {stats.get('rmse_total_m', 0.0) * 100:5.2f} cm")
    print(f"Minimum Observed Sep    : {stats.get('min_observed_separation_m', 0.0):5.2f} m  (Safety threshold: 0.40 m)")
    print("Flight Telemetry Saved  : flight_data.csv")
    print("=" * 75)


