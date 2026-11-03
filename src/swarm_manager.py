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

        # Precompute base geometry setpoints
        self.line_positions = generate_line(self.num_drones, self.line_spacing, self.center)
        self.circle_positions = generate_circle(self.num_drones, self.circle_radius, self.center)

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

