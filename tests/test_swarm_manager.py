"""
================================================================================
Unit Tests for End-to-End Swarm Mission Orchestration
================================================================================
"""

import sys
import os
import pytest
import numpy as np

# Ensure src path is accessible
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from swarm_manager import SwarmMissionManager, SwarmMissionState
from hardware.pluto_interface import PlutoControlInterface, PlutoSwarmManagerHAL


class TestSwarmMissionManager:
    def test_mission_state_transitions(self):
        """Verify the state machine advances through all mission phases."""
        manager = SwarmMissionManager(
            drone_ids=[1, 2, 3, 4],
            center=[0.0, 0.0, 1.5],
            line_spacing=0.8,
            circle_radius=1.0,
            wave_amplitude=0.20,
            wave_frequency=0.50
        )
        assert manager.state == SwarmMissionState.INITIALIZING

        # Simulate initial vision measurements
        for d_id in [1, 2, 3, 4]:
            manager.tracker.process_raw_measurement(
                drone_id=d_id,
                raw_position=np.array([float(d_id), 0.0, 0.1]),
                timestamp=0.0
            )

        # Update step advances to PREFLIGHT_CHECK, then TASK_1_LINE_FORMATION
        manager.update(0.0)
        assert manager.state == SwarmMissionState.PREFLIGHT_CHECK

        manager.update(0.02)
        assert manager.state == SwarmMissionState.TASK_1_LINE_FORMATION

    def test_vision_loss_failsafe(self):
        """Simulate camera occlusion/loss of tracking: must trigger emergency safe state."""
        manager = SwarmMissionManager()
        # Seed initial valid positions
        for d_id in [1, 2, 3, 4]:
            manager.tracker.process_raw_measurement(
                drone_id=d_id,
                raw_position=np.array([float(d_id), 0.0, 1.5]),
                timestamp=0.0
            )
        manager.update(0.0)
        manager.update(0.02)
        assert manager.state == SwarmMissionState.TASK_1_LINE_FORMATION

        # Advance time by 0.5s without vision updates (> 250ms watchdog timeout)
        manager.update(current_time=0.50)
        assert manager.state == SwarmMissionState.EMERGENCY_SAFE_STATE

