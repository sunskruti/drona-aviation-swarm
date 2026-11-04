#!/bin/bash
# ==============================================================================
# Drona Aviation Multi-Drone Swarm - Gazebo Sim 3D Launcher
# ==============================================================================
set -e

# Add custom models path
export GZ_SIM_RESOURCE_PATH=/mnt/c/Desktop/interiitps/gazebo/models:$GZ_SIM_RESOURCE_PATH

echo "=========================================================================="
echo "LAUNCHING GAZEBO SIMULATION WITH 4 PLUTO NANO-DRONES"
echo "=========================================================================="

# 1. Launch Gazebo Sim in background with 3D GUI window
echo "[1/2] Opening 3D Gazebo GUI Arena Window..."
gz sim -r /mnt/c/Desktop/interiitps/gazebo/worlds/swarm_arena.sdf &
GZ_PID=$!

# Trap exit to cleanly close Gazebo when script terminates
trap "echo 'Stopping Gazebo...'; kill -9 $GZ_PID 2>/dev/null || true" EXIT

# 2. Wait for Gazebo scene to initialize
echo "[2/2] Connecting Swarm Mission Controller in 3 seconds..."
sleep 4

# 3. Run closed-loop swarm flight manager
python3 /mnt/c/Desktop/interiitps/gazebo/gazebo_swarm_bridge.py

echo "Swarm execution complete. Press Ctrl+C in this terminal to close Gazebo."
wait $GZ_PID
