#!/bin/bash
export GZ_SIM_RESOURCE_PATH=/mnt/c/Desktop/interiitps/gazebo/models:$GZ_SIM_RESOURCE_PATH
gz sim -s -r /mnt/c/Desktop/interiitps/gazebo/worlds/swarm_arena.sdf > /tmp/gz.log 2>&1 &
GZ_PID=$!
sleep 4
echo "=== ACTIVE GAZEBO TOPICS ==="
gz topic -l
kill -9 $GZ_PID
