# ==============================================================================
# Drona Aviation Multi-Drone Swarm - 1-Click Gazebo Launcher (Windows)
# ==============================================================================
Write-Host "==========================================================================" -ForegroundColor Cyan
Write-Host "  DRONA AVIATION MULTI-DRONE SWARM - GAZEBO SIMULATION" -ForegroundColor Green
Write-Host "==========================================================================" -ForegroundColor Cyan
Write-Host "Launching 3D Gazebo Sim environment in WSL Ubuntu..." -ForegroundColor Yellow

wsl -d Ubuntu bash /mnt/c/Desktop/interiitps/gazebo/launch_gazebo_swarm.sh
