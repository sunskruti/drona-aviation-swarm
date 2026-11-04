# Drona Aviation Multi-Drone Swarm Execution Platform
**Inter IIT Tech Meet 15.0 — Problem Statement: Synchronized Multi-Drone Swarm Execution**

An enterprise-grade, mathematically verified multi-UAV autonomous swarm execution system built for **Pluto nano-drones**. Supports synchronized collinear straight-line and circular formation flight, $C^2$ continuous traveling wave dynamics, external computer vision pose tracking, real-time closed-loop PID drift correction, dynamic inter-drone collision avoidance, geofencing, and Dockerized remote flight evaluation via Airlink.

---

## 1. System Architecture & Evaluation Alignment

```
Total Evaluation (100%)
├── 40% : Physical Swarm Execution & Formation Accuracy
├── 20% : Real-time Localization & Drift Correction
├── 15% : Collision Avoidance & Fail-Safe Architecture
├── 15% : Technical Report (<= 25 pages) & Performance Analysis (<= 10 pages)
└── 10% : Simulation Fidelity, Docker Containerization & Reproducibility
```

### Modular Pipeline
```
[User / Mission Configuration]
              │
              ▼
    [Swarm Mission Manager] ◄────────┐
              │                      │
   ┌──────────┴──────────┐           │  (Closed-Loop Feedback 50Hz)
   ▼                     ▼           │
[Line Formation]   [Circle Formation]│
   └──────────┬──────────┘           │
              ▼                      │
   [Trajectory Generator]            │
              │                      │
              ▼                      │
     [Traveling Wave Engine]         │
              │                      │
              ▼                      │
   [Swarm Safety Preflight]          │
              │                      │
              ▼                      │
    [Closed-Loop 3D PID]             │
              │                      │
              ▼                      │
   [Hardware / Sim Interface]        │
              │                      │
              ▼                      │
    [Pluto 4-Drone Swarm]            │
              │                      │
              ▼                      │
  [External Vision Localization] ────┘
              │
              ▼
   [Flight Telemetry Logger] ──► CSV & Report
```

---

## 2. Mathematical Formulations

### A. Formation Centroid Preservation & Invariance
For $N$ drones arranged along a line with spacing $d$ centered at $\mathbf{p}_c = [x_c, y_c, z_c]^T$:
$$\delta_i = i - \frac{N - 1}{2}, \quad i \in \{0, 1, \dots, N-1\}$$
$$\mathbf{p}_i = \mathbf{p}_c + R(\psi) \begin{bmatrix} \delta_i \cdot d \\ 0 \end{bmatrix}$$
$$\sum_{i=0}^{N-1} \delta_i = \sum_{i=0}^{N-1} i - N\frac{N-1}{2} = \frac{(N-1)N}{2} - \frac{N(N-1)}{2} \equiv 0$$
Thus, the swarm centroid $\frac{1}{N}\sum \mathbf{p}_i = \mathbf{p}_c$ is strictly invariant.

### B. Circular Formation Harmony
For drones on a circle of radius $r$:
$$\theta_i = \theta_0 + i \cdot \frac{2\pi}{N}, \quad x_i = x_c + r \cos(\theta_i), \quad y_i = y_c + r \sin(\theta_i)$$
The theoretical inter-drone adjacent chord distance is:
$$d_{\text{chord}} = 2r \sin\left(\frac{\pi}{N}\right) \quad (\text{for } N=4, r=1.0\text{m} \implies d_{\text{chord}} = \sqrt{2} \approx 1.4142\text{m})$$

### C. Quintic Polynomial Minimum-Jerk Time-Scaling
$$\tau = \frac{t - t_{\text{start}}}{T} \in [0, 1]$$
$$s(\tau) = 10\tau^3 - 15\tau^4 + 6\tau^5$$
$$\dot{s}(\tau) = \frac{1}{T}(30\tau^2 - 60\tau^3 + 30\tau^4), \quad \ddot{s}(\tau) = \frac{1}{T^2}(60\tau - 180\tau^2 + 120\tau^3)$$
Boundary invariants: $s(0)=0, s(1)=1, \dot{s}(0)=\dot{s}(1)=0, \ddot{s}(0)=\ddot{s}(1)=0$.

### D. Synchronized Traveling Wave Dynamics
$$z_i(t) = z_0 + E(t) \cdot A \sin(\omega t - \phi_i)$$
Where $\phi_i = i \cdot \frac{2\pi}{N}$ ($0, 90^\circ, 180^\circ, 270^\circ$ for $N=4$), and $E(t)$ is a $C^2$ smooth ramp envelope.

### E. Discrete PID Feedback + Feedforward Drift Correction
$$\mathbf{e}_k = \mathbf{p}_{\text{target}}(t_k) - \mathbf{p}_{\text{actual}}(t_k)$$
$$\mathbf{u}_k = \mathbf{v}_{\text{ff}, k} + K_p \mathbf{e}_k + K_i \sum_{j} \mathbf{e}_j \Delta t + K_d \mathbf{D}_{\text{filt}, k}$$
Clamped to $\lVert \mathbf{u}_k \rVert \le \mathbf{v}_{\max} = [0.80, 0.80, 0.50]^T\text{ m/s}$.

---

## 3. Directory Layout

```
.
├── config/
│   ├── swarm.yaml               # Swarm size, altitude, loop rates, network
│   ├── formation.yaml           # Line spacing, circle radius, wave params
│   └── safety.yaml              # Collision thresholds, geofencing, watchdogs
├── src/
│   ├── formation/
│   │   ├── line.py              # Collinear generator & distance matrices
│   │   ├── circle.py            # Circular generator & chord verifiers
│   │   └── visualize_formations.py # Publication 2D/3D plots
│   ├── trajectory/
│   │   ├── trajectory_generator.py # Quintic polynomial planner
│   │   ├── wave_generator.py    # Synchronized 3D traveling wave engine
│   │   └── visualize_trajectories.py # Phase delays & space-time ribbons
│   ├── localization/
│   │   ├── coordinate_transform.py # SE(3) Camera <-> World transformations
│   │   └── position_tracker.py  # Alpha-Beta filter & outlier rejection
│   ├── controller/
│   │   ├── pid_controller.py    # Decoupled 3D PID with anti-windup
│   │   └── swarm_controller.py  # Swarm velocity coordination
│   ├── safety/
│   │   └── collision_detection.py # Pairwise separation & geofencing
│   ├── simulation/
│   │   ├── drone_model.py       # First-order motor dynamics & drift bias
│   │   └── swarm_simulator.py   # Full 4-drone closed-loop testbed
│   ├── hardware/
│   │   └── pluto_interface.py   # PlutoControl & ROS 2 HAL abstraction
│   ├── monitoring/
│   │   └── logger.py            # 50 Hz flight recorder & metrics exporter
│   └── swarm_manager.py         # Autonomous state machine coordinator
├── tests/
│   ├── test_formations.py       # Formation geometry unit tests (10/10 PASS)
│   ├── test_trajectories.py     # Trajectory & wave unit tests (6/6 PASS)
│   ├── test_closed_loop_and_safety.py # PID & safety tests (7/7 PASS)
│   └── test_swarm_manager.py    # State machine tests (2/2 PASS)
├── plots/
│   ├── formation_geometry.png   # 2D & 3D base geometry verification
│   ├── trajectory_wave_verification.png # Wave phase offsets & separation
│   └── closed_loop_simulation_verification.png # Tracking error & drift correction
├── Dockerfile                   # Containerized Airlink evaluation environment
├── docker-compose.yml           # Multi-container launcher
├── requirements.txt
└── README.md
```

---
<img width="3200" height="2400" alt="closed_loop_simulation_verification" src="https://github.com/user-attachments/assets/e3cf9ff0-4f63-49f0-aec1-2b3158d71f42" />



## 4. Quick Start & Execution

### Running All Unit Tests (25 / 25 Passing)
```bash
pytest -v tests/
```

### Generating Visualizations
```bash
python src/formation/visualize_formations.py
python src/trajectory/visualize_trajectories.py
python src/simulation/swarm_simulator.py
```

### Running the Full Autonomous Closed-Loop Swarm Mission
```bash
python src/swarm_manager.py
```

### Running in Docker (for Airlink Testing)
```bash
docker compose build
docker compose up
```

