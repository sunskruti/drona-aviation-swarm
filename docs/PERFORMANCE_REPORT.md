# Multi-Drone Swarm Flight Performance Report
**Inter IIT Tech Meet 15.0 — Drona Aviation Multi-Drone Swarm Execution**

---

## 1. Executive Summary & Scoring Verification

| Evaluation Component | Weightage | Target Benchmark | Achieved System Metric | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Physical Swarm Execution & Formation Accuracy** | 40% | Mean Error < 10.0 cm | **5.50 cm** | **EXCEEDED** |
| **Real-time Localization & Drift Correction** | 20% | Correct drift at 50 Hz | **100% Corrected (50 Hz loop)** | **PASSED** |
| **Collision Avoidance & Minimum Separation** | 15% | Distance $\ge 0.40\text{ m}$ | **Min Sep: 0.87 m** | **PASSED** |
| **Synchronized Wave Execution** | Mandatory | Harmonic Phase Delays | **$\Delta\phi = 90^\circ$ uniform** | **PASSED** |

---

## 2. Statistical Position Error Breakdown

- **Total 50 Hz Flight Log Samples**: 6912
- **Overall Mean Formation Tracking Error**: **5.50 cm**
- **Peak Maximum Transient Deviation**: **34.16 cm**

### Root-Mean-Square Error (RMSE) per Axis:
| Axis | Formula | Achieved RMSE | Competition Tolerance |
| :--- | :--- | :---: | :---: |
| **Lateral X-axis** | $\sqrt{\frac{1}{N}\sum e_x^2}$ | **1.54 cm** | $\le 10.0\text{ cm}$ |
| **Longitudinal Y-axis** | $\sqrt{\frac{1}{N}\sum e_y^2}$ | **1.53 cm** | $\le 10.0\text{ cm}$ |
| **Vertical Z-altitude** | $\sqrt{\frac{1}{N}\sum e_z^2}$ | **8.20 cm** | $\le 12.0\text{ cm}$ |
| **Total 3D Spatial RMSE** | $\sqrt{\text{RMSE}_x^2 + \text{RMSE}_y^2 + \text{RMSE}_z^2}$ | **8.49 cm** | $\le 15.0\text{ cm}$ |

---

## 3. Mission Phase-by-Phase Performance

| Mission Flight Phase | Duration (s) | Mean Error (cm) | Max Error (cm) | Min Observed Separation (m) | Safety Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Task 1: Line Wave (Z Oscillation)** | ~11.0s | 3.57 cm | 8.46 cm | 0.96 m | **SAFE** (>0.4m) |
| **Bipartite Formation Transition** | ~5.0s | 2.40 cm | 7.22 cm | 0.87 m | **SAFE** (>0.4m) |
| **Task 2: Circular Wave (Z Oscillation)** | ~11.0s | 3.36 cm | 7.78 cm | 1.36 m | **SAFE** (>0.4m) |
| **Coordinated Swarm Landing** | ~3.6s | 14.96 cm | 34.16 cm | 1.30 m | **SAFE** (>0.4m) |

---

## 4. Multi-Agent Collision Avoidance Analysis

- **Configured Physical Safety Threshold ($d_{\min}$)**: $0.40\text{ m}$
- **Early-Warning Buffer**: $0.55\text{ m}$
- **Absolute Minimum Distance Recorded**: **0.87 m** (Safety Margin: $+0.47\text{ m}$ above threshold)
- **Path Deconfliction Strategy**: Hungarian Bipartite Assignment $\min \sum \lVert p_{\text{start}, i} - p_{\text{target}, \pi(i)} \rVert^2$ eliminated cross-origin flight path intersections.

---

## 5. Artifacts and Raw Telemetry Reference

- **Flight Telemetry Dataset**: [`flight_data.csv`](../flight_data.csv)
- **Geometry Verification Figures**: [`plots/formation_geometry.png`](../plots/formation_geometry.png)
- **Trajectory Phase Delay Plots**: [`plots/trajectory_wave_verification.png`](../plots/trajectory_wave_verification.png)
- **Closed-Loop Drift Correction Curves**: [`plots/closed_loop_simulation_verification.png`](../plots/closed_loop_simulation_verification.png)
