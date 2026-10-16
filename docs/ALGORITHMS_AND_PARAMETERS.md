# SwarmX: Algorithmic Formulations & Parameter Reference

This document provides a rigorous mathematical reference for all decision-making algorithms, scoring equations, stigmergic pheromone fields, perception metrics, and motion models implemented in the SwarmX codebase.

---

## 1. Ant Colony Optimization (ACO) Task Allocation

SwarmX adapts biological Ant Colony Optimization into a decentralized **Multi-Robot Task Allocation (MRTA)** engine. Unlike classical traveling salesperson problems (TSP) where artificial ants construct static tours over a graph, physical ground robots must make continuous, asynchronous decisions over dynamic frontier clusters under non-holonomic kinematics and communication constraints.

### 1.1 Multi-Objective Task Attractiveness Formulation

For a candidate search task $t_i$ evaluated by robot $r_j$ at global pose $\mathbf{x}_j = (x_j, y_j, \theta_j)$ and simulation time $t$, the attractiveness score is computed by [`TaskScorer`](file:///Users/aziz/Desktop/swarm_car_search/swarm/aco/task_scorer.py#L22-L175).

#### Classical Ratio Formulation:
$$\text{Score}_{\text{classical}}(t_i, r_j) = \frac{[\tau_i(t)]^\alpha \cdot [\eta_i]^\beta \cdot [P_i]^\gamma \cdot [G_i]^\delta}{[C(r_j, t_i)]^\varepsilon \cdot [\Omega_i]^\zeta \cdot [K_i(t)]^\theta}$$

Where:
* $\tau_i(t)$: Positive pheromone intensity sampled from the recruitment layer ($\tau_s$) over the task's bounding box.
* $\eta_i = \max(1.0, \text{size}(t_i))$: Heuristic information gain representing the cell count of the frontier cluster.
* $P_i \in [0.01, 1.0]$: Spatial prior probability that a target vehicle occupies the region (sampled from `TargetProbabilityMap` or `UrbanZone`).
* $G_i \in [0.1, 10.0]$: Dynamic mission priority. Escales by $3.5\times$ for `TARGET_CONFIRMATION` tasks and $2.2\times$ for `VEHICLE_INVESTIGATION` tasks.
* $C(r_j, t_i) = \max(d_{\text{min\_offset}}, \|\mathbf{p}_i - \mathbf{x}_j\|_2)$: Euclidean travel distance between robot $r_j$ and task centroid $\mathbf{p}_i$.
* $\Omega_i = 1.0 + N_{\text{committed}}$: Congestion multiplier penalizing tasks already claimed by peer robots.
* $K_i(t)$: Negative avoidance pheromone sampled from the reservation layer ($\tau_a$).

#### Normalized Multi-Factor Synthesis (Active Implementation):
To prevent numerical underflow in vast environments, [`TaskScorer.explain_score()`](file:///Users/aziz/Desktop/swarm_car_search/swarm/aco/task_scorer.py#L25-L150) synthesizes normalized $[0.0, 1.0]$ components:

$$\hat{\eta}_i = \min\left(1.0, \max\left(0.05, \frac{\text{size}(t_i)}{S_{\max}}\right)\right), \quad S_{\max} = 30.0$$

$$\hat{C}_i = \min\left(1.0, \max\left(0.01, \frac{\|\mathbf{p}_i - \mathbf{x}_j\|_2}{D_{\max}}\right)\right), \quad D_{\max} = 70.0\,\text{m}$$

$$\hat{\Omega}_i = \min\left(1.0, \frac{N_{\text{committed}}}{3.0}\right)$$

$$\hat{\tau}_{s, i} = \min\left(1.0, \max\left(0.05, \frac{\tau_s(t)}{20.0}\right)\right), \quad \hat{K}_i = \min\left(1.0, \frac{\tau_a(t)}{15.0}\right)$$

The positive and negative utility components are aggregated:
$$\Phi_{\text{pos}} = w_{\text{info}} \hat{\eta}_i + w_{\text{prior}} P_i + w_{\text{hist}} H_i + w_{\text{explor}} \hat{\tau}_{s, i}$$
$$\Phi_{\text{neg}} = w_{\text{dist}} \hat{C}_i + w_{\text{cong}} \hat{\Omega}_i + w_{\text{risk}} \hat{K}_i + w_{\text{redun}} R_i$$

$$\text{NetUtility}(t_i, r_j) = \max\left(0.05, 1.0 + \Phi_{\text{pos}} - \Phi_{\text{neg}}\right)$$

$$\text{Score}(t_i, r_j) = \text{NetUtility} \cdot \left(\frac{\tau_i^{0.5 \alpha} \cdot G_i^\delta}{C(r_j, t_i)^{0.5 \varepsilon}}\right) \cdot \text{BatteryFactor}(r_j)$$

$$\text{BatteryFactor}(r_j) = \text{clamp}(0.2, 1.0, 0.4 + 0.6 \cdot \text{SoC}_j)$$

---

### 1.2 Scale-Invariant Boltzmann Softmax Task Selection

To prevent the entire swarm from collapsing greedily onto a single high-scoring frontier cluster, [`ACOEngine.select_task()`](file:///Users/aziz/Desktop/swarm_car_search/swarm/aco/aco_engine.py#L60-L150) samples tasks from a Boltzmann probability distribution parameterized by exploration temperature $T$:

$$P(\text{select } t_i \mid r_j) = \frac{\exp\left(\frac{\text{Score}(t_i, r_j) - \text{Score}_{\max}}{T}\right)}{\sum_{k=1}^M \exp\left(\frac{\text{Score}(t_k, r_j) - \text{Score}_{\max}}{T}\right)}$$

* **Numerical Stability**: Subtracting $\text{Score}_{\max} = \max_k \text{Score}(t_k, r_j)$ in the exponent prevents floating-point overflow (`math.exp` overflow).
* **Temperature Tuning ($T = 0.5$)**:
  * $T \to \infty$: Uniform random exploration over all frontier clusters.
  * $T \to 0^+$: Greedy deterministic selection of $\arg\max_i \text{Score}(t_i, r_j)$.
  * Default $T = 0.5$ balances exploitation of high-value target sightings against broad spatial coverage.
* **$\varepsilon$-Greedy Perturbation**: With probability $\varepsilon = 0.05$, a random unassigned frontier is chosen to guarantee escape from local coverage minima.

---

## 2. Tri-Layer Stigmergic Pheromone Field

Rather than maintaining a single scalar trail, the swarm maintains three independent, spatially discretized 2D pheromone fields updated via stigmergic deposits and continuous evaporation:

```
Pheromone Layers:
├── Exploration Layer (tau_e): Maps surveyed territory (rho_e = 0.01)
├── Recruitment Layer (tau_s): Attracts robots to sightings (rho_s = 0.05)
└── Avoidance Layer   (tau_a): Repels robots from collisions/commitments (rho_a = 0.30)
```

### 2.1 Evaporation Dynamics
At each simulation step $\Delta t$, all grid cells $(x, y)$ in layer $k$ decay exponentially:
$$\tau_k(x, y, t + \Delta t) = \max\left(\tau_{\min}, \tau_k(x, y, t) \cdot (1 - \rho_k \cdot \Delta t)\right)$$

| Layer | Symbol | Evaporation Rate ($\rho_k$) | Halflife ($t_{1/2}$) | Purpose |
| :--- | :---: | :---: | :---: | :--- |
| **Exploration** | $\tau_e$ | $0.01\,\text{s}^{-1}$ | $\approx 69.3\,\text{s}$ | Long-term memory of covered territory; suppresses redundant coverage. |
| **Recruitment** | $\tau_s$ | $0.05\,\text{s}^{-1}$ | $\approx 13.9\,\text{s}$ | Temporary recruitment gradient drawing nearby agents toward visual target cues. |
| **Avoidance** | $\tau_a$ | $0.30\,\text{s}^{-1}$ | $\approx 2.3\,\text{s}$ | Fast-decaying spatial reservation preventing multiple robots from entering the same avenue. |

### 2.2 Gaussian Radial Deposit Kernel
When a robot deposits pheromone at global coordinates $(x_0, y_0)$ with radius $R$ and intensity $Q$, the deposit at any neighboring cell $(x, y)$ is calculated via a 2D Gaussian kernel:
$$\Delta\tau_k(x, y) = Q \cdot \exp\left(-\frac{(x - x_0)^2 + (y - y_0)^2}{2 \sigma^2}\right), \quad \sigma = \frac{R}{2.5}$$
Subject to global clamping: $\tau_k(x, y) \le \tau_{\max} = 50.0$.

---

## 3. Bayesian Target Probability Mapping

In addition to pheromone trails, [`TargetProbabilityMap`](file:///Users/aziz/Desktop/swarm_car_search/swarm/pheromone/target_probability_map.py) tracks the spatial probability distribution $P(\text{target at } x, y) \in [0.01, 0.99]$.

* **Positive Evidence Update** (upon candidate vehicle sighting at $x_0, y_0$):
  $$P(x, y) \leftarrow \min\left(0.99, P(x, y) + C_{\text{target}} \cdot \exp\left(-\frac{d^2}{2 \sigma_{\text{pos}}^2}\right)\right), \quad R_{\text{pos}} = 5.0\,\text{m}$$
* **Negative Evidence Update** (as robots survey empty avenues):
  $$P(x, y) \leftarrow \max\left(0.01, P(x, y) \cdot \left(1 - \lambda_{\text{neg}} \cdot \Delta t \cdot \exp\left(-\frac{d^2}{2 \sigma_{\text{neg}}^2}\right)\right)\right), \quad \lambda_{\text{neg}} = 0.08\,\text{s}^{-1}$$

---

## 4. Multi-Stage Perception & Target Verification

The vehicle identification pipeline evaluates visual observations through 5 sequential verification stages:

### 4.1 Stage 1: Line-of-Sight & FoV Raycast ($C_{\text{veh}}$)
* Range Constraint: $d = \|\mathbf{p}_{\text{veh}} - \mathbf{p}_{\text{robot}}\|_2 \le d_{\max} = 14.0\,\text{m}$.
* Field of View: $|\Delta\theta| = |\text{atan2}(y_v - y_r, x_v - x_r) - \theta_r| \le \frac{\text{FoV}}{2} = 45^\circ$.
* Line-of-Sight Raycast: Verified clear of urban building blocks via Bresenham raycasting.
* Distance Attenuation:
  $$C_{\text{veh}} = \text{clamp}\left(0.50, 0.99, C_{\text{base}} \cdot \left(1.0 - \frac{d}{1.5 \cdot d_{\max}}\right)\right), \quad C_{\text{base}} = 0.94$$

### 4.2 Stage 2: License Plate Localization ($C_{\text{plate}}$)
$$C_{\text{plate}} = \begin{cases} 
\text{clamp}(0.85, 0.98, 0.95 - d/20.0), & d \le 6.0\,\text{m} \\ 
\text{clamp}(0.60, 0.85, 0.90 - d/15.0), & 6.0\,\text{m} < d \le 10.0\,\text{m} \\ 
\max(0.35, 0.70 - d/12.0), & d > 10.0\,\text{m} 
\end{cases}$$

### 4.3 Stage 3: OCR Decoding & Levenshtein Similarity ($C_{\text{ocr}} \cdot S_{\text{plate}}$)
Normalized Levenshtein string similarity between observed plate $s_{\text{obs}}$ and watchlist plate $s_{\text{target}}$:
$$S_{\text{plate}} = 1.0 - \frac{\text{Levenshtein}(s_{\text{obs}}, s_{\text{target}})}{\max(|s_{\text{obs}}|, |s_{\text{target}}|)}$$
At range $> 7.0\,\text{m}$, simulated character ambiguity applies optical substitutions ($8 \leftrightarrow B, 0 \leftrightarrow O, 2 \leftrightarrow Z$).

### 4.4 Stage 4: Visual Appearance Verification ($C_{\text{app}}$)
Evaluated by [`VehicleAppearanceVerifier`](file:///Users/aziz/Desktop/swarm_car_search/perception/appearance_verifier.py):
$$d_{\text{RGB}} = \frac{\sqrt{(R_{\text{obs}} - R_{\text{tgt}})^2 + (G_{\text{obs}} - G_{\text{tgt}})^2 + (B_{\text{obs}} - B_{\text{tgt}})^2}}{255 \sqrt{3}} \in [0.0, 1.0]$$
$$C_{\text{app}} = \kappa_{\text{class}} \cdot (1.0 - d_{\text{RGB}}), \quad \kappa_{\text{class}} = \begin{cases} 1.0, & \text{sedan} \\ 0.4, & \text{non-sedan} \end{cases}$$

### 4.5 Stage 5: Multi-Frame Temporal Consensus ($C_{\text{temp}}$)
Evaluated by [`MultiFrameTargetTracker`](file:///Users/aziz/Desktop/swarm_car_search/perception/temporal_tracker.py#L84-L140):
* Spatial Association: Sightings grouped if $\|\mathbf{p}_k - \mathbf{p}_{\text{track}}\|_2 \le 3.0\,\text{m}$ (`association_radius = 3.0`).
* Confirmation Gate: Requires $k \ge 3$ consecutive frames (`required_confirmations = 3`).
* Temporal Consistency: $C_{\text{temp}} = \min\left(1.0, 0.60 + 0.10 \cdot k\right)$.

### 4.6 Fused Composite Target Confidence ($C_{\text{target}}$)
Evaluated in [`MatchingEngine.evaluate_multi_stage_match()`](file:///Users/aziz/Desktop/swarm_car_search/perception/matching_engine.py#L72-L160) using verified stage weights ($w_{\text{veh}} = 0.15, w_{\text{plate}} = 0.15, w_{\text{ocr}} = 0.30, w_{\text{app}} = 0.20, w_{\text{temp}} = 0.20$):
$$C_{\text{target}} = 0.15 C_{\text{veh}} + 0.15 C_{\text{plate}} + 0.30 (C_{\text{ocr}} \cdot S_{\text{plate}}) + 0.20 C_{\text{app}} + 0.20 C_{\text{temp}}$$

$$\text{Classification} = \begin{cases}
\text{CONFIRMED}, & C_{\text{target}} \ge \theta_{\text{match}} \;\land\; k \ge k_{\text{req}} \;\land\; S_{\text{plate}} \ge \theta_{\text{cand}} \\
\text{CANDIDATE}, & C_{\text{target}} \ge 0.40 \;\land\; S_{\text{plate}} \ge \theta_{\text{cand}} \\
\text{NON\_TARGET}, & \text{otherwise}
\end{cases}$$

Where default constructor thresholds are $\theta_{\text{match}} = 0.75$ and $\theta_{\text{cand}} = 0.45$ (often overridden in benchmark scenarios to $0.80$ and $0.50$ respectively).

---

## 5. Comprehensive Parameter Master Reference

All parameter defaults are extracted directly from source code and YAML configurations:

| Parameter | Actual Default | Unit | Description & Role | Source File | Verification Status |
| :--- | :---: | :---: | :--- | :--- | :---: |
| `alpha` | `1.2` | scalar | Exponent of positive pheromone exploitation | `aco_params.yaml:7` | Verified |
| `beta` | `1.5` | scalar | Exponent of frontier information gain | `aco_params.yaml:8` | Verified |
| `gamma` | `1.8` | scalar | Exponent of vehicle detection prior likelihood | `aco_params.yaml:9` | Verified |
| `delta` | `1.0` | scalar | Exponent of dynamic mission urgency | `aco_params.yaml:10` | Verified |
| `epsilon` | `1.3` | scalar | Distance travel cost penalty exponent | `aco_params.yaml:11` | Verified |
| `zeta` | `2.0` | scalar | Peer robot congestion penalty exponent | `aco_params.yaml:12` | Verified |
| `theta` | `1.5` | scalar | Avoidance/reservation pheromone penalty exponent | `aco_params.yaml:13` | Verified |
| `evaporation_rates.exploration` | `0.01` | $\text{s}^{-1}$ | Exploration pheromone ($\tau_e$) decay rate | `aco_params.yaml:18` | Verified |
| `evaporation_rates.success` | `0.05` | $\text{s}^{-1}$ | Recruitment pheromone ($\tau_s$) decay rate | `aco_params.yaml:19` | Verified |
| `evaporation_rates.avoidance` | `0.30` | $\text{s}^{-1}$ | Avoidance pheromone ($\tau_a$) decay rate | `aco_params.yaml:20` | Verified |
| `deposit_values.exploration_visit`| `1.0` | scalar | $\tau_e$ deposit per visited grid tile | `aco_params.yaml:24` | Verified |
| `deposit_values.vehicle_detected` | `5.0` | scalar | $\tau_s$ deposit on candidate vehicle sighting | `aco_params.yaml:25` | Verified |
| `deposit_values.confirmed_match` | `20.0` | scalar | $\tau_s$ deposit on confirmed stolen vehicle match | `aco_params.yaml:26` | Verified |
| `deposit_values.active_reservation`| `3.0` | scalar | $\tau_a$ deposit on claimed frontier waypoint | `aco_params.yaml:27` | Verified |
| `min_pheromone` | `0.01` | scalar | Lower clamping bound for all pheromone fields | `aco_params.yaml:30` | Verified |
| `max_pheromone` | `50.0` | scalar | Upper clamping bound for all pheromone fields | `aco_params.yaml:31` | Verified |
| `temperature` | `1.0` (Config) / `0.5` (Sim) | scalar | Softmax exploration temperature (`1.0` in `aco_config.py:37`, `0.5` in `run_simulation.py:135`) | `aco_config.py:37`, `run_simulation.py:135` | Verified |
| `epsilon_greedy` | `0.05` | scalar | Random frontier selection probability | `aco_config.py:38` | Verified |
| `max_detection_range` | `14.0` | m | Optical sensor camera detection range limit | `vehicle_detector.py:59` | Verified |
| `fov_degrees` | `90.0` | deg | Camera field of view angle ($\pm 45^\circ$) | `vehicle_detector.py:60` | Verified |
| `w_vehicle` | `0.15` | scalar | Target confidence weight: vehicle detection | `matching_engine.py:72` | Verified |
| `w_plate` | `0.15` | scalar | Target confidence weight: plate detection | `matching_engine.py:73` | Verified |
| `w_ocr` | `0.30` | scalar | Target confidence weight: OCR string match | `matching_engine.py:74` | Verified |
| `w_appearance` | `0.20` | scalar | Target confidence weight: color/aspect ratio | `matching_engine.py:75` | Verified |
| `w_temporal` | `0.20` | scalar | Target confidence weight: temporal consensus | `matching_engine.py:76` | Verified |
| `match_threshold` | `0.75` (Class) / `0.80` (Test)| scalar | Composite confidence threshold for target confirmation | `matching_engine.py:66` | Verified |
| `candidate_threshold`| `0.45` (Class) / `0.50` (Test)| scalar | Minimum Levenshtein similarity for candidate status | `matching_engine.py:67` | Verified |
| `association_radius` | `3.0` | m | Spatial clustering radius for temporal track association | `temporal_tracker.py:95` | Verified |
| `required_confirmations` | `3` | frames | Minimum temporal frames for target confirmation | `temporal_tracker.py:94` | Verified |
| `heartbeat_interval_sec` | `1.0` | s | Interval between peer heartbeat broadcasts | `default_config.yaml:14` | Verified |
| `heartbeat_timeout_sec` | `6.0` (Class) / `5.0` (Yaml) | s | Heartbeat silence threshold triggering failure (`6.0s` in code, `5.0s` in YAML) | `failure_detector.py:39`, `default_config.yaml:15` | Verified |
| `stall_timeout_sec` | `8.0` | s | Immobility duration in `NAVIGATING` triggering stall | `failure_detector.py:41` | Verified |
| `stall_distance_threshold` | `0.3` | m | Minimum travel distance to reset stall timer | `failure_detector.py:42` | Verified |
| `battery_critical_threshold`| `0.10` | scalar | State-of-charge cutoff triggering battery fault | `failure_detector.py:40` | Verified |
| `max_linear_velocity` | `1.5` | m/s | Upper kinematic clamp on forward robot speed | `sim_robot.py:34` | Verified |
| `max_angular_velocity` | `1.5` | rad/s | Upper kinematic clamp on yaw rotational speed | `sim_robot.py:35` | Verified |
| `kp_linear` | `0.8` | scalar | Proportional linear waypoint controller gain | `robot_controller.py:56` | Verified |
| `kp_angular` | `2.0` | scalar | Proportional angular heading controller gain | `robot_controller.py:57` | Verified |
