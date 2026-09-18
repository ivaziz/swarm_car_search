# Ant Colony Optimization (ACO) Algorithmic Design for Multi-Robot Search

## 1. Theoretical Motivation & Problem Formulation

In robotic search and rescue or missing asset tracking, standard centralized ACO cannot be directly deployed due to communication latency, dynamic obstacles, and physical robot inertia.

Our design adapts Ant Colony Optimization into a **Decentralized Multi-Robot Task Allocation (MRTA)** framework:
- **Artificial Ants**: Physical ground robots navigating in an urban grid.
- **Paths / Nodes**: Clustered frontier targets $T = \{t_1, t_2, \dots, t_n\}$.
- **Pheromone Trails**: Spatially discretized multi-layer scalar fields updated via local stigmergic interactions and inter-robot heartbeats.

---

## 2. Multi-Objective Task Scoring Formulation

For a candidate search task $t_i$ evaluated by robot $r_j$ at time $t$, the attractiveness score is defined as:

$$\text{Score}(t_i, r_j) = \frac{[\tau_i(t)]^\alpha \cdot [\eta_i]^\beta \cdot [P_i]^\gamma \cdot [G_i]^\delta}{[C(r_j, t_i)]^\varepsilon \cdot [\Omega_i]^\zeta \cdot [K_i(t)]^\theta}$$

### Parameter Justifications

| Factor | Notation | Role & Scientific Justification |
| :--- | :---: | :--- |
| **Positive Pheromone** | $\tau_i(t)$ | Exploitation of high-value regions where target cues or prior search progress were reported. |
| **Information Gain** | $\eta_i$ | Heuristic factor representing the size of the unexplored frontier cluster ($\text{size}(t_i)$). |
| **Detection Prior** | $P_i$ | Spatial probability of containing a vehicle (e.g., parking lot vs. open pedestrian plaza). |
| **Mission Urgency** | $G_i$ | Dynamic mission priority escalated upon confirmed sightings or external dispatch. |
| **Distance Cost** | $C(r_j, t_i)$ | Euclidean or path-planned travel distance penalty from robot $r_j$'s current pose. |
| **Congestion Factor** | $\Omega_i$ | Multiplicative penalty based on the count of other robots committed to target $t_i$ ($1 + N_{\text{committed}}$). |
| **Negative Pheromone** | $K_i(t)$ | Recent-visit repulsion field ensuring robots do not redundant-search freshly swept tiles. |

---

## 3. Pheromone Dynamics & Tri-Layer Architecture

Instead of a single scalar field, our system maintains three distinct pheromone layers:

1. **Exploration Pheromone ($\tau_e$)**: Deposited as robots map new cells. Low evaporation rate ($\rho_e = 0.01$) ensures long-term memory of covered territory.
2. **Success Pheromone ($\tau_s$)**: Deposited when vehicle detection or OCR matches occur. Medium evaporation ($\rho_s = 0.05$) creates a temporary recruitment gradient for nearby robots.
3. **Avoidance / Reservation Pheromone ($\tau_a$)**: Deposited locally by active robots along their current trajectory. High evaporation ($\rho_a = 0.30$) dynamically disperses other robots without deadlock.

### Update and Evaporation Rule

$$\tau_k(x, y, t + \Delta t) = (1 - \rho_k) \cdot \tau_k(x, y, t) + \sum_{r \in \text{robots}} \Delta\tau_k^r(x, y)$$

Subject to clamping: $\tau_k \in [\tau_{\min}, \tau_{\max}]$ to avoid numerical divergence.

---

## 4. Probabilistic Task Selection (Preventing Premature Swarm Convergence)

To guarantee that the entire swarm does not collapse onto a single target, task selection is performed using a Boltzmann / Softmax probability distribution rather than a deterministic greedy argmax:

$$P(\text{select } t_i \mid r_j) = \frac{\exp\left(\frac{\text{Score}(t_i, r_j)}{T}\right)}{\sum_{k=1}^{M} \exp\left(\frac{\text{Score}(t_k, r_j)}{T}\right)}$$

Where $T$ is the exploration temperature:
- High $T$: Higher stochasticity, broad spatial exploration.
- Low $T$: Greedy exploitation of highest scoring frontiers.
