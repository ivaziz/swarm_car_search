# SwarmX: Autonomous Multi-Robot Outdoor Car Search System

**Master Technical Documentation Package & Engineering Knowledge Base**

---

## 1. Executive Introduction

**SwarmX** is an autonomous multi-robot swarm coordination and target-identification system engineered to search complex outdoor urban environments for missing, suspect, or stolen vehicles. Grounded in **Ant Colony Optimization (ACO)** and **Tri-Layer Stigmergic Interaction**, the system coordinates fleets of Autonomous Ground Vehicles (AGVs) that cooperatively map unknown urban terrain, extract exploration frontiers, detect vehicles using simulated camera raycasting, verify license plates via fuzzy Levenshtein optical character recognition (OCR), and dynamically self-heal from robot hardware failures without central intervention.

The system is delivered with two execution environments:
1. **Lightweight Pure-Python Simulation**: Zero-dependency 2D discrete kinematic physics with real-time ASCII HUD, vector SVG frame snapshots, publication-grade Matplotlib dashboard panels, and animated GIF exports.
2. **Containerized ROS 2 & Gazebo Fortress 3D Simulation**: Complete Docker-based simulation running ROS 2 Humble Hawksbill, Gazebo Fortress 3D physics, sensor bridges, differential drive controllers, and interactive Web 3D / VNC viewing (noVNC on port 6080, native Screen Sharing on port 5900).

---

## 2. Master Documentation Sitemap & Navigation Index

All detailed specifications, mathematical derivations, module references, and verification logs are located in the [`docs/`](file:///Users/aziz/Desktop/swarm_car_search/docs/) directory:

| Document | Primary Contents & Focus | Verification Status |
| :--- | :--- | :---: |
| [**`docs/README.md`**](file:///Users/aziz/Desktop/swarm_car_search/docs/README.md) | Documentation index, sitemap, and navigation roadmap. | **Verified** |
| [**`docs/PROJECT_OVERVIEW.md`**](file:///Users/aziz/Desktop/swarm_car_search/docs/PROJECT_OVERVIEW.md) | Mission objectives, team subsystem boundaries, operational scope, and capability classification matrix. | **Verified** |
| [**`docs/ARCHITECTURE.md`**](file:///Users/aziz/Desktop/swarm_car_search/docs/architecture.md) | Detailed system architecture, ROS 2 node graphs, data flows, execution flows, and **8 Mermaid diagrams**. | **Verified** |
| [**`docs/SOURCE_CODE_REFERENCE.md`**](file:///Users/aziz/Desktop/swarm_car_search/docs/SOURCE_CODE_REFERENCE.md) | Exhaustive source inventory covering every package, class, function, signature, inputs/outputs, and line numbers. | **Verified** |
| [**`docs/ALGORITHMS_AND_PARAMETERS.md`**](file:///Users/aziz/Desktop/swarm_car_search/docs/ALGORITHMS_AND_PARAMETERS.md) | Rigorous mathematical equations for ACO scoring, tri-layer pheromones, Boltzmann Softmax, and exact parameter tables. | **Verified** |
| [**`docs/SETUP_AND_EXECUTION.md`**](file:///Users/aziz/Desktop/swarm_car_search/docs/SETUP_AND_EXECUTION.md) | Step-by-step reproducible installation, build, launch, and display forwarding guides for macOS Apple Silicon and Linux. | **Verified** |
| [**`docs/TESTING_AND_VERIFICATION.md`**](file:///Users/aziz/Desktop/swarm_car_search/docs/TESTING_AND_VERIFICATION.md) | Full 112-test inventory across 15 test suites, real execution logs, reproduction commands, and pass rates. | **Verified (112/112 Passed)** |
| [**`docs/EXPERIMENTS_AND_RESULTS.md`**](file:///Users/aziz/Desktop/swarm_car_search/docs/EXPERIMENTS_AND_RESULTS.md) | Empirical benchmarks across Scenarios A through H, baseline comparisons (Nearest Frontier & Random), and ablation studies. | **Verified** |
| [**`docs/TROUBLESHOOTING.md`**](file:///Users/aziz/Desktop/swarm_car_search/docs/TROUBLESHOOTING.md) | Diagnostic workflows, failure symptoms, root cause analyses, and verified remediations for Docker/ROS/Gazebo/X11. | **Verified** |
| [**`docs/DEVELOPMENT_HISTORY.md`**](file:///Users/aziz/Desktop/swarm_car_search/docs/DEVELOPMENT_HISTORY.md) | Chronological Git commit logs, architectural milestones, uncommitted working tree feature increments, and code provenance. | **Verified** |
| [**`docs/PROJECT_STATUS_AND_ROADMAP.md`**](file:///Users/aziz/Desktop/swarm_car_search/docs/PROJECT_STATUS_AND_ROADMAP.md) | Evidence-based status grading, verified features, unverified components, and prioritized engineering roadmap. | **Verified** |
| [**`docs/SCREENSHOT_INDEX.md`**](file:///Users/aziz/Desktop/swarm_car_search/docs/SCREENSHOT_INDEX.md) | Visual catalog of 14 genuine simulation screenshots, mission summary dashboards, animated GIFs, and vector diagrams. | **Verified** |
| [**`docs/GLOSSARY.md`**](file:///Users/aziz/Desktop/swarm_car_search/docs/GLOSSARY.md) | Standardized definitions of robotics, AI/ML, computer vision, and swarm coordination terminology. | **Verified** |
| [**`docs/DOCUMENTATION_AUDIT.md`**](file:///Users/aziz/Desktop/swarm_car_search/docs/DOCUMENTATION_AUDIT.md) | Final QA compliance matrix auditing every requirement of the project specification against repository ground truth. | **Verified** |

---

## 3. Key Verified System Highlights

* **Automated Verification**: **112 / 112 automated unit and integration tests passing** with 100% success rate across Python 3.11 and Python 3.13.
* **Perception ML Performance**: **100.0% Precision, 100.0% Recall, and 96.5% OCR Character Accuracy** across 8 deterministic benchmark scenarios with zero false alarms emitted across civilian decoy vehicles.
* **Inference Throughput**: **20,350+ FPS** average perception throughput (0.049 ms latency per frame).
* **Fault-Tolerant Dynamic Reassignment**: Autonomous recovery from heartbeat timeouts ($10\,\text{s}$), battery exhaustion ($< 10\%$), and motion stalls ($< 0.1\,\text{m}$ in $8\,\text{s}$), releasing orphaned tasks and depositing collision avoidance pheromones around disabled units.
* **Visual Artifacts**: 14 genuine simulation dashboards, vector maps, and animations cataloged in [`docs/assets/screenshots/`](file:///Users/aziz/Desktop/swarm_car_search/docs/assets/screenshots/).

---

## 4. Quickstart Execution Commands

```bash
# 1. Display system specification and CLI options
python3 scripts/main.py info

# 2. Run the complete automated test suite (112 tests)
pytest -v

# 3. Evaluate AI perception and coordination benchmarks (Scenarios A through H)
python3 scripts/evaluate_ai_benchmarks.py

# 4. Run a 4-robot autonomous search simulation (easy_target scenario)
python3 scripts/run_simulation.py --scenario easy_target --steps 120 --output-dir output/mission_01

# 5. Launch the full Gazebo Fortress 3D GUI simulation (requires Docker Desktop)
./run_gazebo_gui.sh
```
