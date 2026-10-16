# SwarmX: Technical Glossary & Acronym Reference

This document provides standardized definitions for robotics, artificial intelligence, computer vision, and swarm coordination terminology used throughout the SwarmX codebase and documentation.

---

## 1. Algorithmic & Swarm Intelligence Terminology

### Ant Colony Optimization (ACO)
A bio-inspired metaheuristic based on the foraging behavior of social ants. In SwarmX, artificial ants correspond to autonomous physical ground vehicles, edges correspond to candidate frontier search tasks, and pheromone trails correspond to continuously decaying 2D spatial scalar fields.

### Multi-Robot Task Allocation (MRTA)
The formal computational problem of assigning a set of operational tasks (e.g., exploring frontier clusters or verifying suspect vehicles) to a fleet of autonomous mobile robots to optimize a global objective while respecting distance and kinematic constraints.

### Stigmergy
A mechanism of indirect coordination among agents via physical or digital modifications of the environment. In SwarmX, robots communicate stigmergically by reading and depositing digital pheromone fields ($\tau_e, \tau_s, \tau_a$) rather than relying exclusively on centralized broadcast servers.

### Boltzmann / Softmax Selection
A temperature-parameterized probability distribution used in reinforcement learning and metaheuristics to select actions proportionally to their estimated value:
$$P(a_i) = \frac{\exp(Q(a_i)/T)}{\sum_k \exp(Q(a_k)/T)}$$
The parameter $T$ (temperature) regulates the exploration-exploitation trade-off.

### $\varepsilon$-Greedy Selection
An action selection policy that chooses the highest-scoring candidate with probability $1 - \varepsilon$, and a random candidate with probability $\varepsilon$, guaranteeing that the swarm does not become trapped in local coverage minima.

### Dynamic Urgency Escalation
An algorithmic scoring mechanism that temporarily boosts the priority weight of a search task by $2.2\times$ to $3.5\times$ when candidate visual cues indicate a likely stolen vehicle match.

---

## 2. Robotics & SLAM Terminology

### SLAM (Simultaneous Localization and Mapping)
The computational process by which an autonomous robot constructs a map of an unknown environment while simultaneously estimating its own pose within that map. Decoupled in SwarmX through [`ISLAMProvider`](file:///Users/aziz/Desktop/swarm_car_search/interfaces/slam_interface.py).

### Frontier-Based Exploration
An autonomous mapping paradigm introduced by Brian Yamauchi where robots identify frontiers—the spatial boundaries dividing mapped free space from unmapped unknown space—and navigate toward frontier centroids to maximize new information gain.

### 2D Occupancy Grid
A discretized spatial representation where an environment is partitioned into square cells, each assigned a discrete occupancy probability: `-1` (unknown space), `0` (free navigable space), and `100` (static obstacle).

### Raycasting
A geometric algorithm (implemented in SwarmX via Bresenham's line algorithm) that simulates the line-of-sight path of sensor beams (e.g., LiDAR or cameras) through an occupancy grid, detecting obstacle collisions and occlusions.

### Differential Drive Kinematics
A non-holonomic mobile robot motion model driven by two independently actuated coaxial wheels. The robot's motion is governed by forward linear velocity $v$ and yaw angular velocity $\omega$.

### Pose2D
A 3-degree-of-freedom spatial coordinate representing a mobile robot's position and heading in a 2D Euclidean plane: $(x, y, \theta)$.

---

## 3. Computer Vision & Target Identification Terminology

### Field of View (FoV)
The observable angular sector of a camera sensor. In SwarmX, cameras are modeled with a $90^\circ$ horizontal FoV ($\pm 45^\circ$ relative to the robot's heading) and an optical range limit of $14.0\,\text{m}$.

### Levenshtein Distance
A string edit metric quantifying the minimum number of single-character edits (insertions, deletions, or substitutions) required to transform one string into another. Used in SwarmX for fuzzy license plate matching under optical noise.

### Normalized Levenshtein Similarity
A scalar metric in $[0.0, 1.0]$ representing string equivalence:
$$S_{\text{plate}} = 1.0 - \frac{\text{Levenshtein}(s_1, s_2)}{\max(|s_1|, |s_2|)}$$

### Multi-Frame Temporal Consensus
A computer vision verification technique that aggregates successive image detections of a stationary object over time, using spatial gating and majority character voting to eliminate single-frame optical glitches and spurious decoy false positives.

### Optical Confusion Map
A substitution matrix modeling character misreadings typical of OCR engines under blur or distance attenuation (e.g., $8 \leftrightarrow B$, $0 \leftrightarrow O$, $2 \leftrightarrow Z$).

---

## 4. Software & Infrastructure Terminology

### ROS 2 (Robot Operating System 2)
An open-source software framework providing middleware, hardware abstraction, device drivers, and inter-process communication (topics, services, actions) for robotics systems. SwarmX targets ROS 2 Humble Hawksbill.

### Gazebo Fortress (Ignition Gazebo 6)
An open-source 3D dynamic physics simulator supporting rigid-body dynamics, sensor rendering (LiDAR, RGB cameras, IMUs), and realistic urban worlds.

### `ros_gz_bridge`
A ROS 2 communication bridge that translates messages between ROS 2 topics and Gazebo Transport topics (e.g., converting `sensor_msgs/LaserScan` to `ignition.msgs.LaserScan`).

### URDF / Xacro (Unified Robot Description Format)
An XML specification used in ROS to describe the kinematic structure, dimensions, masses, visual meshes, collision geometry, and sensor plugins of a robot.

### noVNC
An HTML5 VNC client operating entirely in the browser using WebSockets, allowing operators to interact with virtual 3D GUI sessions without installing native desktop VNC clients.

### Colcon (Collective Construction)
The official build tool for ROS 2 workspaces, responsible for orchestrating package builds, dependencies, and symlink installations.
