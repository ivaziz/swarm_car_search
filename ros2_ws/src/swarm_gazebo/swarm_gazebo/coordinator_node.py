"""
ROS 2 SwarmX Coordinator Node.
Executes Decentralized ACO Swarm Intelligence, Tri-Layer Pheromone Map,
Boltzmann Softmax Task Allocation, and Dynamic Failure Recovery in the Gazebo Simulation.
"""

import math
import time
from typing import Dict, List, Optional, Set, Tuple

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from interfaces.slam_interface import Pose2D, ISLAMProvider
from perception.vehicle_detector import TargetVehicle, SimulatedVehicleDetector
from perception.plate_recognizer import SimulatedPlateRecognizer
from perception.matching_engine import MatchingEngine
from simulation.environment import UrbanEnvironment, UrbanZone
from simulation.obstacle_generator import ObstacleGenerator
from swarm.aco.aco_config import ACOConfig
from swarm.aco.aco_engine import ACOEngine
from swarm.task_allocation.allocation_strategy import ACOAllocationStrategy
from swarm.task_allocation.frontier_task import SearchTask, TaskStatus, TaskType
from swarm.core.swarm_coordinator import SwarmCoordinator
from swarm.core.robot_agent import RobotAgent
from swarm.states.robot_states import RobotState


class GazeboSwarmCoordinatorNode(Node):
    """
    ROS 2 Node encapsulating the full SwarmX Intelligence Engine.
    Coordinates 4 autonomous ground vehicles exploring the Gazebo urban world,
    allocates frontier search tasks via ACO, reacts to target sightings,
    and handles dynamic robot failure injection and task recovery.
    """

    def __init__(
        self,
        slam_provider: ISLAMProvider,
        robot_ids: List[str],
        env_width: float = 50.0,
        env_height: float = 50.0,
        target_plate: str = "DXB-88392",
        target_pos: Tuple[float, float] = (25.0, 25.0),
        inject_failure_tick: Optional[int] = None,
        fail_robot_id: str = "robot_1",
    ) -> None:
        super().__init__("swarm_coordinator_node")
        self.slam_provider = slam_provider
        self.robot_ids = robot_ids
        self.target_plate = target_plate
        self.target_pos = target_pos
        self.inject_failure_tick = inject_failure_tick
        self.fail_robot_id = fail_robot_id
        self.current_tick = 0
        self._failure_injected = False

        # 1. Environment & Zones
        self.env = UrbanEnvironment(width=env_width, height=env_height, resolution=1.0)
        ObstacleGenerator.add_perimeter_walls(self.env, thickness_cells=1)
        w, h = env_width, env_height
        ObstacleGenerator.add_box_obstacle(self.env, w * 0.2, h * 0.2, w * 0.45, h * 0.45)
        ObstacleGenerator.add_box_obstacle(self.env, w * 0.55, h * 0.2, w * 0.8, h * 0.45)
        ObstacleGenerator.add_box_obstacle(self.env, w * 0.2, h * 0.55, w * 0.45, h * 0.8)
        ObstacleGenerator.add_box_obstacle(self.env, w * 0.55, h * 0.55, w * 0.8, h * 0.8)

        self.env.add_zone(
            UrbanZone(
                name="target_search_zone",
                bounds=(target_pos[0] - 8.0, target_pos[1] - 8.0, target_pos[0] + 8.0, target_pos[1] + 8.0),
                vehicle_prior_probability=0.85,
            )
        )

        # 2. Vehicle Database & Perception Models
        self.target_vehicle = TargetVehicle(
            target_id="target_stolen_sedan",
            plate_number=target_plate,
            vehicle_class="sedan",
            color="dark_gray",
            position=target_pos,
            is_stolen=True,
        )
        self.decoy_vehicles = [
            TargetVehicle(
                target_id="decoy_suv_1",
                plate_number="AD-10294",
                vehicle_class="suv",
                color="white",
                position=(10.0, 25.0),
                is_stolen=False,
            ),
            TargetVehicle(
                target_id="decoy_sedan_2",
                plate_number="SHJ-44910",
                vehicle_class="sedan",
                color="red",
                position=(40.0, 25.0),
                is_stolen=False,
            ),
        ]
        self.all_vehicles = [self.target_vehicle] + self.decoy_vehicles

        self.matcher = MatchingEngine(match_threshold=0.80)
        self.matcher.add_target_vehicle(self.target_vehicle)

        # 3. ACO Engine Setup
        aco_cfg = ACOConfig()
        aco_cfg.stochastic.deterministic = False
        aco_cfg.stochastic.temperature = 0.5
        strategy = ACOAllocationStrategy(ACOEngine(aco_cfg))

        self.coordinator = SwarmCoordinator(
            allocation_strategy=strategy,
            environment=self.env,
        )

        # 4. Register Robot Agents with Coordinator
        start_poses = [
            Pose2D(6.0, 6.0, 0.0),
            Pose2D(6.0, 44.0, -math.pi / 2.0),
            Pose2D(44.0, 6.0, math.pi / 2.0),
            Pose2D(44.0, 44.0, math.pi),
        ]
        for idx, rid in enumerate(robot_ids):
            sp = start_poses[idx % len(start_poses)]
            agent = RobotAgent(
                robot_id=rid,
                initial_pose=sp,
                slam_provider=self.slam_provider,
                zone_search_duration=4.0,
                reporting_duration=5.0,
                detector=SimulatedVehicleDetector(max_detection_range=14.0, fov_degrees=90.0),
                ocr=SimulatedPlateRecognizer(base_ocr_confidence=0.95),
                matcher=self.matcher,
                known_vehicles=self.all_vehicles,
            )
            self.coordinator.register_robot(agent)

        # 4. ROS 2 Task Assignment Publishers & Status Subscribers
        self.task_pubs: Dict[str, rclpy.publisher.Publisher] = {}
        self.status_subs: Dict[str, rclpy.subscription.Subscription] = {}

        for rid in robot_ids:
            self.task_pubs[rid] = self.create_publisher(String, f"/{rid}/assigned_task", 10)
            self.status_subs[rid] = self.create_subscription(
                String,
                f"/{rid}/task_status",
                lambda msg, r=rid: self._on_robot_task_status(r, msg),
                10,
            )

        # Swarm telemetry and perception event subscriptions
        self.status_pub = self.create_publisher(String, "/swarm/status", 10)
        self.create_subscription(
            String,
            "/swarm/perception_events",
            self._on_perception_event,
            10,
        )

        # Main Coordination Loop at 5Hz (dt=0.2s)
        self.dt = 0.2
        self.start_time = time.time()
        self.timer = self.create_timer(self.dt, self._coordination_step)
        self.get_logger().info("GazeboSwarmCoordinatorNode initialized and running at 5Hz.")

    def _on_robot_task_status(self, robot_id: str, msg: String) -> None:
        """Handles task completion callbacks from robot controllers."""
        text = msg.data.strip()
        if text.startswith("COMPLETED:"):
            task_id = text.split(":", 1)[1]
            task = self.coordinator.task_manager.tasks.get(task_id)
            if task:
                task.mark_completed(self.current_tick * self.dt)
            agent = self.coordinator.robots.get(robot_id)
            if agent and agent.current_task and agent.current_task.task_id == task_id:
                agent.current_task.status = TaskStatus.COMPLETED
            self.coordinator.completed_tasks_count += 1
            self.coordinator._completed_task_ids.add(task_id)
            self.get_logger().info(f"SwarmCoordinator marked task '{task_id}' as COMPLETED by {robot_id}.")

    def _on_perception_event(self, msg: String) -> None:
        """Handles perception events (target sightings or confirmations)."""
        text = msg.data.strip()
        if text.startswith("TARGET_MATCH:"):
            parts = text.split(":")
            if len(parts) >= 6:
                rid = parts[1]
                plate = parts[2]
                conf = float(parts[3])
                loc = (float(parts[4]), float(parts[5]))
                if not self.coordinator.target_found:
                    self.coordinator.target_found = True
                    self.coordinator.confirmed_target_plate = plate
                    self.coordinator.confirmed_target_location = loc
                    self.coordinator.discovering_robot_id = rid
                    self.coordinator.target_discovery_time = self.current_tick * self.dt
                    self.coordinator.target_confirmation_confidence = conf

                    # Deposit massive recruitment pheromone at target location
                    self.coordinator.pheromone_updater.deposit_recruitment_pheromone(
                        loc[0], loc[1], intensity=100.0, radius=8.0
                    )
                    self.get_logger().warn(
                        f"🎯 MISSION SUCCESS: Target '{plate}' confirmed by {rid} at {loc}!"
                    )

    def _coordination_step(self) -> None:
        """Steps the SwarmX coordination engine."""
        self.current_tick += 1
        sim_time = self.current_tick * self.dt

        # 1. Failure Injection Test Scenario
        if (
            not self._failure_injected
            and self.inject_failure_tick is not None
            and self.current_tick == self.inject_failure_tick
        ):
            self._failure_injected = True
            self.get_logger().warn(
                f"[! ALERT] Injecting failure into '{self.fail_robot_id}' at tick {self.current_tick} (t={sim_time:.1f}s)..."
            )
            # Halt physical robot controller in Gazebo
            halt_pub = self.task_pubs.get(self.fail_robot_id)
            if halt_pub:
                halt_msg = String()
                halt_msg.data = "HALT"
                halt_pub.publish(halt_msg)

            # Trigger SwarmX failure detector and dynamic recovery
            self.coordinator.simulate_robot_failure(
                robot_id=self.fail_robot_id,
                timestamp=sim_time,
                reason="SIMULATED_MOTOR_FAILURE",
            )
            self.get_logger().info(
                f"[+ RECOVERY] Swarm dynamically reclaimed orphaned tasks and marked wreck obstacle."
            )

        # 2. Step SwarmX Coordination Loop
        status = self.coordinator.step_coordination(timestamp=sim_time, dt=self.dt)

        # 3. Dispatch assigned tasks to physical robot controllers
        for rid, agent in self.coordinator.robots.items():
            if not agent.is_alive or agent.current_state == RobotState.FAILED:
                continue

            if agent.current_task and agent.current_task.status in (TaskStatus.ASSIGNED, TaskStatus.IN_PROGRESS):
                tx, ty = agent.current_task.centroid_x, agent.current_task.centroid_y
                tmsg = String()
                tmsg.data = f"TASK:{agent.current_task.task_id}:{tx:.2f}:{ty:.2f}"
                self.task_pubs[rid].publish(tmsg)

        # 4. Publish Swarm Telemetry Status
        status_msg = String()
        status_msg.data = (
            f"TICK:{self.current_tick}|TIME:{sim_time:.1f}|"
            f"ACTIVE:{status['robot_count'] - status['failed_robots_count']}/{status['robot_count']}|"
            f"COV:{status['average_exploration_ratio']*100:.1f}%|"
            f"TASKS_DONE:{status['completed_tasks_count']}|"
            f"TARGET:{'FOUND' if status['target_found'] else 'SEARCHING'}"
        )
        self.status_pub.publish(status_msg)
