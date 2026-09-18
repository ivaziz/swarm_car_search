"""
Task Manager for Ingesting SLAM Frontiers, Deduplication, and Lifecycle Tracking.
"""

from typing import Dict, List, Optional
import math
from interfaces.slam_interface import FrontierCluster
from .frontier_task import SearchTask, TaskStatus, TaskType


class TaskManager:
    """
    Centralizes search task management for the multi-robot swarm:
    - Converts raw SLAM FrontierClusters into SearchTasks.
    - Applies spatial deduplication to prevent near-identical overlapping tasks.
    - Tracks assignments, active executions, and completions.
    - Prunes expired and completed tasks.
    """

    def __init__(
        self,
        deduplication_radius: float = 3.0,
        default_ttl_seconds: float = 120.0,
    ) -> None:
        self.deduplication_radius = deduplication_radius
        self.default_ttl_seconds = default_ttl_seconds
        self.tasks: Dict[str, SearchTask] = {}

    def is_spatially_duplicate(self, centroid_x: float, centroid_y: float) -> bool:
        """
        Checks if a frontier is within deduplication_radius of an existing
        PENDING, ASSIGNED, or IN_PROGRESS task.
        """
        for task in self.tasks.values():
            if task.status in (TaskStatus.PENDING, TaskStatus.ASSIGNED, TaskStatus.IN_PROGRESS):
                dist = math.hypot(task.centroid_x - centroid_x, task.centroid_y - centroid_y)
                if dist < self.deduplication_radius:
                    return True
        return False

    def ingest_frontiers(
        self,
        frontiers: List[FrontierCluster],
        timestamp: float,
        detection_probability: float = 0.5,
        priority: float = 1.0,
    ) -> List[SearchTask]:
        """
        Processes new frontier clusters reported by SLAM, creating SearchTasks
        for clusters that do not overlap with currently tracked tasks.
        """
        new_tasks: List[SearchTask] = []

        for frontier in frontiers:
            if not self.is_spatially_duplicate(frontier.centroid_x, frontier.centroid_y):
                task = SearchTask.from_frontier(
                    frontier=frontier,
                    timestamp=timestamp,
                    detection_probability=detection_probability,
                    priority=priority,
                    ttl_seconds=self.default_ttl_seconds,
                )
                self.tasks[task.task_id] = task
                new_tasks.append(task)

        return new_tasks

    def create_or_update_target_task(
        self,
        x: float,
        y: float,
        timestamp: float,
        target_probability: float = 0.85,
        priority: float = 2.5,
        vehicle_id: Optional[str] = None,
        task_type: TaskType = TaskType.TARGET_CONFIRMATION,
    ) -> SearchTask:
        """
        Creates a high-priority target investigation or confirmation task,
        or escalates an existing nearby task.
        """
        # Check if an existing task covers this location
        for task in self.tasks.values():
            if task.status in (TaskStatus.PENDING, TaskStatus.ASSIGNED, TaskStatus.IN_PROGRESS):
                dist = math.hypot(task.centroid_x - x, task.centroid_y - y)
                if dist < max(self.deduplication_radius, 4.0):
                    # Escalate existing task
                    task.detection_probability = max(task.detection_probability, target_probability)
                    task.priority = max(task.priority, priority)
                    task.task_type = task_type
                    if vehicle_id:
                        task.target_vehicle_id = vehicle_id
                    task.last_updated = timestamp
                    return task

        # Otherwise construct brand new target task
        new_task = SearchTask.from_vehicle_detection(
            x=x,
            y=y,
            timestamp=timestamp,
            target_probability=target_probability,
            priority=priority,
            vehicle_id=vehicle_id,
            task_type=task_type,
            ttl_seconds=self.default_ttl_seconds * 2,
        )
        self.tasks[new_task.task_id] = new_task
        return new_task

    def get_available_tasks(self, current_time: Optional[float] = None) -> List[SearchTask]:
        """Retrieves all non-expired tasks currently awaiting allocation."""
        available: List[SearchTask] = []
        for task in self.tasks.values():
            if task.status == TaskStatus.PENDING:
                if current_time is not None and task.is_expired(current_time):
                    continue
                available.append(task)
        return available

    def assign_task(self, task_id: str, robot_id: str, timestamp: float) -> bool:
        """Marks task as assigned to the specified robot."""
        if task_id in self.tasks:
            self.tasks[task_id].mark_assigned(robot_id, timestamp)
            return True
        return False

    def complete_task(self, task_id: str, timestamp: float) -> bool:
        """Marks task as completed."""
        if task_id in self.tasks:
            self.tasks[task_id].mark_completed(timestamp)
            return True
        return False

    def release_task(self, task_id: str, timestamp: float) -> bool:
        """Releases assigned task back to PENDING."""
        if task_id in self.tasks:
            self.tasks[task_id].mark_released(timestamp)
            return True
        return False

    def get_robot_assigned_task(self, robot_id: str) -> Optional[SearchTask]:
        """Finds task currently assigned to robot_id."""
        for task in self.tasks.values():
            if task.assigned_robot == robot_id and task.status in (TaskStatus.ASSIGNED, TaskStatus.IN_PROGRESS):
                return task
        return None

    def prune_tasks(self, current_time: float, completed_retention_sec: float = 60.0) -> int:
        """
        Removes expired pending tasks and stale completed tasks to conserve memory.
        Returns the number of pruned tasks.
        """
        to_delete: List[str] = []
        for tid, task in self.tasks.items():
            # Prune expired pending tasks
            if task.status == TaskStatus.PENDING and task.is_expired(current_time):
                to_delete.append(tid)
            # Prune old completed or cancelled tasks
            elif task.status in (TaskStatus.COMPLETED, TaskStatus.CANCELLED):
                if (current_time - task.last_updated) > completed_retention_sec:
                    to_delete.append(tid)

        for tid in to_delete:
            del self.tasks[tid]

        return len(to_delete)
