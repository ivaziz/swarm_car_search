"""
Dynamic frontier task extraction, management, and decentralized allocation.
"""

from .frontier_task import SearchTask, TaskStatus
from .task_manager import TaskManager
from .allocation_strategy import (
    ITaskAllocationStrategy,
    ACOAllocationStrategy,
    NearestFrontierStrategy,
    RandomAllocationStrategy,
)

__all__ = [
    "SearchTask",
    "TaskStatus",
    "TaskManager",
    "ITaskAllocationStrategy",
    "ACOAllocationStrategy",
    "NearestFrontierStrategy",
    "RandomAllocationStrategy",
]
