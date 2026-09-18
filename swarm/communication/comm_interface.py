"""
Abstract Interface for Swarm Inter-Robot Communication Backends.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
from .swarm_messages import SwarmMessage


class ICommunicationChannel(ABC):
    """
    Abstract contract defining messaging primitives across the robot swarm.
    Permits transparent swapping between in-memory queues, wireless network simulations,
    and ROS 2 DDS middleware.
    """

    @abstractmethod
    def register_node(self, node_id: str) -> None:
        """Enrolls a robot node into the communication network."""
        pass

    @abstractmethod
    def update_node_pose(self, node_id: str, x: float, y: float) -> None:
        """Updates spatial location of node for range-dependent wireless propagation."""
        pass

    @abstractmethod
    def broadcast(self, message: SwarmMessage) -> None:
        """Transmits message to all active peer nodes in the swarm."""
        pass

    @abstractmethod
    def send_direct(self, recipient_id: str, message: SwarmMessage) -> None:
        """Sends targeted unicast message to a specific peer node."""
        pass

    @abstractmethod
    def receive_inbox(self, node_id: str) -> List[SwarmMessage]:
        """Retrieves and clears all pending messages for the specified node."""
        pass

    @abstractmethod
    def step(self, timestamp: float) -> None:
        """Advances network simulation clocks, routing packets subject to latency."""
        pass
