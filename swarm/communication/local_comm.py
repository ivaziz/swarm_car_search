"""
In-Memory Local Message Bus for Synchronous Swarm Communication.
"""

from typing import Dict, List, Optional
from .comm_interface import ICommunicationChannel
from .swarm_messages import SwarmMessage


class LocalMessageBus(ICommunicationChannel):
    """
    Zero-latency, in-memory communication channel for rapid simulation and local tests.
    """

    def __init__(self) -> None:
        self._inboxes: Dict[str, List[SwarmMessage]] = {}
        self._poses: Dict[str, tuple] = {}

    def register_node(self, node_id: str) -> None:
        """Registers a node mailbox."""
        if node_id not in self._inboxes:
            self._inboxes[node_id] = []

    def update_node_pose(self, node_id: str, x: float, y: float) -> None:
        """Stores node position."""
        self._poses[node_id] = (x, y)

    def broadcast(self, message: SwarmMessage) -> None:
        """Broadcasts message to all registered nodes except the sender."""
        for recipient_id, inbox in self._inboxes.items():
            if recipient_id != message.sender_id:
                inbox.append(message)

    def send_direct(self, recipient_id: str, message: SwarmMessage) -> None:
        """Delivers message directly into recipient inbox."""
        if recipient_id in self._inboxes:
            self._inboxes[recipient_id].append(message)

    def receive_inbox(self, node_id: str) -> List[SwarmMessage]:
        """Drains and returns all queued messages for node_id."""
        if node_id not in self._inboxes:
            return []
        messages = list(self._inboxes[node_id])
        self._inboxes[node_id].clear()
        return messages

    def step(self, timestamp: float) -> None:
        """Instantaneous delivery requires no time advancement."""
        pass
