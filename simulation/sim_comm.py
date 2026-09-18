"""
Realistic Wireless Ad-Hoc Communication Simulator with Range-Limits, Packet Loss, and Latency.
"""

from typing import Dict, List, Optional, Tuple
import math
import random
from swarm.communication.comm_interface import ICommunicationChannel
from swarm.communication.swarm_messages import SwarmMessage


class SimulatedNetworkChannel(ICommunicationChannel):
    """
    Simulates realistic wireless ad-hoc RF channel characteristics:
    - Spatial distance cutoff (communication range limit).
    - Statistical packet drop / loss (noise, multipath fading).
    - Transmission latency / propagation delay queue.
    - Full telemetry statistics on packet delivery performance.
    """

    def __init__(
        self,
        communication_range: Optional[float] = 30.0,
        packet_loss_rate: float = 0.05,
        latency_seconds: float = 0.05,
        seed: Optional[int] = None,
    ) -> None:
        self.communication_range = communication_range
        self.packet_loss_rate = packet_loss_rate
        self.latency_seconds = latency_seconds
        self.rng = random.Random(seed)

        self._nodes: set = set()
        self._node_poses: Dict[str, Tuple[float, float]] = {}
        self._inboxes: Dict[str, List[SwarmMessage]] = {}

        # Transit queue: list of (deliver_at_time, recipient_id, SwarmMessage)
        self._transit_queue: List[Tuple[float, str, SwarmMessage]] = []

        # Diagnostics & Metrics
        self.stats = {
            "total_sent": 0,
            "delivered": 0,
            "dropped_range": 0,
            "dropped_loss": 0,
        }

    def register_node(self, node_id: str) -> None:
        """Enrolls a node in the simulated RF network."""
        self._nodes.add(node_id)
        if node_id not in self._inboxes:
            self._inboxes[node_id] = []

    def update_node_pose(self, node_id: str, x: float, y: float) -> None:
        """Updates known spatial position of robot node."""
        self._node_poses[node_id] = (x, y)

    def _is_in_range(self, sender_id: str, recipient_id: str) -> bool:
        """Determines if two nodes are within line-of-sight RF transmission distance."""
        if self.communication_range is None:
            return True

        p1 = self._node_poses.get(sender_id)
        p2 = self._node_poses.get(recipient_id)
        if p1 is None or p2 is None:
            # If coordinates unknown, assume out of range
            return False

        dist = math.hypot(p1[0] - p2[0], p1[1] - p2[1])
        return dist <= self.communication_range

    def _route_packet(self, recipient_id: str, message: SwarmMessage) -> bool:
        """Evaluates RF link conditions and enqueues packet if successfully transmitted."""
        self.stats["total_sent"] += 1

        # Check physical distance
        if not self._is_in_range(message.sender_id, recipient_id):
            self.stats["dropped_range"] += 1
            return False

        # Check probabilistic packet drop
        if self.packet_loss_rate > 0.0 and self.rng.random() < self.packet_loss_rate:
            self.stats["dropped_loss"] += 1
            return False

        deliver_at = message.timestamp + self.latency_seconds
        self._transit_queue.append((deliver_at, recipient_id, message))
        return True

    def broadcast(self, message: SwarmMessage) -> None:
        """Broadcasts packet to all peers within RF range."""
        for node in self._nodes:
            if node != message.sender_id:
                self._route_packet(node, message)

    def send_direct(self, recipient_id: str, message: SwarmMessage) -> None:
        """Unicast transmission to recipient_id."""
        if recipient_id in self._nodes:
            self._route_packet(recipient_id, message)

    def step(self, timestamp: float) -> None:
        """
        Advances network time: drains transit queue and delivers packets whose
        delivery time has arrived into respective mailboxes.
        """
        remaining: List[Tuple[float, str, SwarmMessage]] = []

        for deliver_at, recipient_id, msg in self._transit_queue:
            if deliver_at <= timestamp:
                if recipient_id in self._inboxes:
                    self._inboxes[recipient_id].append(msg)
                    self.stats["delivered"] += 1
            else:
                remaining.append((deliver_at, recipient_id, msg))

        self._transit_queue = remaining

    def receive_inbox(self, node_id: str) -> List[SwarmMessage]:
        """Retrieves and clears delivered packets in mailbox."""
        if node_id not in self._inboxes:
            return []
        packets = list(self._inboxes[node_id])
        self._inboxes[node_id].clear()
        return packets
