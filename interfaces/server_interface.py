"""
Abstract Interface for External Cloud Server / Dashboard Reporting.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional
from .perception_interface import MatchEvent


@dataclass
class DetectionReport:
    """Report dispatched to remote dashboard upon confirmed target vehicle discovery."""
    report_id: str
    match_event: MatchEvent
    evidence_url: Optional[str] = None
    swarm_status_summary: Optional[Dict[str, Any]] = None


@dataclass
class MissionReport:
    """Periodic telemetry report on swarm search progress and coverage."""
    mission_id: str
    timestamp: float
    total_coverage_ratio: float
    active_robot_count: int
    elapsed_time_sec: float
    confirmed_targets_found: int


class IServerReporter(ABC):
    """Abstract interface for reporting alerts and telemetry to external base station."""

    @abstractmethod
    def report_match(self, report: DetectionReport) -> bool:
        """Transmits target match evidence and alert to cloud dashboard."""
        pass

    @abstractmethod
    def send_telemetry(self, telemetry: MissionReport) -> bool:
        """Sends periodic mission coverage and swarm health telemetry."""
        pass
