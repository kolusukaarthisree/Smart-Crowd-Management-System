"""
Smart Crowd Management System

Core package for the Smart Crowd Management System.

Author: System Architect
Version: 1.0.0
"""

from .detector import Detector
from .counter import Counter, CounterState
from .occupancy import (
    Occupancy,
    OccupancyState,
    OccupancyStatus,
    OccupancyTrend,
)
from .risk import (
    Risk,
    RiskState,
    RiskLevel,
)
from .alerts import (
    Alerts,
    Alert,
    AlertState,
    AlertLevel,
    AlertType,
)
from .logger import (
    Logger,
    LoggerState,
)
from .dashboard import Dashboard

__all__ = [
    "Detector",
    "Counter",
    "CounterState",
    "Occupancy",
    "OccupancyState",
    "OccupancyStatus",
    "OccupancyTrend",
    "Risk",
    "RiskState",
    "RiskLevel",
    "Alerts",
    "Alert",
    "AlertState",
    "AlertLevel",
    "AlertType",
    "Logger",
    "LoggerState",
    "Dashboard",
]