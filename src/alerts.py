"""
alerts.py - Professional Alert Engine Module

This module manages the alert lifecycle for the Smart Crowd Management System.
It answers the question:

    "Given the current risk assessment, should the system create, update,
     maintain, or resolve an alert?"

Alerts is NOT a notification module. It is a managed event system.

Architecture:
    RiskState → Alert Engine → AlertState
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import List, Optional, Dict, Set

from .risk import RiskState, RiskLevel, RiskFactor

# Configure logging
logger = logging.getLogger(__name__)


# ============================================================================
# Enums
# ============================================================================

class AlertLevel(Enum):
    """
    Alert urgency levels.
    
    INFO     → Informational, no immediate action needed
    WARNING  → Caution, monitor the situation
    CRITICAL → Immediate action required
    """
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class AlertType(Enum):
    """
    Types of alerts that can be generated.
    
    Each alert type represents a specific operational event.
    New types can be added without changing the architecture.
    """
    NONE = "NONE"
    HIGH_OCCUPANCY = "HIGH_OCCUPANCY"
    CAPACITY_LIMIT = "CAPACITY_LIMIT"
    OVER_CAPACITY = "OVER_CAPACITY"
    RAPID_GROWTH = "RAPID_GROWTH"


class AlertStatus(Enum):
    """
    Lifecycle status of an alert.
    
    ACTIVE      → Currently ongoing
    RESOLVED    → Condition has cleared
    ACKNOWLEDGED → Operator has acknowledged
    ARCHIVED    → No longer needed for active display
    """
    ACTIVE = "ACTIVE"
    RESOLVED = "RESOLVED"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    ARCHIVED = "ARCHIVED"


# ============================================================================
# Alert Dataclass
# ============================================================================

@dataclass
class Alert:
    """
    Individual alert object.
    
    Each alert has a lifecycle and contains all information needed
    for display, logging, and management.
    
    Attributes:
        alert_id: Unique identifier for the alert
        alert_type: Type of alert (HIGH_OCCUPANCY, etc.)
        level: Urgency level (INFO, WARNING, CRITICAL)
        message: Human-readable alert message
        created_at: When the alert was created
        resolved_at: When the alert was resolved (if applicable)
        acknowledged_at: When the alert was acknowledged (if applicable)
        frame_number: Frame number when alert was created
        status: Current lifecycle status
        risk_level: Risk level that triggered the alert
        metadata: Additional information about the alert
    """
    alert_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    alert_type: AlertType = AlertType.NONE
    level: AlertLevel = AlertLevel.INFO
    message: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    resolved_at: Optional[datetime] = None
    acknowledged_at: Optional[datetime] = None
    frame_number: int = 0
    status: AlertStatus = AlertStatus.ACTIVE
    risk_level: Optional[RiskLevel] = None
    metadata: dict = field(default_factory=dict)


# ============================================================================
# AlertState
# ============================================================================

@dataclass
class AlertState:
    """
    State of the alert engine at any point.
    
    This is the PRIMARY interface for other modules.
    All alert information is exposed through this object.
    
    Attributes:
        active_alerts: List of currently active alerts
        resolved_alerts: List of recently resolved alerts (for display)
        highest_level: Highest alert level among active alerts
        newest_alert: The most recently created alert (if any)
        total_active: Number of active alerts
        total_resolved: Total resolved alerts (history)
        frame_number: Current frame number
        timestamp: When the state was computed
    """
    active_alerts: List[Alert] = field(default_factory=list)
    resolved_alerts: List[Alert] = field(default_factory=list)
    highest_level: AlertLevel = AlertLevel.INFO
    newest_alert: Optional[Alert] = None
    total_active: int = 0
    total_resolved: int = 0
    frame_number: int = 0
    timestamp: datetime = field(default_factory=datetime.now)


# ============================================================================
# Alert Engine
# ============================================================================

class Alerts:
    """
    Professional Alert Engine.
    
    Manages the complete alert lifecycle including:
        - Risk evaluation
        - Alert type determination
        - Priority assignment
        - Duplicate suppression
        - Lifecycle management (cooldown)
        - History maintenance
    
    Features:
        - Alert hysteresis (cooldown) to prevent flapping
        - Stateful alert management
        - O(1) time complexity
    
    Attributes:
        cooldown_frames: Number of frames to wait before recreating a resolved alert
        active_alerts: Dictionary mapping alert_type to current Alert
        alert_history: List of all resolved/archived alerts
        frame_number: Current frame number
        state: Current alert state
        previous_risk_level: Previous risk level for change detection
    """
    
    # Default cooldown (5 seconds at 30 FPS = 150 frames)
    DEFAULT_COOLDOWN_FRAMES = 150
    
    # Alert type mapping (Risk level → Alert type)
    RISK_TO_ALERT_MAP = {
        RiskLevel.NORMAL: AlertType.NONE,
        RiskLevel.ELEVATED: AlertType.HIGH_OCCUPANCY,
        RiskLevel.HIGH: AlertType.HIGH_OCCUPANCY,
        RiskLevel.CRITICAL: AlertType.OVER_CAPACITY
    }
    
    def __init__(
        self,
        cooldown_frames: int = DEFAULT_COOLDOWN_FRAMES,
        max_history: int = 1000
    ) -> None:
        """
        Initialize the Alert Engine.
        
        Args:
            cooldown_frames: Frames to wait before recreating a resolved alert
            max_history: Maximum number of alerts to keep in history
        """
        self.cooldown_frames = cooldown_frames
        self.max_history = max_history
        
        # Active alerts (only one per type)
        self.active_alerts: Dict[AlertType, Alert] = {}
        
        # History of resolved alerts
        self.alert_history: List[Alert] = []
        
        # Tracking for cooldown
        self.resolved_frame: Dict[AlertType, int] = {}
        
        # State
        self.frame_number = 0
        self.state = AlertState()
        self.previous_risk_level: Optional[RiskLevel] = None
        
        logger.info(
            f"Alert Engine initialized: cooldown={cooldown_frames} frames, "
            f"max_history={max_history}"
        )
    
    # ========================================================================
    # Public Interface
    # ========================================================================
    
    def update(self, risk_state: RiskState) -> AlertState:
        """
        Update alert state with new risk assessment.
        
        This is the MAIN ENTRY POINT of the alert engine.
        
        Workflow:
            1. Evaluate risk
            2. Determine alert type
            3. Determine alert priority
            4. Suppress duplicates (cooldown)
            5. Update alert lifecycle
            6. Update history
            7. Build state
        
        Args:
            risk_state: RiskState from Risk module
            
        Returns:
            AlertState containing current alert state
        """
        self.frame_number = risk_state.frame_number
        
        # Stage 1: Evaluate risk
        alert_type = self._evaluate_risk(risk_state)
        
        # Stage 2 & 3: Determine type and priority
        if alert_type != AlertType.NONE:
            alert_level = self._determine_level(risk_state)
            message = self._generate_message(alert_type, risk_state)
            
            # Stage 4: Duplicate suppression (cooldown check)
            if self._can_create_alert(alert_type):
                # Stage 5: Update lifecycle
                self._create_or_update_alert(
                    alert_type=alert_type,
                    level=alert_level,
                    message=message,
                    risk_state=risk_state
                )
            else:
                # Alert is in cooldown, keep existing if any
                pass
        else:
            # Stage 5: Resolve active alerts if risk is NORMAL
            self._resolve_alerts(risk_state)
        
        # Stage 6: Update history
        self._update_history()
        
        # Stage 7: Build state
        self.state = self._build_state(risk_state)
        
        # Store for next iteration
        self.previous_risk_level = risk_state.level
        
        logger.debug(
            f"Alerts: active={len(self.active_alerts)}, "
            f"highest={self.state.highest_level.value}"
        )
        
        return self.state
    
    # ========================================================================
    # Private Methods (Stage 1-7)
    # ========================================================================
    
    def _evaluate_risk(self, risk_state: RiskState) -> AlertType:
        """
        Stage 1: Evaluate risk and determine appropriate alert type.
        
        Args:
            risk_state: Current risk state
            
        Returns:
            AlertType based on risk assessment
        """
        # Check for over capacity (highest priority)
        if risk_state.is_over_capacity:
            return AlertType.OVER_CAPACITY
        
        # Check for capacity limit
        if risk_state.status.value == "FULL":
            return AlertType.CAPACITY_LIMIT
        
        # Check for rapid growth
        if risk_state.primary_factor == RiskFactor.RAPID_INCREASE:
            return AlertType.RAPID_GROWTH
        
        # Map risk level to alert type
        return self.RISK_TO_ALERT_MAP.get(risk_state.level, AlertType.NONE)
    
    def _determine_level(self, risk_state: RiskState) -> AlertLevel:
        """
        Stage 2: Determine alert priority level.
        
        Args:
            risk_state: Current risk state
            
        Returns:
            AlertLevel based on risk severity
        """
        if risk_state.level == RiskLevel.CRITICAL:
            return AlertLevel.CRITICAL
        elif risk_state.level == RiskLevel.HIGH:
            return AlertLevel.WARNING
        elif risk_state.level == RiskLevel.ELEVATED:
            return AlertLevel.WARNING
        else:
            return AlertLevel.INFO
    
    def _generate_message(self, alert_type: AlertType, risk_state: RiskState) -> str:
        """
        Generate a human-readable alert message.
        
        Args:
            alert_type: Type of alert
            risk_state: Current risk state
            
        Returns:
            Alert message string
        """
        messages = {
            AlertType.HIGH_OCCUPANCY: (
                f"High occupancy detected: {risk_state.occupancy_percentage:.1f}% "
                f"({risk_state.status.value})"
            ),
            AlertType.CAPACITY_LIMIT: (
                f"At capacity limit: {risk_state.occupancy_percentage:.1f}% "
                f"({risk_state.status.value})"
            ),
            AlertType.OVER_CAPACITY: (
                f"OVER CAPACITY! {risk_state.occupancy_percentage:.1f}% "
                f"exceeds safe limit"
            ),
            AlertType.RAPID_GROWTH: (
                f"Rapid occupancy increase detected: "
                f"{risk_state.occupancy_percentage:.1f}%"
            ),
            AlertType.NONE: "No active alerts"
        }
        return messages.get(alert_type, f"Alert: {alert_type.value}")
    
    def _can_create_alert(self, alert_type: AlertType) -> bool:
        """
        Stage 3: Check if a new alert can be created.
        
        Implements cooldown mechanism to prevent alert flapping.
        
        Args:
            alert_type: Type of alert to check
            
        Returns:
            True if alert can be created, False if in cooldown
        """
        # If no alert of this type exists, create it
        if alert_type not in self.active_alerts:
            return True
        
        # If alert is already active, don't recreate
        existing = self.active_alerts.get(alert_type)
        if existing and existing.status == AlertStatus.ACTIVE:
            return False
        
        # Check cooldown
        if alert_type in self.resolved_frame:
            frames_since_resolved = self.frame_number - self.resolved_frame[alert_type]
            if frames_since_resolved < self.cooldown_frames:
                logger.debug(
                    f"Alert {alert_type.value} in cooldown: "
                    f"{frames_since_resolved}/{self.cooldown_frames} frames"
                )
                return False
        
        return True
    
    def _create_or_update_alert(
        self,
        alert_type: AlertType,
        level: AlertLevel,
        message: str,
        risk_state: RiskState
    ) -> None:
        """
        Stage 4: Create a new alert or update existing one.
        
        Args:
            alert_type: Type of alert
            level: Alert urgency level
            message: Alert message
            risk_state: Current risk state
        """
        if alert_type in self.active_alerts:
            # Update existing alert
            alert = self.active_alerts[alert_type]
            alert.message = message
            alert.level = level
            alert.risk_level = risk_state.level
            alert.metadata.update({
                'last_updated': datetime.now(),
                'frame_number': self.frame_number,
                'occupancy': risk_state.occupancy_percentage
            })
            logger.debug(f"Updated alert: {alert_type.value}")
        else:
            # Create new alert
            alert = Alert(
                alert_type=alert_type,
                level=level,
                message=message,
                frame_number=self.frame_number,
                risk_level=risk_state.level,
                metadata={
                    'created_occupancy': risk_state.occupancy_percentage,
                    'created_status': risk_state.status.value
                }
            )
            self.active_alerts[alert_type] = alert
            logger.info(f"Created alert: {alert_type.value} ({level.value})")
    
    def _resolve_alerts(self, risk_state: RiskState) -> None:
        """
        Stage 5: Resolve active alerts when risk is NORMAL.
        
        Args:
            risk_state: Current risk state
        """
        to_resolve = []
        
        for alert_type, alert in self.active_alerts.items():
            # Resolve if risk is NORMAL
            if risk_state.level == RiskLevel.NORMAL:
                alert.status = AlertStatus.RESOLVED
                alert.resolved_at = datetime.now()
                to_resolve.append(alert_type)
                self.resolved_frame[alert_type] = self.frame_number
                logger.debug(f"Resolved alert: {alert_type.value}")
        
        # Move resolved alerts to history
        for alert_type in to_resolve:
            alert = self.active_alerts.pop(alert_type)
            self.alert_history.append(alert)
    
    def _update_history(self) -> None:
        """
        Stage 6: Maintain alert history.
        
        Keeps history within max_history limit.
        """
        # Clean up old history if needed
        if len(self.alert_history) > self.max_history:
            excess = len(self.alert_history) - self.max_history
            self.alert_history = self.alert_history[excess:]
    
    def _build_state(self, risk_state: RiskState) -> AlertState:
        """
        Stage 7: Build the current alert state.
        
        Args:
            risk_state: Current risk state
            
        Returns:
            AlertState object
        """
        active_alerts = list(self.active_alerts.values())
        
        # Determine highest level
        highest_level = AlertLevel.INFO
        for alert in active_alerts:
            if alert.level == AlertLevel.CRITICAL:
                highest_level = AlertLevel.CRITICAL
                break
            elif alert.level == AlertLevel.WARNING:
                highest_level = AlertLevel.WARNING
        
        # Get newest active alert
        newest_alert = None
        if active_alerts:
            newest_alert = max(active_alerts, key=lambda a: a.created_at)
        
        return AlertState(
            active_alerts=active_alerts,
            resolved_alerts=self.alert_history[-5:],  # Last 5 for display
            highest_level=highest_level,
            newest_alert=newest_alert,
            total_active=len(active_alerts),
            total_resolved=len(self.alert_history),
            frame_number=self.frame_number,
            timestamp=datetime.now()
        )
    
    # ========================================================================
    # Public Methods
    # ========================================================================
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """
        Acknowledge a specific alert.
        
        Args:
            alert_id: ID of the alert to acknowledge
            
        Returns:
            True if acknowledged, False if not found
        """
        for alert in self.active_alerts.values():
            if alert.alert_id == alert_id:
                alert.status = AlertStatus.ACKNOWLEDGED
                alert.acknowledged_at = datetime.now()
                logger.info(f"Alert acknowledged: {alert_id}")
                return True
        
        # Check history
        for alert in self.alert_history:
            if alert.alert_id == alert_id:
                alert.status = AlertStatus.ACKNOWLEDGED
                alert.acknowledged_at = datetime.now()
                logger.info(f"Historical alert acknowledged: {alert_id}")
                return True
        
        return False
    
    def get_alert_by_type(self, alert_type: AlertType) -> Optional[Alert]:
        """
        Get the active alert of a specific type.
        
        Args:
            alert_type: Type of alert to retrieve
            
        Returns:
            Alert if active, None otherwise
        """
        return self.active_alerts.get(alert_type)
    
    def get_active_alerts(self) -> List[Alert]:
        """
        Get all active alerts.
        
        Returns:
            List of active Alert objects
        """
        return list(self.active_alerts.values())
    
    def get_alert_history(self, limit: int = 100) -> List[Alert]:
        """
        Get alert history.
        
        Args:
            limit: Maximum number of alerts to return
            
        Returns:
            List of historical alerts (most recent first)
        """
        return self.alert_history[-limit:][::-1]
    
    def clear_resolved_alerts(self) -> None:
        """
        Clear all resolved alerts from history.
        
        This can be called periodically to keep history manageable.
        """
        self.alert_history.clear()
        self.resolved_frame.clear()
        logger.info("Cleared resolved alerts")
    
    def reset(self) -> None:
        """
        Reset the alert engine state.
        
        Useful when starting a new monitoring session.
        """
        self.active_alerts.clear()
        self.alert_history.clear()
        self.resolved_frame.clear()
        self.frame_number = 0
        self.state = AlertState()
        self.previous_risk_level = None
        logger.info("Alert Engine reset")
    
    def __repr__(self) -> str:
        """String representation of the alert engine."""
        return (f"Alerts(active={len(self.active_alerts)}, "
                f"history={len(self.alert_history)}, "
                f"highest={self.state.highest_level.value})")