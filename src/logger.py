"""
logger.py - Professional Logger Engine Module

This module is the PERSISTENCE LAYER of the Smart Crowd Management System.
It answers the question:

    "What happened during this monitoring session, and how can it be retrieved later?"

Logger is NOT a print module. It is part of the system architecture.
Its purpose is to persist the operational history.

Architecture:
    CounterState + OccupancyState + RiskState + AlertState → Logger Engine → Log Files
"""

from __future__ import annotations

import csv
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Any

from .counter import CounterState
from .occupancy import OccupancyState
from .risk import RiskState
from .alerts import AlertState, Alert, AlertStatus

# Configure logging
logger = logging.getLogger(__name__)


# ============================================================================
# LoggerState
# ============================================================================

@dataclass
class LoggerState:
    """
    State of the logger engine at any point.
    
    This is the PRIMARY interface for other modules.
    All logger information is exposed through this object.
    
    Attributes:
        session_start: When the monitoring session started
        session_duration: Duration of the current session in seconds
        frames_logged: Number of frames logged
        events_logged: Number of events logged
        alerts_logged: Number of alerts logged
        errors_logged: Number of errors logged
        log_directory: Directory where logs are stored
        is_active: Whether logging is active
        timestamp: When the state was computed
    """
    session_start: datetime = field(default_factory=datetime.now)
    session_duration: float = 0.0
    frames_logged: int = 0
    events_logged: int = 0
    alerts_logged: int = 0
    errors_logged: int = 0
    log_directory: str = "logs"
    is_active: bool = True
    timestamp: datetime = field(default_factory=datetime.now)


# ============================================================================
# Logger Engine
# ============================================================================

class Logger:
    """
    Professional Logger Engine.
    
    Persists the operational history of the Smart Crowd Management System.
    
    Features:
        - Event-aware logging (logs only when something changes)
        - Session management
        - Alert logging
        - Performance logging
        - Error logging
        - File management
    
    Attributes:
        log_dir: Directory for log files
        event_buffer: Buffer of events to write
        alert_buffer: Buffer of alerts to write
        session_start: When the session started
        frame_count: Total frames processed
        state: Current logger state
        
        # State tracking for event-aware logging
        _last_people_count: Last logged people count
        _last_occupancy_status: Last logged occupancy status
        _last_risk_level: Last logged risk level
        _last_alert_state: Last logged alert state
        _heartbeat_counter: Counter for periodic heartbeat
        _heartbeat_interval: Frames between heartbeats
    """
    
    # Default heartbeat interval (10 seconds at 30 FPS)
    DEFAULT_HEARTBEAT_INTERVAL = 300
    
    def __init__(
        self,
        log_dir: str = "logs",
        heartbeat_interval: int = DEFAULT_HEARTBEAT_INTERVAL,
        buffer_size: int = 100
    ) -> None:
        """
        Initialize the Logger Engine.
        
        Args:
            log_dir: Directory for log files
            heartbeat_interval: Frames between heartbeat logs
            buffer_size: Number of events to buffer before writing
        """
        self.log_dir = Path(log_dir)
        self.heartbeat_interval = heartbeat_interval
        self.buffer_size = buffer_size
        
        # Create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Session management
        self.session_start = datetime.now()
        self.frame_count = 0
        
        # Buffers
        self.event_buffer: List[Dict] = []
        self.alert_buffer: List[Dict] = []
        self.error_buffer: List[Dict] = []
        
        # State tracking for event-aware logging
        self._last_people_count: Optional[int] = None
        self._last_occupancy_status: Optional[str] = None
        self._last_risk_level: Optional[str] = None
        self._last_alert_count: Optional[int] = None
        self._heartbeat_counter = 0
        
        # Statistics
        self.frames_logged = 0
        self.events_logged = 0
        self.alerts_logged = 0
        self.errors_logged = 0
        
        # State
        self.state = LoggerState(
            session_start=self.session_start,
            log_directory=str(self.log_dir)
        )
        
        # Initialize log files
        self._initialize_log_files()
        
        logger.info(
            f"Logger Engine initialized: log_dir='{log_dir}', "
            f"heartbeat={heartbeat_interval} frames"
        )
    
    # ========================================================================
    # Public Interface
    # ========================================================================
    
    def log(
        self,
        counter_state: CounterState,
        occupancy_state: OccupancyState,
        risk_state: RiskState,
        alert_state: AlertState,
        fps: float = 0.0,
        processing_time: float = 0.0
    ) -> LoggerState:
        """
        Log system state.
        
        This is the MAIN ENTRY POINT of the logger engine.
        
        Uses event-aware logging:
            - Logs when people count changes
            - Logs when occupancy status changes
            - Logs when risk level changes
            - Logs when alerts change
            - Logs periodic heartbeats
        
        Args:
            counter_state: CounterState from Counter
            occupancy_state: OccupancyState from Occupancy
            risk_state: RiskState from Risk
            alert_state: AlertState from Alerts
            fps: Current FPS
            processing_time: Frame processing time in milliseconds
            
        Returns:
            LoggerState containing current logger state
        """
        self.frame_count += 1
        self._heartbeat_counter += 1
        
        # Check if anything changed or heartbeat needed
        should_log = self._should_log(
            counter_state,
            occupancy_state,
            risk_state,
            alert_state
        )
        
        if should_log:
            # Log event
            self._log_event(
                counter_state,
                occupancy_state,
                risk_state,
                alert_state,
                fps,
                processing_time
            )
            self.frames_logged += 1
        
        # Log alerts if they exist
        if alert_state.active_alerts:
            self._log_alerts(alert_state)
        
        # Flush buffers if needed
        if len(self.event_buffer) >= self.buffer_size:
            self._flush_events()
        
        if len(self.alert_buffer) >= self.buffer_size:
            self._flush_alerts()
        
        # Update state
        self.state = self._build_state()
        
        return self.state
    
    def log_error(
        self,
        error_message: str,
        severity: str = "ERROR",
        exception: Optional[Exception] = None
    ) -> None:
        """
        Log an error.
        
        Args:
            error_message: Description of the error
            severity: Severity level (ERROR, WARNING, CRITICAL)
            exception: Exception object if available
        """
        error_entry = {
            'timestamp': datetime.now().isoformat(),
            'frame_number': self.frame_count,
            'severity': severity,
            'message': error_message,
            'exception': str(exception) if exception else None
        }
        
        self.error_buffer.append(error_entry)
        self.errors_logged += 1
        
        # Flush errors immediately
        self._flush_errors()
        
        # Also log to Python logger
        if severity == "CRITICAL":
            logger.critical(error_message)
        elif severity == "ERROR":
            logger.error(error_message)
        elif severity == "WARNING":
            logger.warning(error_message)
        else:
            logger.info(error_message)
    
    def start_session(self) -> None:
        """
        Start a new logging session.
        
        Creates a new session log file.
        """
        self.session_start = datetime.now()
        self.frame_count = 0
        self.frames_logged = 0
        self.events_logged = 0
        self.alerts_logged = 0
        self.errors_logged = 0
        
        self._last_people_count = None
        self._last_occupancy_status = None
        self._last_risk_level = None
        self._last_alert_count = None
        self._heartbeat_counter = 0
        
        self._initialize_log_files()
        self._log_session_start()
        
        logger.info("New logging session started")
    
    def end_session(self) -> None:
        """
        End the current logging session.
        
        Flushes all buffers and logs session end.
        """
        # Flush remaining buffers
        self._flush_events()
        self._flush_alerts()
        self._flush_errors()
        
        # Log session end
        self._log_session_end()
        
        logger.info(
            f"Session ended: {self.frames_logged} frames, "
            f"{self.events_logged} events, {self.alerts_logged} alerts"
        )
    
    def reset(self) -> None:
        """
        Reset the logger engine.
        
        Ends the current session and starts a new one.
        """
        self.end_session()
        self.start_session()
        logger.info("Logger Engine reset")
    
    # ========================================================================
    # Private Methods
    # ========================================================================
    
    def _initialize_log_files(self) -> None:
        """
        Initialize log files with headers.
        """
        # Events CSV
        events_path = self.log_dir / "events.csv"
        if not events_path.exists():
            with open(events_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'frame', 'people', 'occupancy_percent',
                    'capacity', 'remaining', 'status', 'risk_level',
                    'trend', 'is_over_capacity', 'fps', 'processing_time_ms'
                ])
        
        # Alerts CSV
        alerts_path = self.log_dir / "alerts.csv"
        if not alerts_path.exists():
            with open(alerts_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'frame', 'alert_id', 'alert_type',
                    'alert_level', 'status', 'message', 'risk_level',
                    'occupancy_percent'
                ])
        
        # Errors CSV
        errors_path = self.log_dir / "errors.csv"
        if not errors_path.exists():
            with open(errors_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'frame', 'severity', 'message', 'exception'
                ])
    
    def _should_log(
        self,
        counter_state: CounterState,
        occupancy_state: OccupancyState,
        risk_state: RiskState,
        alert_state: AlertState
    ) -> bool:
        """
        Determine if the current state should be logged.
        
        Event-aware logging:
            - Log when people count changes
            - Log when occupancy status changes
            - Log when risk level changes
            - Log when alert count changes
            - Log on heartbeat interval
        
        Returns:
            True if logging is needed
        """
        # Check for changes
        people_changed = (
            self._last_people_count is None or
            counter_state.current_count != self._last_people_count
        )
        
        status_changed = (
            self._last_occupancy_status is None or
            occupancy_state.status.value != self._last_occupancy_status
        )
        
        risk_changed = (
            self._last_risk_level is None or
            risk_state.level.value != self._last_risk_level
        )
        
        alert_count = len(alert_state.active_alerts)
        alert_changed = (
            self._last_alert_count is None or
            alert_count != self._last_alert_count
        )
        
        # Heartbeat
        heartbeat = self._heartbeat_counter >= self.heartbeat_interval
        
        # Update tracking
        if people_changed:
            self._last_people_count = counter_state.current_count
        
        if status_changed:
            self._last_occupancy_status = occupancy_state.status.value
        
        if risk_changed:
            self._last_risk_level = risk_state.level.value
        
        if alert_changed:
            self._last_alert_count = alert_count
        
        if heartbeat:
            self._heartbeat_counter = 0
        
        return people_changed or status_changed or risk_changed or alert_changed or heartbeat
    
    def _log_event(
        self,
        counter_state: CounterState,
        occupancy_state: OccupancyState,
        risk_state: RiskState,
        alert_state: AlertState,
        fps: float,
        processing_time: float
    ) -> None:
        """
        Log an event to the event buffer.
        
        Args:
            counter_state: CounterState
            occupancy_state: OccupancyState
            risk_state: RiskState
            alert_state: AlertState
            fps: Current FPS
            processing_time: Processing time in milliseconds
        """
        event = {
            'timestamp': datetime.now().isoformat(),
            'frame': counter_state.frame_number,
            'people': counter_state.current_count,
            'occupancy_percent': round(occupancy_state.occupancy_percentage, 1),
            'capacity': occupancy_state.capacity,
            'remaining': occupancy_state.remaining_capacity,
            'status': occupancy_state.status.value,
            'risk_level': risk_state.level.value,
            'trend': occupancy_state.trend.value,
            'is_over_capacity': occupancy_state.is_over_capacity,
            'fps': round(fps, 1),
            'processing_time_ms': round(processing_time, 2)
        }
        
        self.event_buffer.append(event)
        self.events_logged += 1
    
    def _log_alerts(self, alert_state: AlertState) -> None:
        """
        Log alerts to the alert buffer.
        
        Only logs new alerts or alerts that have changed status.
        
        Args:
            alert_state: AlertState
        """
        # Track which alerts we've already logged
        logged_ids = {a.get('alert_id', '') for a in self.alert_buffer}
        
        for alert in alert_state.active_alerts:
            if alert.alert_id not in logged_ids:
                alert_entry = {
                    'timestamp': datetime.now().isoformat(),
                    'frame': alert_state.frame_number,
                    'alert_id': alert.alert_id,
                    'alert_type': alert.alert_type.value,
                    'alert_level': alert.level.value,
                    'status': alert.status.value,
                    'message': alert.message,
                    'risk_level': alert.risk_level.value if alert.risk_level else 'UNKNOWN',
                    'occupancy_percent': alert.metadata.get('created_occupancy', 0)
                }
                
                self.alert_buffer.append(alert_entry)
                self.alerts_logged += 1
                logged_ids.add(alert.alert_id)
    
    def _flush_events(self) -> None:
        """
        Flush event buffer to CSV file.
        """
        if not self.event_buffer:
            return
        
        events_path = self.log_dir / "events.csv"
        file_exists = events_path.exists()
        
        with open(events_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.event_buffer[0].keys())
            if not file_exists:
                writer.writeheader()
            writer.writerows(self.event_buffer)
        
        self.event_buffer.clear()
    
    def _flush_alerts(self) -> None:
        """
        Flush alert buffer to CSV file.
        """
        if not self.alert_buffer:
            return
        
        alerts_path = self.log_dir / "alerts.csv"
        file_exists = alerts_path.exists()
        
        with open(alerts_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.alert_buffer[0].keys())
            if not file_exists:
                writer.writeheader()
            writer.writerows(self.alert_buffer)
        
        self.alert_buffer.clear()
    
    def _flush_errors(self) -> None:
        """
        Flush error buffer to CSV file.
        """
        if not self.error_buffer:
            return
        
        errors_path = self.log_dir / "errors.csv"
        file_exists = errors_path.exists()
        
        with open(errors_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.error_buffer[0].keys())
            if not file_exists:
                writer.writeheader()
            writer.writerows(self.error_buffer)
        
        self.error_buffer.clear()
    
    def _log_session_start(self) -> None:
        """
        Log session start to the session log file.
        """
        session_path = self.log_dir / "session.log"
        with open(session_path, 'a') as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"SESSION START: {self.session_start.isoformat()}\n")
            f.write(f"{'='*60}\n")
    
    def _log_session_end(self) -> None:
        """
        Log session end to the session log file.
        """
        session_path = self.log_dir / "session.log"
        end_time = datetime.now()
        duration = (end_time - self.session_start).total_seconds()
        
        with open(session_path, 'a') as f:
            f.write(f"\nSESSION END: {end_time.isoformat()}\n")
            f.write(f"DURATION: {duration:.1f} seconds\n")
            f.write(f"FRAMES: {self.frames_logged}\n")
            f.write(f"EVENTS: {self.events_logged}\n")
            f.write(f"ALERTS: {self.alerts_logged}\n")
            f.write(f"ERRORS: {self.errors_logged}\n")
            f.write(f"{'='*60}\n")
    
    def _build_state(self) -> LoggerState:
        """
        Build the current logger state.
        
        Returns:
            LoggerState object
        """
        duration = (datetime.now() - self.session_start).total_seconds()
        
        return LoggerState(
            session_start=self.session_start,
            session_duration=duration,
            frames_logged=self.frames_logged,
            events_logged=self.events_logged,
            alerts_logged=self.alerts_logged,
            errors_logged=self.errors_logged,
            log_directory=str(self.log_dir),
            is_active=True,
            timestamp=datetime.now()
        )
    
    # ========================================================================
    # Public Utility Methods
    # ========================================================================
    
    def get_log_files(self) -> List[str]:
        """
        Get list of all log files.
        
        Returns:
            List of filenames in the log directory
        """
        return [f.name for f in self.log_dir.iterdir() if f.is_file()]
    
    def get_session_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current session.
        
        Returns:
            Dictionary with session summary
        """
        duration = (datetime.now() - self.session_start).total_seconds()
        
        return {
            'session_start': self.session_start.isoformat(),
            'session_duration_seconds': duration,
            'frames_logged': self.frames_logged,
            'events_logged': self.events_logged,
            'alerts_logged': self.alerts_logged,
            'errors_logged': self.errors_logged,
            'average_fps': self.frames_logged / duration if duration > 0 else 0
        }
    
    def flush(self) -> None:
        """
        Force flush all buffers to disk.
        """
        self._flush_events()
        self._flush_alerts()
        self._flush_errors()
    
    def __repr__(self) -> str:
        """String representation of the logger engine."""
        return (f"Logger(events={self.events_logged}, "
                f"alerts={self.alerts_logged}, "
                f"errors={self.errors_logged})")