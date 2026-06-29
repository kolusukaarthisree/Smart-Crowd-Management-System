"""
dashboard.py - Dashboard Module

This module is the PURE RENDERING LAYER of the Smart Crowd Management System.
It has exactly ONE responsibility:

    Render all system state onto the camera frame.

The dashboard does NOT calculate anything. It only consumes outputs from
other modules and draws them on the frame.

Architecture:
    frame + people + counter_state + occupancy_state + risk_state + alert_state
                                    ↓
                              Dashboard
                                    ↓
                           Annotated Frame

Author: System Architect
Version: 2.1.0 (FINAL - Locked)
"""

from __future__ import annotations

import cv2
import numpy as np
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

from .models.person import Person
from .counter import CounterState
from .occupancy import OccupancyState
from .risk import RiskState, RiskLevel
from .alerts import AlertState


# ============================================================================
# Professional Surveillance Color Palette
# ============================================================================

class Colors:
    """
    Professional surveillance color palette.
    
    Design Philosophy:
        - 95% grayscale for backgrounds and text
        - 5% color for accents and alerts
        - Orange as primary accent (industrial monitoring feel)
        - No blue (avoids enterprise SaaS look)
        - Risk: Gray → Amber → Orange → Red
        - Bounding boxes: Soft green (always)
    """
    
    # Backgrounds (Grayscale)
    BACKGROUND = (22, 22, 22)           # Very Dark Gray (#1A1A1A)
    PANEL = (35, 35, 35)                # Slightly lighter (#252525)
    PANEL_DARK = (28, 28, 28)           # Darker panel (#1C1C1C)
    PANEL_LIGHT = (45, 45, 45)          # Lighter panel (#2D2D2D)
    HEADER = (28, 28, 30)               # Header background (#1C1C1E)
    
    # Borders
    BORDER = (60, 60, 60)               # Subtle borders (#3C3C3C)
    BORDER_ACCENT = (0, 120, 200)       # Orange accent border (slightly darker)
    
    # Text
    TEXT_PRIMARY = (255, 255, 255)      # White
    TEXT_SECONDARY = (190, 190, 190)    # Light Gray (#BEBEBE)
    TEXT_MUTED = (130, 130, 130)        # Medium Gray (#828282)
    TEXT_DIM = (80, 80, 80)             # Dim Gray (#505050)
    
    # Accent Colors (Orange theme - softer, more premium)
    ACCENT = (0, 150, 235)              # Soft Orange (#FF9610)
    ACCENT_DIM = (0, 110, 185)          # Dim Orange
    ACCENT_DARK = (0, 70, 130)          # Dark Orange
    
    # Risk Colors (Gray → Amber → Orange → Red)
    RISK_NORMAL = (130, 130, 130)       # Gray - No attention needed
    RISK_ELEVATED = (0, 200, 255)       # Amber (#FFC800)
    RISK_HIGH = (0, 140, 255)           # Orange (#FF8C00)
    RISK_CRITICAL = (0, 0, 255)         # Red (#FF0000)
    
    # Bounding Boxes (Soft Green - easier on the eyes)
    BOX_GREEN = (60, 180, 75)           # Soft green (#4BB44B)
    BOX_GREEN_DIM = (30, 90, 38)        # Dim green for inactive
    
    # Status Indicators
    STATUS_ONLINE = (0, 200, 0)         # Green
    STATUS_OFFLINE = (0, 0, 255)        # Red
    STATUS_WARNING = (0, 200, 255)      # Amber
    
    # Alerts
    ALERT_INFO = (190, 190, 190)        # Gray
    ALERT_WARNING = (0, 200, 255)       # Amber
    ALERT_HIGH = (0, 140, 255)          # Orange
    ALERT_CRITICAL = (0, 0, 255)        # Red
    
    @classmethod
    def get_risk_color(cls, risk_level: RiskLevel) -> Tuple[int, int, int]:
        """Get color for risk level."""
        if risk_level == RiskLevel.NORMAL:
            return cls.RISK_NORMAL
        elif risk_level == RiskLevel.ELEVATED:
            return cls.RISK_ELEVATED
        elif risk_level == RiskLevel.HIGH:
            return cls.RISK_HIGH
        elif risk_level == RiskLevel.CRITICAL:
            return cls.RISK_CRITICAL
        return cls.RISK_NORMAL
    
    @classmethod
    def get_risk_text(cls, risk_level: RiskLevel) -> str:
        """Get risk level text."""
        return risk_level.value.upper()


# ============================================================================
# Dashboard - Pure Rendering Layer
# ============================================================================

class Dashboard:
    """
    Pure rendering layer for the Smart Crowd Management System.
    
    The dashboard has ZERO business logic. It only consumes state from
    other modules and renders it on the frame.
    
    Inputs:
        - frame: Camera frame
        - people: List of Person objects
        - counter_state: CounterState from Counter
        - occupancy_state: OccupancyState from Occupancy
        - risk_state: RiskState from Risk
        - alert_state: AlertState from Alerts
    
    Output:
        - Annotated frame with all visual elements
    
    Attributes:
        frame_width: Width of the frame
        frame_height: Height of the frame
        show_fps: Whether to show FPS
        show_frame_number: Whether to show frame number (debug)
        recording: Whether recording is active
        start_time: When the monitoring session started
    """
    
    def __init__(
        self,
        show_fps: bool = True,
        show_frame_number: bool = False,
        camera_name: str = "Camera 01"
    ) -> None:
        """
        Initialize the dashboard.
        
        Args:
            show_fps: Whether to show FPS counter
            show_frame_number: Whether to show frame number (debug)
            camera_name: Camera display name
        """
        self.show_fps = show_fps
        self.show_frame_number = show_frame_number
        self.camera_name = camera_name
        self.recording = False
        
        self.frame_width = 0
        self.frame_height = 0
        self.fps = 0
        self.frame_count = 0
        self.start_time = datetime.now()
    
    def render(
        self,
        frame: np.ndarray,
        people: List[Person],
        counter_state: CounterState,
        occupancy_state: OccupancyState,
        risk_state: RiskState,
        alert_state: AlertState,
        fps: float = 0.0
    ) -> np.ndarray:
        """
        Render all system state onto the frame.
        
        This is the MAIN ENTRY POINT of the dashboard.
        
        Args:
            frame: Camera frame (BGR)
            people: List of Person objects from Detector
            counter_state: CounterState from Counter
            occupancy_state: OccupancyState from Occupancy
            risk_state: RiskState from Risk
            alert_state: AlertState from Alerts
            fps: Current FPS
            
        Returns:
            Annotated frame
        """
        # Store frame dimensions
        self.frame_height, self.frame_width = frame.shape[:2]
        self.fps = fps
        self.frame_count += 1
        
        # Create a copy to avoid modifying original
        annotated = frame.copy()
        
        # Draw all elements (order matters for layering)
        self._draw_bounding_boxes(annotated, people)
        self._draw_header(annotated)
        self._draw_analytics_bar(annotated, counter_state, occupancy_state, risk_state)
        self._draw_alert_panel(annotated, alert_state)
        self._draw_watermark(annotated)
        self._draw_system_status(annotated)
        self._draw_frame_debug(annotated)
        
        return annotated
    
    def _draw_bounding_boxes(self, frame: np.ndarray, people: List[Person]) -> None:
        """
        Draw bounding boxes with track IDs.
        
        Boxes are ALWAYS soft green - they represent tracked individuals,
        not the crowd state. Risk is shown in the analytics bar.
        """
        for person in people:
            x1, y1, x2, y2 = person.get_bbox_int()
            
            # Soft green - easier on the eyes during long monitoring
            color = Colors.BOX_GREEN
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label background
            label = f"ID {person.track_id}"
            (label_w, label_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(
                frame,
                (x1, y1 - label_h - 8),
                (x1 + label_w + 8, y1),
                color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                frame,
                label,
                (x1 + 4, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1
            )
    
    def _draw_header(self, frame: np.ndarray) -> None:
        """
        Draw the header bar.
        
        Layout:
            [SCMS v1.0]                    [● LIVE] [Camera 01] [Time]
        
        Minimal vertical space. No large text.
        """
        height = 38
        padding = 15
        
        # Background
        cv2.rectangle(
            frame,
            (0, 0),
            (self.frame_width, height),
            Colors.HEADER,
            -1
        )
        
        # Accent line (Soft Orange - more premium)
        cv2.line(
            frame,
            (0, height - 2),
            (self.frame_width, height - 2),
            Colors.ACCENT,
            2
        )
        
        # System name (small, professional)
        cv2.putText(
            frame,
            "SCMS v1.0",
            (padding, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            Colors.TEXT_SECONDARY,
            1
        )
        
        # Right side elements
        x_pos = self.frame_width - padding
        
        # Recording indicator (if active)
        if self.recording:
            cv2.circle(frame, (x_pos - 60, 20), 5, Colors.STATUS_ONLINE, -1)
            cv2.putText(
                frame,
                "REC",
                (x_pos - 48, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                Colors.STATUS_ONLINE,
                1
            )
            x_pos -= 80
        
        # Live indicator
        cv2.circle(frame, (x_pos - 50, 20), 5, Colors.STATUS_ONLINE, -1)
        cv2.putText(
            frame,
            "LIVE",
            (x_pos - 38, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            Colors.TEXT_PRIMARY,
            1
        )
        
        # Camera name
        cv2.putText(
            frame,
            self.camera_name,
            (x_pos - 150, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            Colors.TEXT_SECONDARY,
            1
        )
        
        # Current time
        current_time = datetime.now().strftime("%H:%M:%S")
        cv2.putText(
            frame,
            current_time,
            (x_pos - 220, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            Colors.TEXT_MUTED,
            1
        )
    
    def _draw_analytics_bar(
        self,
        frame: np.ndarray,
        counter_state: CounterState,
        occupancy_state: OccupancyState,
        risk_state: RiskState
    ) -> None:
        """
        Draw the analytics bar below the video feed.
        
        Layout (Value first, label below):
            12        48%        12/25      NORMAL     30       00:12:35
            People    Occupancy  Capacity   Risk       FPS      Duration
        
        Values are LARGE. Labels are small and below.
        """
        bar_height = 70
        y_offset = self.frame_height - bar_height
        padding = 30
        
        # Background
        cv2.rectangle(
            frame,
            (0, y_offset),
            (self.frame_width, self.frame_height),
            Colors.BACKGROUND,
            -1
        )
        
        # Border line at top (Soft Orange accent)
        cv2.line(
            frame,
            (0, y_offset),
            (self.frame_width, y_offset),
            Colors.ACCENT,
            1
        )
        
        # Calculate session duration
        duration = datetime.now() - self.start_time
        duration_str = str(duration).split('.')[0]  # HH:MM:SS
        
        # Metrics configuration
        metrics = [
            ("PEOPLE", str(counter_state.current_count), Colors.TEXT_PRIMARY),
            ("OCCUPANCY", f"{occupancy_state.occupancy_percentage:.0f}%", Colors.TEXT_PRIMARY),
            ("CAPACITY", f"{counter_state.current_count}/{occupancy_state.capacity}", Colors.TEXT_PRIMARY),
            ("RISK", Colors.get_risk_text(risk_state.level), Colors.get_risk_color(risk_state.level)),
            ("FPS", f"{self.fps:.0f}" if self.show_fps else "--", Colors.TEXT_MUTED),
            ("DURATION", duration_str, Colors.TEXT_MUTED)
        ]
        
        x_pos = padding
        spacing = (self.frame_width - (padding * 2)) // len(metrics)
        
        for i, (label, value, color) in enumerate(metrics):
            x = x_pos + (i * spacing)
            
            # Vertical separator (subtle)
            if i > 0:
                cv2.line(
                    frame,
                    (x - 10, y_offset + 8),
                    (x - 10, y_offset + bar_height - 8),
                    Colors.BORDER,
                    1
                )
            
            # Value (LARGE, primary)
            font_scale = 1.0 if len(value) <= 4 else 0.8
            cv2.putText(
                frame,
                value,
                (x, y_offset + 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                color,
                2
            )
            
            # Label (small, below value)
            cv2.putText(
                frame,
                label,
                (x, y_offset + 58),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                Colors.TEXT_MUTED,
                1
            )
    
    def _draw_alert_panel(
        self,
        frame: np.ndarray,
        alert_state: AlertState
    ) -> None:
        """
        Draw the alert panel.
        
        Position: Above the analytics bar, right side
        Only shows when there are active alerts.
        """
        if not alert_state.active_alerts:
            return
        
        # Position: Top-right corner of the video feed
        x_offset = self.frame_width - 320
        y_offset = 55
        width = 300
        max_height = 160
        
        # Background
        cv2.rectangle(
            frame,
            (x_offset, y_offset),
            (x_offset + width, y_offset + max_height),
            Colors.PANEL,
            -1
        )
        
        # Border
        cv2.rectangle(
            frame,
            (x_offset, y_offset),
            (x_offset + width, y_offset + max_height),
            Colors.BORDER,
            1
        )
        
        # Soft Orange accent line at top
        cv2.line(
            frame,
            (x_offset, y_offset),
            (x_offset + width, y_offset),
            Colors.ACCENT,
            2
        )
        
        # Header
        cv2.putText(
            frame,
            "ALERTS",
            (x_offset + 12, y_offset + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            Colors.ACCENT,
            1
        )
        
        # Show alerts (top 3, newest first)
        y = y_offset + 45
        for alert in alert_state.active_alerts[:3]:
            # Color based on severity
            if alert.severity == "CRITICAL":
                color = Colors.ALERT_CRITICAL
            elif alert.severity == "HIGH":
                color = Colors.ALERT_HIGH
            elif alert.severity == "ELEVATED":
                color = Colors.ALERT_WARNING
            else:
                color = Colors.ALERT_INFO
            
            # Alert icon
            cv2.putText(
                frame,
                "⚠",
                (x_offset + 12, y + 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1
            )
            
            # Alert message (truncated)
            msg = alert.message[:35]
            cv2.putText(
                frame,
                msg,
                (x_offset + 32, y + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                Colors.TEXT_SECONDARY,
                1
            )
            y += 26
    
    def _draw_watermark(self, frame: np.ndarray) -> None:
        """
        Draw small watermark.
        
        Position: Bottom-left corner
        """
        padding = 10
        y_offset = self.frame_height - 12
        
        cv2.putText(
            frame,
            "SCMS v1.0",
            (padding, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            Colors.TEXT_DIM,
            1
        )
    
    def _draw_system_status(self, frame: np.ndarray) -> None:
        """
        Draw system status indicators.
        
        Position: Bottom-right corner
        Small, unobtrusive.
        """
        y_offset = self.frame_height - 12
        x_offset = self.frame_width - 240
        
        statuses = [
            ("Det", Colors.STATUS_ONLINE),
            ("Trak", Colors.STATUS_ONLINE),
            ("Cnt", Colors.STATUS_ONLINE),
            ("Cam", Colors.STATUS_ONLINE)
        ]
        
        x = x_offset
        for name, color in statuses:
            cv2.putText(
                frame,
                f"● {name}",
                (x, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                color,
                1
            )
            x += 65
    
    def _draw_frame_debug(self, frame: np.ndarray) -> None:
        """
        Draw frame number for debugging.
        
        Position: Bottom-center, very dim
        Only shown if show_frame_number is True.
        """
        if not self.show_frame_number:
            return
        
        y_offset = self.frame_height - 12
        text = f"Frame {self.frame_count}"
        
        cv2.putText(
            frame,
            text,
            (self.frame_width // 2 - 40, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            Colors.TEXT_DIM,
            1
        )
    
    def set_recording(self, active: bool) -> None:
        """
        Set recording status.
        
        Args:
            active: True if recording is active
        """
        self.recording = active
    
    def set_camera_name(self, name: str) -> None:
        """
        Set camera display name.
        
        Args:
            name: Camera name
        """
        self.camera_name = name
    
    def reset_timer(self) -> None:
        """
        Reset the session timer.
        
        Useful when starting a new monitoring session.
        """
        self.start_time = datetime.now()
    
    def draw_zone_info(
        self,
        frame: np.ndarray,
        zone_counts: dict
    ) -> None:
        """
        Optional: Draw zone information if available.
        
        Args:
            frame: Frame to draw on
            zone_counts: Dictionary mapping zone names to counts
        """
        if not zone_counts:
            return
        
        y_offset = 55
        x_offset = 10
        width = 120
        height = len(zone_counts) * 28 + 30
        
        cv2.rectangle(
            frame,
            (x_offset, y_offset),
            (x_offset + width, y_offset + height),
            Colors.PANEL,
            -1
        )
        cv2.rectangle(
            frame,
            (x_offset, y_offset),
            (x_offset + width, y_offset + height),
            Colors.BORDER,
            1
        )
        
        # Soft Orange accent line at top
        cv2.line(
            frame,
            (x_offset, y_offset),
            (x_offset + width, y_offset),
            Colors.ACCENT,
            2
        )
        
        y = y_offset + 25
        for zone_name, count in zone_counts.items():
            cv2.putText(
                frame,
                f"{zone_name}: {count}",
                (x_offset + 10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                Colors.TEXT_SECONDARY,
                1
            )
            y += 28