"""
occupancy.py - Professional Occupancy Engine Module

This module performs occupancy state estimation for the monitored area.
It answers the question:

    "Given the current number of people and the maximum safe capacity,
     what is the operational state of this monitored area?"

Occupancy is NOT a people counter. The Counter already answers "how many."
Occupancy answers "what is the state of space utilization?"

Architecture:
    CounterState → Occupancy Engine → OccupancyState
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

from .counter import CounterState

# Configure logging
logger = logging.getLogger(__name__)


# ============================================================================
# Enums
# ============================================================================

class OccupancyStatus(Enum):
    """
    Capacity classification states.
    
    Follows industrial alarm system philosophy:
        - NORMAL: No attention needed
        - ELEVATED: Monitor closely
        - HIGH: Action may be required
        - FULL: At capacity
        - OVER_CAPACITY: Exceeds capacity
    """
    NORMAL = "NORMAL"
    ELEVATED = "ELEVATED"
    HIGH = "HIGH"
    FULL = "FULL"
    OVER_CAPACITY = "OVER CAPACITY"


class OccupancyTrend(Enum):
    """
    Occupancy trend states.
    
    Indicates how occupancy is changing over time:
        - INCREASING: Occupancy is going up
        - STABLE: Occupancy is relatively constant
        - DECREASING: Occupancy is going down
    """
    INCREASING = "INCREASING"
    STABLE = "STABLE"
    DECREASING = "DECREASING"


# ============================================================================
# OccupancyState
# ============================================================================

@dataclass
class OccupancyState:
    """
    State of the occupancy engine at any point.
    
    This is the PRIMARY interface for other modules.
    All occupancy information is exposed through this object.
    
    No downstream module should need to recompute anything.
    This state contains everything needed for Risk, Dashboard, and Logger.
    
    Attributes:
        venue_name: Name of the monitored venue (for multi-venue support)
        current_people: Current number of people
        capacity: Maximum capacity of the area
        occupancy_percentage: Current occupancy percentage (0-100+)
        remaining_capacity: Number of additional people allowed (clamped to 0)
        utilization_ratio: occupancy_percentage / 100 (0.0 - 1.0+)
        status: Capacity classification (NORMAL → OVER_CAPACITY)
        trend: Occupancy trend (INCREASING / STABLE / DECREASING)
        is_over_capacity: True if occupancy > 100%
        frame_number: Current frame number
        timestamp: When the state was computed
    """
    venue_name: str = "Default Venue"
    current_people: int = 0
    capacity: int = 0
    occupancy_percentage: float = 0.0
    remaining_capacity: int = 0
    utilization_ratio: float = 0.0
    status: OccupancyStatus = OccupancyStatus.NORMAL
    trend: OccupancyTrend = OccupancyTrend.STABLE
    is_over_capacity: bool = False
    frame_number: int = 0
    timestamp: datetime = field(default_factory=datetime.now)


# ============================================================================
# Occupancy Engine
# ============================================================================

class Occupancy:
    """
    Professional Occupancy Engine.
    
    Performs state estimation for the monitored area using:
        - Input validation
        - Occupancy estimation
        - Capacity analysis
        - Remaining capacity calculation
        - Occupancy classification
        - Trend analysis
        - State construction
    
    This engine follows the same architecture as Counter and Risk:
        - Single responsibility
        - O(1) time complexity
        - Clean, modular design
        - Configurable thresholds
    
    Attributes:
        venue_name: Name of the monitored venue
        capacity: Maximum capacity of the monitored area
        thresholds: Dictionary mapping status to percentage thresholds
        trend_threshold: Percentage change required to detect a trend
        previous_percentage: Previous occupancy percentage (for trend analysis)
        state: Current occupancy state
        frame_number: Current frame number
    """
    
    # Default thresholds (industrial standard)
    DEFAULT_THRESHOLDS = {
        OccupancyStatus.NORMAL: 0.60,       # 0-60%
        OccupancyStatus.ELEVATED: 0.80,     # 60-80%
        OccupancyStatus.HIGH: 0.95,         # 80-95%
        OccupancyStatus.FULL: 1.00,         # 95-100%
        # OVER_CAPACITY is > 1.00 (handled separately)
    }
    
    def __init__(
        self,
        capacity: int = 50,
        venue_name: str = "Default Venue",
        thresholds: Optional[dict] = None,
        trend_threshold: float = 2.0
    ) -> None:
        """
        Initialize the Occupancy Engine.
        
        Args:
            capacity: Maximum capacity of the monitored area
            venue_name: Name of the monitored venue
            thresholds: Custom status thresholds (uses defaults if None)
            trend_threshold: Percentage change required to detect a trend
            
        Raises:
            ValueError: If capacity is <= 0
        """
        # Validate capacity
        if capacity <= 0:
            raise ValueError(f"Capacity must be > 0, got {capacity}")
        
        self.venue_name = venue_name
        self.capacity = capacity
        self.thresholds = thresholds or self.DEFAULT_THRESHOLDS
        self.trend_threshold = trend_threshold
        
        # State tracking
        self.previous_percentage: Optional[float] = None
        self.frame_number = 0
        self.state = OccupancyState()
        
        logger.info(
            f"Occupancy Engine initialized: venue='{venue_name}', "
            f"capacity={capacity}, trend_threshold={trend_threshold}%"
        )
    
    # ========================================================================
    # Public Interface
    # ========================================================================
    
    def update(self, counter_state: CounterState) -> OccupancyState:
        """
        Update occupancy state with new counter data.
        
        This is the MAIN ENTRY POINT of the occupancy engine.
        
        Workflow:
            1. Validate input
            2. Calculate occupancy ratio
            3. Calculate remaining capacity
            4. Classify status
            5. Analyze trend
            6. Build state
        
        Args:
            counter_state: CounterState from Counter module
            
        Returns:
            OccupancyState containing current occupancy state
            
        Raises:
            ValueError: If counter_state is invalid
        """
        self.frame_number = counter_state.frame_number
        
        # Stage 1: Input Validation
        self._validate_input(counter_state)
        
        # Stage 2: Occupancy Estimation
        ratio, percentage = self._calculate_occupancy(counter_state.current_count)
        
        # Stage 3 & 4: Capacity Analysis & Remaining Capacity
        remaining = self._calculate_remaining_capacity(counter_state.current_count)
        is_over_capacity = counter_state.current_count > self.capacity
        
        # Stage 5: Occupancy Classification
        status = self._classify_status(percentage, is_over_capacity)
        
        # Stage 6: Trend Analysis
        trend = self._analyze_trend(percentage)
        
        # Stage 7: State Construction
        self.state = self._build_state(
            current_people=counter_state.current_count,
            ratio=ratio,
            percentage=percentage,
            remaining=remaining,
            is_over_capacity=is_over_capacity,
            status=status,
            trend=trend
        )
        
        # Store for next iteration
        self.previous_percentage = percentage
        
        logger.debug(
            f"Occupancy: {percentage:.1f}% ({status.value}), "
            f"trend: {trend.value}, remaining: {remaining}"
        )
        
        return self.state
    
    # ========================================================================
    # Private Methods (Stage 1-7)
    # ========================================================================
    
    def _validate_input(self, counter_state: CounterState) -> None:
        """
        Stage 1: Validate input.
        
        Never assume inputs are valid.
        Professional software never silently accepts impossible values.
        
        Args:
            counter_state: CounterState to validate
            
        Raises:
            ValueError: If input is invalid
        """
        if counter_state.current_count < 0:
            raise ValueError(
                f"Current people cannot be negative: {counter_state.current_count}"
            )
        
        if self.capacity <= 0:
            raise ValueError(
                f"Capacity must be > 0, got {self.capacity}"
            )
    
    def _calculate_occupancy(self, current_people: int) -> tuple[float, float]:
        """
        Stage 2: Estimate occupancy.
        
        Calculates both ratio and percentage to avoid repeated computation.
        
        Formula:
            ratio = current_people / capacity
            percentage = ratio * 100
        
        Args:
            current_people: Current number of people
            
        Returns:
            Tuple of (ratio, percentage)
        """
        ratio = current_people / self.capacity
        percentage = ratio * 100
        return ratio, percentage
    
    def _calculate_remaining_capacity(self, current_people: int) -> int:
        """
        Stages 3 & 4: Capacity analysis and remaining capacity.
        
        Calculates remaining capacity and clamps to 0.
        
        Args:
            current_people: Current number of people
            
        Returns:
            Remaining capacity (>= 0)
        """
        remaining = self.capacity - current_people
        return max(remaining, 0)
    
    def _classify_status(
        self,
        percentage: float,
        is_over_capacity: bool
    ) -> OccupancyStatus:
        """
        Stage 5: Classify occupancy status.
        
        Uses configured thresholds. OVER_CAPACITY is handled separately.
        
        Args:
            percentage: Occupancy percentage
            is_over_capacity: Whether the area is over capacity
            
        Returns:
            OccupancyStatus classification
        """
        # OVER_CAPACITY takes priority
        if is_over_capacity:
            return OccupancyStatus.OVER_CAPACITY
        
        # Check thresholds in order (highest first)
        if percentage >= self.thresholds[OccupancyStatus.FULL] * 100:
            return OccupancyStatus.FULL
        elif percentage >= self.thresholds[OccupancyStatus.HIGH] * 100:
            return OccupancyStatus.HIGH
        elif percentage >= self.thresholds[OccupancyStatus.ELEVATED] * 100:
            return OccupancyStatus.ELEVATED
        else:
            return OccupancyStatus.NORMAL
    
    def _analyze_trend(self, current_percentage: float) -> OccupancyTrend:
        """
        Stage 6: Analyze occupancy trend.
        
        Compares current percentage with previous value.
        
        Args:
            current_percentage: Current occupancy percentage
            
        Returns:
            OccupancyTrend: INCREASING, STABLE, or DECREASING
        """
        if self.previous_percentage is None:
            return OccupancyTrend.STABLE
        
        difference = current_percentage - self.previous_percentage
        
        if difference > self.trend_threshold:
            return OccupancyTrend.INCREASING
        elif difference < -self.trend_threshold:
            return OccupancyTrend.DECREASING
        else:
            return OccupancyTrend.STABLE
    
    def _build_state(
        self,
        current_people: int,
        ratio: float,
        percentage: float,
        remaining: int,
        is_over_capacity: bool,
        status: OccupancyStatus,
        trend: OccupancyTrend
    ) -> OccupancyState:
        """
        Stage 7: Construct the final state.
        
        No calculations should occur after this point.
        
        Args:
            current_people: Current number of people
            ratio: Utilization ratio
            percentage: Occupancy percentage
            remaining: Remaining capacity
            is_over_capacity: Whether over capacity
            status: Occupancy classification
            trend: Occupancy trend
            
        Returns:
            OccupancyState object
        """
        return OccupancyState(
            venue_name=self.venue_name,
            current_people=current_people,
            capacity=self.capacity,
            occupancy_percentage=percentage,
            remaining_capacity=remaining,
            utilization_ratio=ratio,
            status=status,
            trend=trend,
            is_over_capacity=is_over_capacity,
            frame_number=self.frame_number,
            timestamp=datetime.now()
        )
    
    # ========================================================================
    # Public Methods
    # ========================================================================
    
    def set_capacity(self, new_capacity: int) -> None:
        """
        Update the maximum capacity dynamically.
        
        Args:
            new_capacity: New capacity value
            
        Raises:
            ValueError: If new_capacity is <= 0
        """
        if new_capacity <= 0:
            raise ValueError(f"Capacity must be > 0, got {new_capacity}")
        
        self.capacity = new_capacity
        logger.info(f"Capacity updated to: {new_capacity}")
    
    def set_threshold(self, status: OccupancyStatus, threshold: float) -> None:
        """
        Update a specific classification threshold.
        
        Args:
            status: OccupancyStatus to update
            threshold: New threshold value (0.0 to 1.0)
            
        Raises:
            ValueError: If threshold is invalid
        """
        if status == OccupancyStatus.OVER_CAPACITY:
            raise ValueError("OVER_CAPACITY threshold is handled separately")
        
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Threshold must be between 0 and 1, got {threshold}")
        
        self.thresholds[status] = threshold
        logger.info(f"Threshold for {status.value} updated to: {threshold}")
    
    def get_status_description(self, status: OccupancyStatus) -> str:
        """
        Get a human-readable description of a status.
        
        Args:
            status: OccupancyStatus
            
        Returns:
            Description string
        """
        descriptions = {
            OccupancyStatus.NORMAL: "Normal occupancy. No action needed.",
            OccupancyStatus.ELEVATED: "Elevated occupancy. Monitor closely.",
            OccupancyStatus.HIGH: "High occupancy. Action may be required.",
            OccupancyStatus.FULL: "At capacity. Entry may need to be restricted.",
            OccupancyStatus.OVER_CAPACITY: "Over capacity. Immediate action required!"
        }
        return descriptions.get(status, "Unknown status")
    
    def get_status_color(self, status: OccupancyStatus) -> tuple:
        """
        Get the color associated with a status.
        
        Args:
            status: OccupancyStatus
            
        Returns:
            RGB color tuple
        """
        colors = {
            OccupancyStatus.NORMAL: (0, 200, 0),        # Green
            OccupancyStatus.ELEVATED: (0, 200, 255),    # Amber
            OccupancyStatus.HIGH: (0, 140, 255),        # Orange
            OccupancyStatus.FULL: (0, 100, 255),        # Dark Orange
            OccupancyStatus.OVER_CAPACITY: (0, 0, 255)  # Red
        }
        return colors.get(status, (255, 255, 255))
    
    def reset(self) -> None:
        """
        Reset the occupancy engine state.
        
        Useful when starting a new monitoring session.
        """
        self.previous_percentage = None
        self.frame_number = 0
        self.state = OccupancyState()
        logger.info("Occupancy Engine reset")
    
    def __repr__(self) -> str:
        """String representation of the occupancy engine."""
        return (f"Occupancy(venue='{self.venue_name}', "
                f"capacity={self.capacity}, "
                f"current={self.state.current_people}, "
                f"percentage={self.state.occupancy_percentage:.1f}%)")