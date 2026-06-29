"""
risk.py - Professional Risk Assessment Engine

This module performs risk assessment for the monitored area.
It evaluates multiple factors including occupancy, trends, and
future conditions to determine the overall risk level.

The risk engine does NOT simply check if occupancy > 80%.
It performs a comprehensive risk assessment considering:
    - Current occupancy
    - Occupancy trends
    - Capacity status
    - Rate of change
    - Multiple risk factors

Architecture:
    CounterState + OccupancyState → Risk Engine → RiskState
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

from .counter import CounterState
from .occupancy import OccupancyState, OccupancyStatus, OccupancyTrend

# Configure logging
logger = logging.getLogger(__name__)


# ============================================================================
# Enums
# ============================================================================

class RiskLevel(Enum):
    """
    Risk levels following industrial alarm philosophy.
    
    NORMAL     → Everything is fine. No attention needed.
    ELEVATED   → Monitor closely. Situation developing.
    HIGH       → Immediate attention recommended.
    CRITICAL   → Immediate action required!
    """
    NORMAL = "NORMAL"
    ELEVATED = "ELEVATED"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class RiskFactor(Enum):
    """
    Factors that contribute to the overall risk assessment.
    
    Each factor can independently trigger a risk level.
    The overall risk is the maximum of all factors.
    """
    OCCUPANCY = "OCCUPANCY"
    TREND = "TREND"
    CAPACITY = "CAPACITY"
    OVERFLOW = "OVERFLOW"
    RAPID_INCREASE = "RAPID_INCREASE"


# ============================================================================
# RiskState
# ============================================================================

@dataclass
class RiskState:
    """
    State of the risk engine at any point.
    
    This is the PRIMARY interface for other modules.
    All risk information is exposed through this object.
    
    Attributes:
        level: Overall risk level (NORMAL → CRITICAL)
        factors: Dictionary of risk factors and their levels
        primary_factor: The factor with the highest risk level
        occupancy_percentage: Current occupancy percentage
        status: Current occupancy status
        trend: Current occupancy trend
        is_over_capacity: Whether the area is over capacity
        message: Human-readable risk message
        color: Color associated with the risk level
        frame_number: Current frame number
        timestamp: When the state was computed
    """
    level: RiskLevel = RiskLevel.NORMAL
    factors: dict = field(default_factory=dict)
    primary_factor: Optional[RiskFactor] = None
    occupancy_percentage: float = 0.0
    status: OccupancyStatus = OccupancyStatus.NORMAL
    trend: OccupancyTrend = OccupancyTrend.STABLE
    is_over_capacity: bool = False
    message: str = "Normal operation"
    color: tuple = (0, 200, 0)  # Green
    frame_number: int = 0
    timestamp: datetime = field(default_factory=datetime.now)


# ============================================================================
# Risk Engine
# ============================================================================

class Risk:
    """
    Professional Risk Assessment Engine.
    
    Evaluates multiple risk factors to determine the overall risk level.
    Uses the highest-risk factor as the overall risk level.
    
    Risk Factors:
        - OCCUPANCY: Based on occupancy percentage (uses configured thresholds)
        - TREND: Based on occupancy trend
        - CAPACITY: Based on status (FULL, OVER_CAPACITY)
        - OVERFLOW: Based on over-capacity
        - RAPID_INCREASE: Based on rapid occupancy increase
    
    This engine follows the same architecture as Counter and Occupancy:
        - Single responsibility
        - O(1) time complexity
        - Clean, modular design
    
    Attributes:
        thresholds: Dictionary mapping risk levels to percentage thresholds
        rapid_increase_threshold: Percentage change for rapid increase detection
        state: Current risk state
        frame_number: Current frame number
        previous_percentage: Previous occupancy percentage (for change detection)
    """
    
    # Default thresholds (industrial standard)
    DEFAULT_THRESHOLDS = {
        RiskLevel.NORMAL: 0.60,        # 0-60%
        RiskLevel.ELEVATED: 0.80,      # 60-80%
        RiskLevel.HIGH: 0.90,          # 80-90%
        RiskLevel.CRITICAL: 0.95       # >90%
    }
    
    def __init__(
        self,
        thresholds: Optional[dict] = None,
        rapid_increase_threshold: float = 5.0
    ) -> None:
        """
        Initialize the Risk Engine.
        
        Args:
            thresholds: Custom risk thresholds (uses defaults if None)
            rapid_increase_threshold: Percentage change that triggers rapid increase
        """
        self.thresholds = thresholds or self.DEFAULT_THRESHOLDS
        self.rapid_increase_threshold = rapid_increase_threshold
        
        # State tracking
        self.previous_percentage: Optional[float] = None
        self.frame_number = 0
        self.state = RiskState()
        
        logger.info(f"Risk Engine initialized: thresholds={self.thresholds}")
    
    def update(
        self,
        counter_state: CounterState,
        occupancy_state: OccupancyState
    ) -> RiskState:
        """
        Update risk assessment with new counter and occupancy data.
        
        This is the MAIN ENTRY POINT of the risk engine.
        
        Algorithm:
            1. Evaluate all risk factors
            2. Determine the highest risk level
            3. Identify the primary factor
            4. Build and return state
            5. Store current percentage for next iteration
        
        Args:
            counter_state: CounterState from Counter module
            occupancy_state: OccupancyState from Occupancy module
            
        Returns:
            RiskState containing current risk assessment
        """
        self.frame_number = counter_state.frame_number
        
        # Step 1: Evaluate all risk factors
        factors = self._evaluate_factors(occupancy_state)
        
        # Step 2: Determine overall risk level
        level, primary_factor = self._determine_overall_risk(factors)
        
        # Step 3: Build state
        self.state = self._build_state(
            level=level,
            factors=factors,
            primary_factor=primary_factor,
            occupancy_state=occupancy_state
        )
        
        # Step 4: Store current percentage for next iteration (FIXED)
        self.previous_percentage = occupancy_state.occupancy_percentage
        
        logger.debug(f"Risk: {level.value} (primary: {primary_factor.value if primary_factor else 'None'})")
        
        return self.state
    
    def _evaluate_factors(self, occupancy_state: OccupancyState) -> dict:
        """
        Evaluate all risk factors.
        
        Args:
            occupancy_state: Current occupancy state
            
        Returns:
            Dictionary mapping RiskFactor to RiskLevel
        """
        factors = {}
        
        # Factor 1: Occupancy (uses configured thresholds)
        factors[RiskFactor.OCCUPANCY] = self._evaluate_occupancy(
            occupancy_state.occupancy_percentage
        )
        
        # Factor 2: Trend
        factors[RiskFactor.TREND] = self._evaluate_trend(
            occupancy_state.trend
        )
        
        # Factor 3: Capacity
        factors[RiskFactor.CAPACITY] = self._evaluate_capacity(
            occupancy_state.status
        )
        
        # Factor 4: Overflow
        factors[RiskFactor.OVERFLOW] = self._evaluate_overflow(
            occupancy_state.is_over_capacity
        )
        
        # Factor 5: Rapid Increase
        factors[RiskFactor.RAPID_INCREASE] = self._evaluate_rapid_increase(
            occupancy_state.occupancy_percentage
        )
        
        return factors
    
    def _evaluate_occupancy(self, percentage: float) -> RiskLevel:
        """
        Evaluate occupancy risk factor using configured thresholds.
        
        Args:
            percentage: Occupancy percentage
            
        Returns:
            RiskLevel based on occupancy
        """
        # Convert percentage to ratio (0.0 - 1.0+)
        ratio = percentage / 100.0
        
        # Check thresholds in order (highest risk first)
        if ratio >= self.thresholds[RiskLevel.CRITICAL]:
            return RiskLevel.CRITICAL
        elif ratio >= self.thresholds[RiskLevel.HIGH]:
            return RiskLevel.HIGH
        elif ratio >= self.thresholds[RiskLevel.ELEVATED]:
            return RiskLevel.ELEVATED
        else:
            return RiskLevel.NORMAL
    
    def _evaluate_trend(self, trend: OccupancyTrend) -> RiskLevel:
        """
        Evaluate trend risk factor.
        
        Args:
            trend: Occupancy trend
            
        Returns:
            RiskLevel based on trend
        """
        if trend == OccupancyTrend.INCREASING:
            return RiskLevel.ELEVATED
        else:
            return RiskLevel.NORMAL
    
    def _evaluate_capacity(self, status: OccupancyStatus) -> RiskLevel:
        """
        Evaluate capacity risk factor.
        
        Args:
            status: Occupancy status
            
        Returns:
            RiskLevel based on capacity status
        """
        if status == OccupancyStatus.OVER_CAPACITY:
            return RiskLevel.CRITICAL
        elif status == OccupancyStatus.FULL:
            return RiskLevel.HIGH
        elif status == OccupancyStatus.HIGH:
            return RiskLevel.ELEVATED
        else:
            return RiskLevel.NORMAL
    
    def _evaluate_overflow(self, is_over_capacity: bool) -> RiskLevel:
        """
        Evaluate overflow risk factor.
        
        Args:
            is_over_capacity: Whether the area is over capacity
            
        Returns:
            RiskLevel based on overflow status
        """
        if is_over_capacity:
            return RiskLevel.CRITICAL
        else:
            return RiskLevel.NORMAL
    
    def _evaluate_rapid_increase(self, current_percentage: float) -> RiskLevel:
        """
        Evaluate rapid increase risk factor.
        
        Uses previous_percentage stored from the last update.
        
        Args:
            current_percentage: Current occupancy percentage
            
        Returns:
            RiskLevel based on rapid increase detection
        """
        if self.previous_percentage is None:
            return RiskLevel.NORMAL
        
        increase = current_percentage - self.previous_percentage
        
        if increase > self.rapid_increase_threshold * 2:
            return RiskLevel.CRITICAL
        elif increase > self.rapid_increase_threshold:
            return RiskLevel.HIGH
        else:
            return RiskLevel.NORMAL
    
    def _determine_overall_risk(
        self,
        factors: dict
    ) -> tuple[RiskLevel, Optional[RiskFactor]]:
        """
        Determine the overall risk level from all factors.
        
        Uses the highest-risk factor as the overall level.
        
        Args:
            factors: Dictionary mapping RiskFactor to RiskLevel
            
        Returns:
            Tuple of (overall RiskLevel, primary RiskFactor)
        """
        # Map risk levels to numeric values
        level_values = {
            RiskLevel.NORMAL: 0,
            RiskLevel.ELEVATED: 1,
            RiskLevel.HIGH: 2,
            RiskLevel.CRITICAL: 3
        }
        
        # Find the highest risk level
        max_level = RiskLevel.NORMAL
        primary_factor = None
        max_value = -1
        
        for factor, level in factors.items():
            value = level_values[level]
            if value > max_value:
                max_value = value
                max_level = level
                primary_factor = factor
        
        return max_level, primary_factor
    
    def _build_state(
        self,
        level: RiskLevel,
        factors: dict,
        primary_factor: Optional[RiskFactor],
        occupancy_state: OccupancyState
    ) -> RiskState:
        """
        Build the current risk state.
        
        Args:
            level: Overall risk level
            factors: All risk factors
            primary_factor: Primary risk factor
            occupancy_state: Current occupancy state
            
        Returns:
            RiskState object
        """
        # Color mapping
        colors = {
            RiskLevel.NORMAL: (0, 200, 0),        # Green
            RiskLevel.ELEVATED: (0, 200, 255),    # Amber
            RiskLevel.HIGH: (0, 140, 255),        # Orange
            RiskLevel.CRITICAL: (0, 0, 255)       # Red
        }
        
        # Message mapping
        messages = {
            RiskLevel.NORMAL: "Normal operation. No action needed.",
            RiskLevel.ELEVATED: "Elevated risk. Monitor closely.",
            RiskLevel.HIGH: "High risk. Immediate attention recommended.",
            RiskLevel.CRITICAL: "CRITICAL! Immediate action required!"
        }
        
        return RiskState(
            level=level,
            factors={k.value: v.value for k, v in factors.items()},
            primary_factor=primary_factor,
            occupancy_percentage=occupancy_state.occupancy_percentage,
            status=occupancy_state.status,
            trend=occupancy_state.trend,
            is_over_capacity=occupancy_state.is_over_capacity,
            message=messages[level],
            color=colors[level],
            frame_number=self.frame_number,
            timestamp=datetime.now()
        )
    
    # ========================================================================
    # Public Methods
    # ========================================================================
    
    def get_risk_description(self, level: RiskLevel) -> str:
        """
        Get a human-readable description of a risk level.
        
        Args:
            level: RiskLevel
            
        Returns:
            Description string
        """
        descriptions = {
            RiskLevel.NORMAL: "Normal operation. No action needed.",
            RiskLevel.ELEVATED: "Elevated risk. Monitor closely.",
            RiskLevel.HIGH: "High risk. Immediate attention recommended.",
            RiskLevel.CRITICAL: "CRITICAL! Immediate action required!"
        }
        return descriptions.get(level, "Unknown risk level")
    
    def get_risk_color(self, level: RiskLevel) -> tuple:
        """
        Get the color associated with a risk level.
        
        Args:
            level: RiskLevel
            
        Returns:
            RGB color tuple
        """
        colors = {
            RiskLevel.NORMAL: (0, 200, 0),        # Green
            RiskLevel.ELEVATED: (0, 200, 255),    # Amber
            RiskLevel.HIGH: (0, 140, 255),        # Orange
            RiskLevel.CRITICAL: (0, 0, 255)       # Red
        }
        return colors.get(level, (255, 255, 255))
    
    def set_threshold(self, level: RiskLevel, threshold: float) -> None:
        """
        Update a specific risk threshold.
        
        Args:
            level: RiskLevel to update
            threshold: New threshold value (0.0 to 1.0)
            
        Raises:
            ValueError: If threshold is invalid
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Threshold must be between 0 and 1, got {threshold}")
        
        self.thresholds[level] = threshold
        logger.info(f"Threshold for {level.value} updated to: {threshold}")
    
    def reset(self) -> None:
        """
        Reset the risk engine state.
        
        Useful when starting a new monitoring session.
        """
        self.previous_percentage = None
        self.frame_number = 0
        self.state = RiskState()
        logger.info("Risk Engine reset")
    
    def __repr__(self) -> str:
        """String representation of the risk engine."""
        return f"Risk(level={self.state.level.value}, primary={self.state.primary_factor})"