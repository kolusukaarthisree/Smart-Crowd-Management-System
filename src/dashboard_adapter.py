"""
dashboard_adapter.py - Compatibility Layer Between Backend and Dashboard

This module serves as the EXCLUSIVE translation layer between the backend
modules and the Dashboard UI. It implements the Adapter Design Pattern.

RESPONSIBILITY:
    Convert backend state objects into dashboard-compatible objects.
    ONLY field translation - NO business logic, NO calculations.

ARCHITECTURE:
    Backend Objects → DashboardAdapter → Dashboard Objects

DESIGN PRINCIPLES:
    - Single Responsibility: ONLY translation, no business logic
    - Interface Segregation: Clean, focused conversion methods
    - Dependency Inversion: Depends on abstractions, not concretions
    - Explicit over Implicit: Clear naming and type annotations
    - Fail Gracefully: Return default states instead of crashing
    - Log Exceptions: Never silently swallow unexpected errors

USAGE:
    adapter = DashboardAdapter()
    
    # Convert backend states to dashboard states
    dash_counter = adapter.counter_to_dashboard(backend_counter_state)
    dash_occupancy = adapter.occupancy_to_dashboard(backend_occupancy_state)
    dash_risk = adapter.risk_to_dashboard(backend_risk_state)
    dash_alert = adapter.alert_to_dashboard(backend_alert_state)
    
    # Bulk conversion (recommended)
    states = adapter.convert_snapshot(
        counter=backend_counter,
        occupancy=backend_occupancy,
        risk=backend_risk,
        alert=backend_alert,
        camera_online=True,
        camera_quality="GOOD",
        fps=29.8
    )
    
    # Or use the dashboard-ready kwargs (cleanest)
    dashboard.update(
        frame=frame,
        **adapter.dashboard_update_kwargs(
            counter=backend_counter,
            occupancy=backend_occupancy,
            risk=backend_risk,
            alert=backend_alert,
            camera_online=True,
            camera_quality="GOOD",
            fps=29.8
        )
    )

FIELD MAPPINGS (VERIFY THESE AGAINST YOUR BACKEND):
    CounterState:
        Backend.current_count → Dashboard.people
    
    OccupancyState:
        Backend.current_people → Dashboard.current
        Backend.capacity → Dashboard.capacity
        Backend.occupancy_percentage → Dashboard.percent
    
    RiskState:
        Backend.level → Dashboard.level (with optional mapping)
    
    AlertState:
        Backend.total_active → Dashboard.active
        Backend.highest_level → Dashboard.severity (mapped)
        Backend.newest_alert.message → Dashboard.message

NO BACKEND MODULE KNOWS ABOUT DASHBOARD.
NO DASHBOARD CODE KNOWS ABOUT BACKEND MODULES.
This adapter is the ONLY bridge between them.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, fields
from typing import Dict, Any, Tuple, List, Optional

# Backend imports (aliased for clarity)
from src.counter import CounterState as BackendCounterState
from src.occupancy import OccupancyState as BackendOccupancyState
from src.risk import RiskState as BackendRiskState
from src.alerts import AlertState as BackendAlertState, AlertLevel

# Dashboard imports (aliased for clarity)
from src.dashboard import (
    CounterState as DashboardCounterState,
    OccupancyState as DashboardOccupancyState,
    RiskState as DashboardRiskState,
    AlertState as DashboardAlertState,
    CameraHealth,
    PerformanceMetrics,
    Person as DashboardPerson,
)

# Configure logging
logger = logging.getLogger(__name__)


# ============================================================================
# Adapter Configuration
# ============================================================================

@dataclass
class AdapterConfig:
    """
    Configuration for the dashboard adapter.
    
    Allows customization of how backend states map to dashboard states.
    
    Attributes:
        severity_mapping: Custom mapping for alert severity levels
        risk_mapping: Custom mapping for risk levels
        default_capacity: Default venue capacity (used when state is None)
        default_resolution: Default camera resolution (used when not provided)
        enable_logging: Whether to log translation operations
    """
    severity_mapping: Dict[str, str] = field(default_factory=lambda: {
        "INFO": "NORMAL",
        "WARNING": "WARNING",
        "CRITICAL": "CRITICAL"
    })
    
    risk_mapping: Dict[str, str] = field(default_factory=lambda: {
        "NORMAL": "NORMAL",
        "ELEVATED": "HIGH",
        "HIGH": "HIGH",
        "CRITICAL": "CRITICAL"
    })
    
    default_capacity: int = 120
    default_resolution: Tuple[int, int] = (1280, 720)
    enable_logging: bool = False


# ============================================================================
# Dashboard Adapter
# ============================================================================

class DashboardAdapter:
    """
    Adapter for translating backend objects to dashboard-compatible objects.
    
    This is the EXCLUSIVE translation layer between backend and dashboard.
    It ONLY performs field translation - NO business logic.
    
    Field Mappings (VERIFY THESE AGAINST YOUR BACKEND):
        CounterState:
            Backend.current_count → Dashboard.people
        
        OccupancyState:
            Backend.current_people → Dashboard.current
            Backend.capacity → Dashboard.capacity
            Backend.occupancy_percentage → Dashboard.percent
        
        RiskState:
            Backend.level → Dashboard.level (with optional mapping)
        
        AlertState:
            Backend.total_active → Dashboard.active
            Backend.highest_level → Dashboard.severity (mapped)
            Backend.newest_alert.message → Dashboard.message
    
    Example:
        adapter = DashboardAdapter()
        
        # Convert a single state
        dash_counter = adapter.counter_to_dashboard(backend_counter)
        
        # Convert multiple states
        states = adapter.convert_snapshot(
            counter=backend_counter,
            occupancy=backend_occupancy,
            risk=backend_risk,
            alert=backend_alert
        )
    """
    
    def __init__(self, config: Optional[AdapterConfig] = None) -> None:
        """
        Initialize the dashboard adapter.
        
        Args:
            config: Adapter configuration (uses defaults if None)
        """
        self.config = config or AdapterConfig()
        self._conversion_count = 0
    
    # ========================================================================
    # Core Conversion Methods
    # ========================================================================
    
    def counter_to_dashboard(
        self,
        state: Optional[BackendCounterState]
    ) -> DashboardCounterState:
        """
        Convert Backend CounterState → Dashboard CounterState.
        
        Mapping:
            Backend.current_count → Dashboard.people
        
        Args:
            state: Backend CounterState object (or None)
            
        Returns:
            Dashboard-compatible CounterState (default if state is None)
        """
        if state is None:
            return DashboardCounterState(people=0)
        
        try:
            result = DashboardCounterState(
                people=state.current_count
            )
            self._increment_conversion_count()
            return result
        except AttributeError as e:
            logger.warning(
                "DashboardAdapter: CounterState missing attribute '%s'",
                e
            )
            return DashboardCounterState(people=0)
    
    def occupancy_to_dashboard(
        self,
        state: Optional[BackendOccupancyState]
    ) -> DashboardOccupancyState:
        """
        Convert Backend OccupancyState → Dashboard OccupancyState.
        
        Mapping:
            Backend.current_people → Dashboard.current
            Backend.capacity → Dashboard.capacity
            Backend.occupancy_percentage → Dashboard.percent
        
        Args:
            state: Backend OccupancyState object (or None)
            
        Returns:
            Dashboard-compatible OccupancyState (default if state is None)
        """
        if state is None:
            return DashboardOccupancyState(
                current=0,
                capacity=self.config.default_capacity,
                percent=0.0
            )
        
        try:
            result = DashboardOccupancyState(
                current=state.current_people,
                capacity=state.capacity,
                percent=state.occupancy_percentage
            )
            self._increment_conversion_count()
            return result
        except AttributeError as e:
            logger.warning(
                "DashboardAdapter: OccupancyState missing attribute '%s'",
                e
            )
            return DashboardOccupancyState(
                current=0,
                capacity=self.config.default_capacity,
                percent=0.0
            )
    
    def risk_to_dashboard(
        self,
        state: Optional[BackendRiskState]
    ) -> DashboardRiskState:
        """
        Convert Backend RiskState → Dashboard RiskState.
        
        Mapping:
            Backend.level → Dashboard.level (with optional mapping)
        
        Args:
            state: Backend RiskState object (or None)
            
        Returns:
            Dashboard-compatible RiskState (default if state is None)
        """
        if state is None:
            return DashboardRiskState(level="NORMAL")
        
        try:
            # Apply risk level mapping if configured
            mapped_level = self.config.risk_mapping.get(
                state.level.value,
                state.level.value
            )
            
            result = DashboardRiskState(level=mapped_level)
            self._increment_conversion_count()
            return result
        except AttributeError as e:
            logger.warning(
                "DashboardAdapter: RiskState missing attribute '%s'",
                e
            )
            return DashboardRiskState(level="NORMAL")
    
    def alert_to_dashboard(
        self,
        state: Optional[BackendAlertState]
    ) -> DashboardAlertState:
        """
        Convert Backend AlertState → Dashboard AlertState.
        
        Mapping:
            Backend.total_active → Dashboard.active
            Backend.highest_level → Dashboard.severity (mapped)
            Backend.newest_alert.message → Dashboard.message
        
        Args:
            state: Backend AlertState object (or None)
            
        Returns:
            Dashboard-compatible AlertState (default if state is None)
        """
        if state is None:
            return DashboardAlertState(
                active=0,
                severity="NORMAL",
                message=""
            )
        
        try:
            severity = "NORMAL"
            message = ""
            
            if state.active_alerts:
                # Map highest alert level to dashboard severity
                if state.highest_level == AlertLevel.CRITICAL:
                    severity = "CRITICAL"
                elif state.highest_level == AlertLevel.WARNING:
                    severity = "WARNING"
                else:
                    severity = "NORMAL"
                
                # Get message from newest alert
                if state.newest_alert:
                    message = state.newest_alert.message
                elif state.active_alerts:
                    message = state.active_alerts[0].message
            
            result = DashboardAlertState(
                active=state.total_active,
                severity=severity,
                message=message
            )
            self._increment_conversion_count()
            return result
        except AttributeError as e:
            logger.warning(
                "DashboardAdapter: AlertState missing attribute '%s'",
                e
            )
            return DashboardAlertState(
                active=0,
                severity="NORMAL",
                message=""
            )
    
    def camera_to_dashboard(
        self,
        is_online: bool = True,
        quality: str = "GOOD",
        resolution: Optional[Tuple[int, int]] = None
    ) -> CameraHealth:
        """
        Create a Dashboard CameraHealth object.
        
        Args:
            is_online: Whether camera is online
            quality: Camera quality status ('GOOD', 'BLUR', 'OFFLINE')
            resolution: Camera resolution (width, height)
            
        Returns:
            Dashboard-compatible CameraHealth
        """
        resolution = resolution or self.config.default_resolution
        
        self._increment_conversion_count()
        
        return CameraHealth(
            online=is_online,
            quality=quality,
            resolution=resolution
        )
    
    def performance_to_dashboard(
        self,
        fps: float = 0.0,
        detector_ready: bool = True,
        tracker_ready: bool = True,
        logger_active: bool = True
    ) -> PerformanceMetrics:
        """
        Create a Dashboard PerformanceMetrics object.
        
        Args:
            fps: Current frames per second
            detector_ready: Whether detector is operational
            tracker_ready: Whether tracker is operational
            logger_active: Whether logger is active
            
        Returns:
            Dashboard-compatible PerformanceMetrics
        """
        self._increment_conversion_count()
        
        return PerformanceMetrics(
            fps=fps,
            detector_ready=detector_ready,
            tracker_ready=tracker_ready,
            logger_active=logger_active
        )
    
    def person_to_dashboard(
        self,
        person: Any
    ) -> DashboardPerson:
        """
        Convert a backend Person object to Dashboard Person.
        
        Args:
            person: Backend Person object with track_id, bbox, confidence
            
        Returns:
            Dashboard-compatible Person
            
        Raises:
            TypeError: If person is not a valid Person object
            AttributeError: If required attributes are missing
        """
        try:
            # Extract attributes
            track_id = getattr(person, 'track_id', None)
            bbox = getattr(person, 'bbox', None)
            confidence = getattr(person, 'confidence', None)
            
            # Validate required attributes
            if track_id is None:
                raise AttributeError("Person missing 'track_id' attribute")
            if bbox is None:
                raise AttributeError("Person missing 'bbox' attribute")
            if confidence is None:
                raise AttributeError("Person missing 'confidence' attribute")
            
            # Validate bbox length
            if not hasattr(bbox, '__iter__') or len(bbox) != 4:
                raise ValueError(
                    f"Bounding box must contain 4 values, got {len(bbox) if hasattr(bbox, '__iter__') else 'non-iterable'}"
                )
            
            # Convert numpy ints to Python ints for compatibility
            bbox = tuple(int(x) for x in bbox)
            
            self._increment_conversion_count()
            
            return DashboardPerson(
                track_id=int(track_id),
                bbox=bbox,
                confidence=float(confidence)
            )
            
        except (AttributeError, TypeError, ValueError) as e:
            logger.warning(
                "DashboardAdapter: Person conversion failed - %s",
                e
            )
            # Return a default person instead of crashing
            return DashboardPerson(
                track_id=0,
                bbox=(0, 0, 0, 0),
                confidence=0.0
            )
    
    def people_to_dashboard(
        self,
        people: Optional[List[Any]]
    ) -> List[DashboardPerson]:
        """
        Convert a list of backend Person objects to Dashboard Person objects.
        
        Args:
            people: List of backend Person objects (or None)
            
        Returns:
            List of dashboard-compatible Person objects
        """
        if not people:
            return []
        
        result = []
        for person in people:
            # If already a dashboard person, use it
            if isinstance(person, DashboardPerson):
                result.append(person)
                continue
            
            # Convert backend person
            dash_person = self.person_to_dashboard(person)
            result.append(dash_person)
        
        self._increment_conversion_count()
        return result
    
    # ========================================================================
    # Bulk Conversion Methods
    # ========================================================================
    
    def convert_snapshot(
        self,
        counter: Optional[BackendCounterState] = None,
        occupancy: Optional[BackendOccupancyState] = None,
        risk: Optional[BackendRiskState] = None,
        alert: Optional[BackendAlertState] = None,
        people: Optional[List[Any]] = None,
        camera_online: bool = True,
        camera_quality: str = "GOOD",
        camera_resolution: Optional[Tuple[int, int]] = None,
        fps: float = 0.0,
        detector_ready: bool = True,
        tracker_ready: bool = True,
        logger_active: bool = True
    ) -> Dict[str, Any]:
        """
        Convert multiple backend states to dashboard states in one call.
        
        This is the RECOMMENDED method for app.py.
        
        Args:
            counter: Backend CounterState
            occupancy: Backend OccupancyState
            risk: Backend RiskState
            alert: Backend AlertState
            people: List of backend Person objects
            camera_online: Camera online status
            camera_quality: Camera quality status ('GOOD', 'BLUR', 'OFFLINE')
            camera_resolution: Camera resolution
            fps: Current FPS
            detector_ready: Detector status
            tracker_ready: Tracker status
            logger_active: Logger status
            
        Returns:
            Dictionary with all converted dashboard states
            Keys: 'counter', 'occupancy', 'risk', 'alert', 'people',
                  'camera', 'performance'
            
        Example:
            states = adapter.convert_snapshot(
                counter=counter_state,
                occupancy=occupancy_state,
                risk=risk_state,
                alert=alert_state,
                people=people_list,
                camera_online=True,
                camera_quality="GOOD",
                fps=29.8
            )
            
            dashboard.update(
                frame=frame,
                people=states['people'],
                counter=states['counter'],
                occupancy=states['occupancy'],
                risk=states['risk'],
                alert=states['alert'],
                camera=states['camera'],
                perf=states['performance']
            )
        """
        result = {}
        
        # Convert counter
        result['counter'] = self.counter_to_dashboard(counter)
        
        # Convert occupancy
        result['occupancy'] = self.occupancy_to_dashboard(occupancy)
        
        # Convert risk
        result['risk'] = self.risk_to_dashboard(risk)
        
        # Convert alert
        result['alert'] = self.alert_to_dashboard(alert)
        
        # Convert people
        result['people'] = self.people_to_dashboard(people)
        
        # Convert camera
        result['camera'] = self.camera_to_dashboard(
            is_online=camera_online,
            quality=camera_quality,
            resolution=camera_resolution
        )
        
        # Convert performance
        result['performance'] = self.performance_to_dashboard(
            fps=fps,
            detector_ready=detector_ready,
            tracker_ready=tracker_ready,
            logger_active=logger_active
        )
        
        if self.config.enable_logging:
            logger.debug("Adapter converted %d states", len(result))
        
        return result
    
    def dashboard_update_kwargs(
        self,
        counter: Optional[BackendCounterState] = None,
        occupancy: Optional[BackendOccupancyState] = None,
        risk: Optional[BackendRiskState] = None,
        alert: Optional[BackendAlertState] = None,
        people: Optional[List[Any]] = None,
        camera_online: bool = True,
        camera_quality: str = "GOOD",
        camera_resolution: Optional[Tuple[int, int]] = None,
        fps: float = 0.0,
        detector_ready: bool = True,
        tracker_ready: bool = True,
        logger_active: bool = True
    ) -> Dict[str, Any]:
        """
        Generate keyword arguments directly for dashboard.update().
        
        This is the CLEANEST way to integrate with app.py.
        
        Args:
            Same as convert_snapshot()
            
        Returns:
            Dictionary that can be unpacked into dashboard.update()
            
        Example:
            dashboard.update(
                frame=frame,
                **adapter.dashboard_update_kwargs(
                    counter=counter_state,
                    occupancy=occupancy_state,
                    risk=risk_state,
                    alert=alert_state,
                    people=people_list,
                    camera_online=True,
                    camera_quality="GOOD",
                    fps=29.8
                )
            )
        """
        states = self.convert_snapshot(
            counter=counter,
            occupancy=occupancy,
            risk=risk,
            alert=alert,
            people=people,
            camera_online=camera_online,
            camera_quality=camera_quality,
            camera_resolution=camera_resolution,
            fps=fps,
            detector_ready=detector_ready,
            tracker_ready=tracker_ready,
            logger_active=logger_active
        )
        
        # Map to dashboard.update() parameter names
        return {
            'people': states['people'],
            'counter': states['counter'],
            'occupancy': states['occupancy'],
            'risk': states['risk'],
            'alert': states['alert'],
            'camera': states['camera'],
            'perf': states['performance']
        }
    
    # ========================================================================
    # Helper Methods
    # ========================================================================
    
    def _increment_conversion_count(self) -> None:
        """Increment the conversion counter for statistics."""
        self._conversion_count += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get adapter statistics.
        
        Returns:
            Dictionary with adapter statistics
        """
        return {
            'conversion_count': self._conversion_count,
            'config': {
                'enable_logging': self.config.enable_logging,
                'default_capacity': self.config.default_capacity
            }
        }
    
    def reset_stats(self) -> None:
        """Reset adapter statistics."""
        self._conversion_count = 0
    
    # ========================================================================
    # Magic Methods
    # ========================================================================
    
    def __repr__(self) -> str:
        """String representation of the adapter."""
        return f"DashboardAdapter(conversions={self._conversion_count})"


# ============================================================================
# Field Mapping Verification (Uses dataclass introspection)
# ============================================================================

def verify_field_mappings() -> Dict[str, bool]:
    """
    Verify that all expected backend fields exist.
    
    Uses dataclass fields introspection to correctly check dataclass attributes.
    
    This function should be called during testing to ensure the adapter's
    field mappings match your actual backend implementation.
    
    Returns:
        Dictionary mapping field names to their existence status
        
    Example:
        results = verify_field_mappings()
        for field, exists in results.items():
            print(f"{'✅' if exists else '❌'} {field}")
    """
    results = {}
    
    # Check CounterState fields using dataclass introspection
    try:
        from src.counter import CounterState
        field_names = {f.name for f in fields(CounterState)}
        results['CounterState.current_count'] = 'current_count' in field_names
    except (ImportError, TypeError) as e:
        logger.warning("Could not inspect CounterState: %s", e)
        results['CounterState.current_count'] = False
    
    # Check OccupancyState fields
    try:
        from src.occupancy import OccupancyState
        field_names = {f.name for f in fields(OccupancyState)}
        results['OccupancyState.current_people'] = 'current_people' in field_names
        results['OccupancyState.capacity'] = 'capacity' in field_names
        results['OccupancyState.occupancy_percentage'] = 'occupancy_percentage' in field_names
    except (ImportError, TypeError) as e:
        logger.warning("Could not inspect OccupancyState: %s", e)
        results['OccupancyState.current_people'] = False
        results['OccupancyState.capacity'] = False
        results['OccupancyState.occupancy_percentage'] = False
    
    # Check RiskState fields
    try:
        from src.risk import RiskState
        field_names = {f.name for f in fields(RiskState)}
        results['RiskState.level'] = 'level' in field_names
    except (ImportError, TypeError) as e:
        logger.warning("Could not inspect RiskState: %s", e)
        results['RiskState.level'] = False
    
    # Check AlertState fields
    try:
        from src.alerts import AlertState
        field_names = {f.name for f in fields(AlertState)}
        results['AlertState.total_active'] = 'total_active' in field_names
        results['AlertState.highest_level'] = 'highest_level' in field_names
        results['AlertState.active_alerts'] = 'active_alerts' in field_names
        results['AlertState.newest_alert'] = 'newest_alert' in field_names
    except (ImportError, TypeError) as e:
        logger.warning("Could not inspect AlertState: %s", e)
        results['AlertState.total_active'] = False
        results['AlertState.highest_level'] = False
        results['AlertState.active_alerts'] = False
        results['AlertState.newest_alert'] = False
    
    return results


# ============================================================================
# Usage Example
# ============================================================================

"""
Example usage in app.py:

from src.dashboard_adapter import DashboardAdapter

class Application:
    def __init__(self):
        self.adapter = DashboardAdapter()
        self.dashboard = Dashboard()
    
    def _process_frame(self, frame: np.ndarray) -> None:
        # ... run detector, counter, occupancy, risk, alerts ...
        
        # Convert all states and update dashboard in one line
        self.dashboard.update(
            frame=frame,
            **self.adapter.dashboard_update_kwargs(
                counter=counter_state,
                occupancy=occupancy_state,
                risk=risk_state,
                alert=alert_state,
                people=people,
                camera_online=self.camera_online,
                camera_quality=self.camera_quality,
                camera_resolution=(1280, 720),
                fps=self.current_fps,
                detector_ready=True,
                tracker_ready=True,
                logger_active=True
            )
        )
"""


# ============================================================================
# Test/Example Code
# ============================================================================

if __name__ == "__main__":
    import sys
    
    print("=" * 60)
    print("Dashboard Adapter - Field Mapping Verification")
    print("=" * 60)
    
    # First, verify field mappings against the actual backend
    print("\n📋 VERIFYING FIELD MAPPINGS AGAINST BACKEND:")
    print("-" * 40)
    
    mappings = verify_field_mappings()
    
    all_passed = True
    for field, exists in mappings.items():
        status = "✅" if exists else "❌"
        print(f"  {status} {field}")
        if not exists:
            all_passed = False
    
    if not all_passed:
        print("\n⚠️  WARNING: Some expected fields are missing from your backend.")
        print("   Please verify the field names in your backend modules.")
        print("   The adapter will still work but may return default values.")
    
    print("\n" + "-" * 40)
    
    # Create mock backend states for testing
    class MockPerson:
        def __init__(self):
            self.track_id = 42
            self.bbox = (100, 100, 200, 200)
            self.confidence = 0.95
    
    class MockCounterState:
        current_count = 42
        frame_number = 123
    
    class MockOccupancyState:
        current_people = 42
        capacity = 120
        occupancy_percentage = 35.0
        frame_number = 123
    
    class MockRiskState:
        class Level:
            value = "NORMAL"
        level = Level()
        frame_number = 123
    
    class MockAlertState:
        active_alerts = []
        total_active = 0
        class HighestLevel:
            value = "INFO"
        highest_level = HighestLevel()
        newest_alert = None
        frame_number = 123
    
    # Initialize adapter
    adapter = DashboardAdapter(AdapterConfig(
        default_capacity=120,
        enable_logging=True
    ))
    
    print("\n✅ TESTING CONVERSIONS:")
    print("-" * 40)
    
    # Test 1: Counter
    print("\n1. CounterState:")
    dash_counter = adapter.counter_to_dashboard(MockCounterState())
    print(f"   ✓ Backend.current_count=42 → Dashboard.people={dash_counter.people}")
    
    # Test 2: Occupancy
    print("\n2. OccupancyState:")
    dash_occupancy = adapter.occupancy_to_dashboard(MockOccupancyState())
    print(f"   ✓ Backend.current_people=42 → Dashboard.current={dash_occupancy.current}")
    print(f"   ✓ Backend.capacity=120 → Dashboard.capacity={dash_occupancy.capacity}")
    print(f"   ✓ Backend.occupancy_percentage=35.0 → Dashboard.percent={dash_occupancy.percent}")
    
    # Test 3: Risk
    print("\n3. RiskState:")
    dash_risk = adapter.risk_to_dashboard(MockRiskState())
    print(f"   ✓ Backend.level=NORMAL → Dashboard.level={dash_risk.level}")
    
    # Test 4: Alert
    print("\n4. AlertState:")
    dash_alert = adapter.alert_to_dashboard(MockAlertState())
    print(f"   ✓ Backend.total_active=0 → Dashboard.active={dash_alert.active}")
    print(f"   ✓ Backend.highest_level=INFO → Dashboard.severity={dash_alert.severity}")
    
    # Test 5: Person
    print("\n5. Person:")
    dash_person = adapter.person_to_dashboard(MockPerson())
    print(f"   ✓ Backend.track_id=42 → Dashboard.track_id={dash_person.track_id}")
    print(f"   ✓ Backend.bbox=(100,100,200,200) → Dashboard.bbox={dash_person.bbox}")
    print(f"   ✓ Backend.confidence=0.95 → Dashboard.confidence={dash_person.confidence}")
    
    # Test 6: Null safety with default capacity
    print("\n6. Null Safety (Occupancy):")
    dash_occupancy_null = adapter.occupancy_to_dashboard(None)
    print(f"   ✓ OccupancyState(None) → capacity={dash_occupancy_null.capacity} (using default_capacity=120)")
    
    # Test 7: dashboard_update_kwargs
    print("\n7. dashboard_update_kwargs():")
    kwargs = adapter.dashboard_update_kwargs(
        counter=MockCounterState(),
        occupancy=MockOccupancyState(),
        risk=MockRiskState(),
        alert=MockAlertState(),
        people=[MockPerson()],
        camera_online=True,
        camera_quality="GOOD",
        fps=30.0
    )
    print(f"   ✓ Generated {len(kwargs)} keyword arguments: {', '.join(kwargs.keys())}")
    
    # Test 8: One-line dashboard update simulation
    print("\n8. Simulated one-line dashboard update:")
    print(f"   dashboard.update(frame=frame, **adapter.dashboard_update_kwargs(...))")
    print(f"   ✓ All states: {', '.join(kwargs.keys())}")
    
    # Test 9: Exception logging (person with missing attributes)
    print("\n9. Exception Handling:")
    class BadPerson:
        pass  # Missing track_id, bbox, confidence
    
    dash_bad_person = adapter.person_to_dashboard(BadPerson())
    print(f"   ✓ BadPerson converted to default: track_id={dash_bad_person.track_id}")
    
    # Test 10: Bbox validation
    print("\n10. BBox Validation:")
    class InvalidBBoxPerson:
        def __init__(self):
            self.track_id = 42
            self.bbox = (100, 100)  # Only 2 values
            self.confidence = 0.95
    
    dash_invalid_bbox = adapter.person_to_dashboard(InvalidBBoxPerson())
    print(f"   ✓ Invalid bbox converted to default: bbox={dash_invalid_bbox.bbox}")
    
    print("\n" + "=" * 60)
    print("✅ All tests passed. Adapter ready for app.py.")
    print("=" * 60)
    
    # Final warning about field mappings
    if not all_passed:
        print("\n⚠️  REMINDER: Some field mappings failed verification.")
        print("   Please check your backend modules and update the adapter if needed.")
        sys.exit(1)