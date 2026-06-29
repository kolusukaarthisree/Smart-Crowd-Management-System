"""
person.py - Person Entity Module

This module defines the Person class representing a tracked individual
in the people counting and occupancy monitoring system.

Author: System Architect
Version: 1.0.0
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, List
from datetime import datetime
import uuid
import math


@dataclass
class Person:
    """
    Represents a tracked person with detection and tracking information.
    
    Attributes:
        track_id (int): Unique identifier assigned by the tracker (ByteTrack)
        bbox (Tuple[float, float, float, float]): Bounding box coordinates (x1, y1, x2, y2)
        confidence (float): Detection confidence score (0.0 to 1.0)
        center (Tuple[float, float]): Center point of the bounding box
        width (float): Width of the bounding box
        height (float): Height of the bounding box
        area (float): Area of the bounding box (width * height)
        aspect_ratio (float): Width/height ratio
        timestamp (datetime): When the person was first detected
        last_seen (datetime): When the person was last detected
        entry_time (Optional[datetime]): When the person entered the monitored zone
        exit_time (Optional[datetime]): When the person exited the monitored zone
        trajectory (List[Tuple[float, float]]): History of center positions
        is_active (bool): Whether the person is currently being tracked
        has_entered (bool): Whether the person has entered the zone
        has_exited (bool): Whether the person has exited the zone
        direction (str): Movement direction ('in', 'out', 'stationary')
        velocity (float): Movement speed in pixels per second
        angle (float): Movement angle in radians
        status (str): Current status ('tracking', 'lost', 'entered', 'exited')
        metadata (dict): Additional custom metadata
    """
    
    # Core identification and detection
    track_id: int
    bbox: Tuple[float, float, float, float]
    confidence: float
    center: Tuple[float, float] = field(init=False)
    width: float = field(init=False)
    height: float = field(init=False)
    area: float = field(init=False)
    aspect_ratio: float = field(init=False)
    
    # Timestamps
    timestamp: datetime = field(default_factory=datetime.now)
    last_seen: datetime = field(default_factory=datetime.now)
    entry_time: Optional[datetime] = None
    exit_time: Optional[datetime] = None
    
    # Tracking history
    trajectory: List[Tuple[float, float]] = field(default_factory=list)
    max_trajectory_length: int = 50
    
    # State flags
    is_active: bool = True
    has_entered: bool = False
    has_exited: bool = False
    
    # Movement attributes
    direction: str = 'stationary'
    velocity: float = 0.0
    angle: float = 0.0
    status: str = 'tracking'
    
    # Additional data
    metadata: dict = field(default_factory=dict)
    _uuid: str = field(default_factory=lambda: str(uuid.uuid4()), init=False)
    
    def __post_init__(self):
        """
        Initialize derived attributes after dataclass initialization.
        Calculates center, width, height, area, and aspect ratio.
        """
        x1, y1, x2, y2 = self.bbox
        
        # Calculate center point
        self.center = ((x1 + x2) / 2, (y1 + y2) / 2)
        
        # Calculate dimensions
        self.width = x2 - x1
        self.height = y2 - y1
        self.area = self.width * self.height
        self.aspect_ratio = self.width / self.height if self.height > 0 else 0
        
        # Initialize trajectory with current center
        self.trajectory.append(self.center)
    
    def update(self, bbox: Tuple[float, float, float, float], confidence: float) -> None:
        """
        Update person's detection information.
        
        Args:
            bbox: New bounding box coordinates (x1, y1, x2, y2)
            confidence: New detection confidence score
        """
        # Store previous position for movement calculation
        prev_center = self.center
        
        # Update core attributes
        self.bbox = bbox
        self.confidence = confidence
        
        # Recalculate derived attributes
        x1, y1, x2, y2 = bbox
        self.center = ((x1 + x2) / 2, (y1 + y2) / 2)
        self.width = x2 - x1
        self.height = y2 - y1
        self.area = self.width * self.height
        self.aspect_ratio = self.width / self.height if self.height > 0 else 0
        
        # Update trajectory
        self.trajectory.append(self.center)
        if len(self.trajectory) > self.max_trajectory_length:
            self.trajectory.pop(0)
        
        # Calculate movement metrics
        self._calculate_movement(prev_center)
        
        # Update timestamps
        self.last_seen = datetime.now()
        self.is_active = True
    
    def _calculate_movement(self, prev_center: Tuple[float, float]) -> None:
        """
        Calculate movement metrics based on center displacement.
        
        Args:
            prev_center: Previous center position
        """
        dx = self.center[0] - prev_center[0]
        dy = self.center[1] - prev_center[1]
        
        # Calculate Euclidean distance
        distance = math.sqrt(dx**2 + dy**2)
        
        # Calculate velocity (pixels per frame)
        self.velocity = distance
        
        # Calculate angle in radians (0 = right, π/2 = down)
        if distance > 0:
            self.angle = math.atan2(dy, dx)
            # Determine direction
            if abs(dx) > abs(dy):
                self.direction = 'left' if dx < 0 else 'right'
            else:
                self.direction = 'up' if dy < 0 else 'down'
        else:
            self.direction = 'stationary'
    
    def mark_entered(self) -> None:
        """Mark the person as having entered the monitored zone."""
        if not self.has_entered:
            self.has_entered = True
            self.entry_time = datetime.now()
            self.status = 'entered'
    
    def mark_exited(self) -> None:
        """Mark the person as having exited the monitored zone."""
        if not self.has_exited:
            self.has_exited = True
            self.exit_time = datetime.now()
            self.status = 'exited'
            self.is_active = False
    
    def mark_lost(self) -> None:
        """Mark the person as lost by the tracker."""
        self.is_active = False
        self.status = 'lost'
    
    def get_dwell_time(self) -> Optional[float]:
        """
        Calculate the dwell time of the person in seconds.
        
        Returns:
            Dwell time in seconds, or None if the person hasn't entered yet
        """
        if self.has_entered:
            end_time = self.exit_time if self.has_exited else datetime.now()
            return (end_time - self.entry_time).total_seconds()
        return None
    
    def get_bbox_int(self) -> Tuple[int, int, int, int]:
        """
        Get bounding box as integer coordinates.
        
        Returns:
            Tuple of (x1, y1, x2, y2) as integers
        """
        return (int(self.bbox[0]), int(self.bbox[1]), 
                int(self.bbox[2]), int(self.bbox[3]))
    
    def get_bbox_xywh(self) -> Tuple[float, float, float, float]:
        """
        Get bounding box in (x, y, width, height) format.
        
        Returns:
            Tuple of (x, y, width, height)
        """
        x, y, x2, y2 = self.bbox
        return (x, y, x2 - x, y2 - y)
    
    def get_center_int(self) -> Tuple[int, int]:
        """
        Get center point as integer coordinates.
        
        Returns:
            Tuple of (cx, cy) as integers
        """
        return (int(self.center[0]), int(self.center[1]))
    
    def compute_iou(self, other_bbox: Tuple[float, float, float, float]) -> float:
        """
        Compute Intersection over Union (IoU) with another bounding box.
        
        Args:
            other_bbox: Other bounding box (x1, y1, x2, y2)
            
        Returns:
            IoU score between 0 and 1
        """
        x1 = max(self.bbox[0], other_bbox[0])
        y1 = max(self.bbox[1], other_bbox[1])
        x2 = min(self.bbox[2], other_bbox[2])
        y2 = min(self.bbox[3], other_bbox[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = self.width * self.height
        area2 = (other_bbox[2] - other_bbox[0]) * (other_bbox[3] - other_bbox[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def is_inside_zone(self, zone: Tuple[float, float, float, float]) -> bool:
        """
        Check if the person's center is inside a zone.
        
        Args:
            zone: Zone boundaries (x1, y1, x2, y2)
            
        Returns:
            True if center is inside the zone
        """
        zx1, zy1, zx2, zy2 = zone
        cx, cy = self.center
        return zx1 <= cx <= zx2 and zy1 <= cy <= zy2
    
    def get_velocity_magnitude(self) -> float:
        """
        Get the velocity magnitude (speed) in pixels per frame.
        
        Returns:
            Velocity magnitude
        """
        return self.velocity
    
    def get_movement_direction(self) -> str:
        """
        Get the movement direction as a string.
        
        Returns:
            Direction string: 'left', 'right', 'up', 'down', or 'stationary'
        """
        return self.direction
    
    def to_dict(self) -> dict:
        """
        Convert person object to dictionary for serialization.
        
        Returns:
            Dictionary representation of the person
        """
        return {
            'track_id': self.track_id,
            'uuid': self._uuid,
            'bbox': self.bbox,
            'bbox_xywh': self.get_bbox_xywh(),
            'center': self.center,
            'width': self.width,
            'height': self.height,
            'area': self.area,
            'aspect_ratio': self.aspect_ratio,
            'confidence': self.confidence,
            'timestamp': self.timestamp.isoformat(),
            'last_seen': self.last_seen.isoformat(),
            'entry_time': self.entry_time.isoformat() if self.entry_time else None,
            'exit_time': self.exit_time.isoformat() if self.exit_time else None,
            'is_active': self.is_active,
            'has_entered': self.has_entered,
            'has_exited': self.has_exited,
            'direction': self.direction,
            'velocity': self.velocity,
            'angle': self.angle,
            'status': self.status,
            'dwell_time': self.get_dwell_time(),
            'trajectory_length': len(self.trajectory),
            'metadata': self.metadata
        }
    
    def __repr__(self) -> str:
        """String representation of the Person object."""
        return (f"Person(id={self.track_id}, "
                f"center={self.center}, "
                f"status={self.status}, "
                f"conf={self.confidence:.2f}, "
                f"dwell={self.get_dwell_time():.1f}s if entered else None)")

    def __str__(self) -> str:
        """Human-readable string representation."""
        return self.__repr__()


# Factory function for creating Person instances
def create_person(track_id: int, 
                  bbox: Tuple[float, float, float, float], 
                  confidence: float,
                  metadata: Optional[dict] = None) -> Person:
    """
    Factory function to create a Person instance with validation.
    
    Args:
        track_id: Unique track ID
        bbox: Bounding box coordinates (x1, y1, x2, y2)
        confidence: Detection confidence score
        metadata: Optional additional metadata
        
    Returns:
        Person instance
        
    Raises:
        ValueError: If input validation fails
    """
    # Validate bbox
    if len(bbox) != 4:
        raise ValueError(f"bbox must have 4 elements, got {len(bbox)}")
    
    x1, y1, x2, y2 = bbox
    if x1 >= x2 or y1 >= y2:
        raise ValueError(f"Invalid bbox: {bbox}")
    
    # Validate confidence
    if not 0.0 <= confidence <= 1.0:
        raise ValueError(f"Confidence must be between 0 and 1, got {confidence}")
    
    # Create person
    person = Person(track_id=track_id, bbox=bbox, confidence=confidence)
    
    if metadata:
        person.metadata = metadata
    
    return person