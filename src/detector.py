"""
detector.py - Detector Wrapper Module

This module serves as the interface between Ultralytics (YOLO + ByteTrack)
and our Smart Crowd Management System. It has exactly ONE responsibility:

    Convert a camera frame into a List<Person>

The detector knows nothing about counting, occupancy, risk, alerts,
or dashboards. It only translates between Ultralytics objects and
our domain objects (Person).

Architecture:
    Camera Frame → YOLO → ByteTrack → Results → Person Objects

Author: System Architect
Version: 2.3.0 (FINAL - Locked)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
from ultralytics import YOLO

# Local imports - Using proper relative import
from .models.person import Person

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class DetectorConfig:
    """
    Configuration for the detector wrapper.
    
    These are the ONLY settings the detector knows about.
    Everything else is handled by Ultralytics internally.
    
    Attributes:
        model_path: Path to YOLO model file (.pt, .engine, etc.)
        confidence_threshold: Minimum confidence for detections (0.0-1.0)
        iou_threshold: IoU threshold for NMS (0.0-1.0)
        device: Computing device ('cpu', 'cuda', '0', '1', etc.)
        image_size: Input image size for YOLO
        max_detections: Maximum detections per frame
        tracker_config: Path to tracker configuration file
        person_class_id: COCO class ID for person (default: 0)
        half_precision: Use FP16 precision if available
        verbose: Print YOLO debug information
    """
    model_path: str = "yolov8n.pt"
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.45
    device: str = "cpu"
    image_size: int = 640
    max_detections: int = 100
    tracker_config: str = "bytetrack.yaml"
    person_class_id: int = 0
    half_precision: bool = False
    verbose: bool = False
    
    def __post_init__(self) -> None:
        """Validate configuration values."""
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError(f"confidence_threshold must be [0,1], got {self.confidence_threshold}")
        if not 0.0 <= self.iou_threshold <= 1.0:
            raise ValueError(f"iou_threshold must be [0,1], got {self.iou_threshold}")
        if self.person_class_id < 0:
            raise ValueError(f"person_class_id must be >= 0, got {self.person_class_id}")


class Detector:
    """
    Detector Wrapper - Interface between Ultralytics and our system.
    
    This class wraps YOLO + ByteTrack and exposes a clean interface:
        Frame → List<Person>
    
    It does NOT:
        - Count people (Counter does that)
        - Calculate occupancy (Occupancy does that)
        - Assess risk (Risk does that)
        - Generate alerts (Alerts does that)
        - Write logs (Logger does that)
        - Render dashboards (Dashboard does that)
        - Open camera streams (app.py does that)
    
    The detector is a pure translator. It takes a frame from ANY source
    (webcam, CCTV, IP camera, video file) and returns structured data.
    
    Attributes:
        config (DetectorConfig): Detector configuration
        model (YOLO): Ultralytics YOLO model instance
        frame_count (int): Internal counter for tracking
    """
    
    def __init__(self, config: Optional[DetectorConfig] = None) -> None:
        """
        Initialize the detector wrapper.
        
        This loads the YOLO model ONCE at startup and keeps it ready.
        The model is never reloaded during the application lifetime.
        
        Args:
            config: Detector configuration (uses defaults if None)
            
        Raises:
            RuntimeError: If model loading fails
        """
        self.config = config or DetectorConfig()
        self.frame_count = 0
        
        # Load the YOLO model (ONCE)
        # Ultralytics will auto-download if model doesn't exist locally
        self._load_model()
        
        logger.info("Detector initialized successfully")
    
    @property
    def model_name(self) -> str:
        """Get the model path for debugging purposes."""
        return self.config.model_path
    
    @property
    def class_names(self) -> dict:
        """Get the class names from the YOLO model."""
        return self.model.names if hasattr(self.model, 'names') else {}
    
    def _load_model(self) -> None:
        """
        Load the YOLO model.
        
        This is called ONCE during initialization.
        The model is kept in memory for the entire application lifetime.
        Ultralytics will automatically download pretrained models if needed.
        
        Raises:
            RuntimeError: If model loading fails
        """
        try:
            logger.info(f"Loading YOLO model: {self.config.model_path}")
            self.model = YOLO(self.config.model_path)
            logger.info("YOLO model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise RuntimeError(f"Model loading failed: {e}") from e
    
    def detect(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Person]]:
        """
        Process a frame and return annotated frame + Person objects.
        
        This is the MAIN ENTRY POINT of the detector.
        
        Workflow:
            1. Validate input frame
            2. Run YOLO + ByteTrack (Ultralytics)
            3. Parse Results object
            4. Filter for person class only
            5. Build Person objects
            6. Sort by track ID (deterministic ordering)
            7. Return (frame, people)
        
        Args:
            frame: Input image in BGR format (numpy.ndarray)
            
        Returns:
            Tuple containing:
                - Original frame (numpy.ndarray)
                - List of Person objects (sorted by track ID)
            
        Raises:
            ValueError: If frame is invalid
            RuntimeError: If detection fails
        """
        # Step 1: Validate input
        if frame is None or frame.size == 0:
            raise ValueError("Invalid frame: frame is None or empty")
        
        if len(frame.shape) != 3 or frame.shape[2] != 3:
            raise ValueError(f"Invalid frame shape: {frame.shape}, expected (H, W, 3)")
        
        self.frame_count += 1
        
        try:
            # Step 2: Run Ultralytics YOLO + ByteTrack
            # This is the ONLY place we call the official API
            results = self.model.track(
                source=frame,
                conf=self.config.confidence_threshold,
                iou=self.config.iou_threshold,
                classes=[self.config.person_class_id],  # Only track persons
                max_det=self.config.max_detections,
                device=self.config.device,
                imgsz=self.config.image_size,
                half=self.config.half_precision,
                verbose=self.config.verbose,
                persist=True,  # Persist tracks across frames
                tracker=self.config.tracker_config  # ByteTrack
            )
            
            # Step 3: Parse Ultralytics Results → Person objects
            people = self._parse_results(results)
            
            # Step 4: Sort for deterministic ordering
            people.sort(key=lambda p: p.track_id)
            
            # Step 5: Return original frame + people
            # Dashboard will handle annotation
            return frame, people
            
        except Exception as e:
            logger.error(f"Detection failed for frame {self.frame_count}: {e}")
            # Return original frame with no people on failure
            return frame, []
    
    def _parse_results(self, results) -> List[Person]:
        """
        Parse Ultralytics Results object into Person objects.
        
        This is the TRANSLATION LAYER.
        It converts Ultralytics-specific objects into our domain objects.
        
        Args:
            results: Ultralytics Results object
            
        Returns:
            List of Person objects
        """
        people = []
        
        # Check if we have valid results
        if not results or len(results) == 0:
            return people
        
        # model.track() returns one Results object per input frame
        result = results[0]
        
        # Check if there are boxes
        if result.boxes is None or result.boxes.id is None:
            return people
        
        # Extract data from boxes
        boxes = result.boxes
        track_ids = boxes.id.cpu().numpy() if boxes.id is not None else []
        
        # If no track IDs, return empty list
        if len(track_ids) == 0:
            return people
        
        # Get bounding boxes, confidence, and class IDs
        xyxy = boxes.xyxy.cpu().numpy()
        confidences = boxes.conf.cpu().numpy()
        class_ids = boxes.cls.cpu().numpy()
        
        # Iterate through all detections
        for i in range(len(track_ids)):
            # Extract information
            track_id = int(track_ids[i])
            bbox = xyxy[i]  # [x1, y1, x2, y2]
            confidence = float(confidences[i])
            class_id = int(class_ids[i])
            
            # Defensive programming: Double-check it's a person
            # Even though we filtered in track(), we verify again
            if class_id != self.config.person_class_id:
                logger.warning(f"Skipping non-person detection (class {class_id})")
                continue
            
            # Validate confidence
            if confidence < self.config.confidence_threshold:
                continue
            
            # Create Person object directly
            try:
                person = Person(
                    track_id=track_id,
                    bbox=(float(bbox[0]), float(bbox[1]), 
                          float(bbox[2]), float(bbox[3])),
                    confidence=confidence,
                    metadata={
                        'frame_id': self.frame_count,
                        'class_id': class_id,
                        'class_name': 'person'
                    }
                )
                people.append(person)
                
            except Exception as e:
                logger.warning(f"Failed to create Person for track {track_id}: {e}")
                continue
        
        return people