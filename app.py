"""
app.py - Smart Crowd Management System Orchestrator

This module is the ENTRY POINT and ORCHESTRATOR of the Smart Crowd Management System.
It has exactly ONE responsibility:

    Orchestrate the flow of data between backend modules and the dashboard.

app.py does NOT:
    - Perform detection (Detector does that)
    - Count people (Counter does that)
    - Calculate occupancy (Occupancy does that)
    - Assess risk (Risk does that)
    - Generate alerts (Alerts does that)
    - Write logs (Logger does that)
    - Render the UI (Dashboard does that)
    - Process camera frames (CameraProcessor does that)
    - Translate states to dashboard (DashboardAdapter does that)

app.py ONLY:
    - Initializes modules
    - Orchestrates the pipeline
    - Maintains performance metrics
    - Handles graceful shutdown

Architecture:
    Camera → CameraProcessor → Detector → Counter → Occupancy → Risk → Alerts → Logger → DashboardAdapter → Dashboard

Pipeline:
    CameraProcessor.get_frame() → Enhanced Frame
    Enhanced Frame → Detector.detect() → List[Person]
    List[Person] → Counter.update() → CounterState
    CounterState → Occupancy.update() → OccupancyState
    CounterState + OccupancyState → Risk.update() → RiskState
    RiskState → Alerts.update() → AlertState
    All States → Logger.log() → LoggerState
    All States → DashboardAdapter.dashboard_update_kwargs() → Dashboard.update()
"""

from __future__ import annotations

import json
import logging
from queue import Queue, Empty, Full
import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from queue import Empty, Full
from typing import Optional, Tuple, List, Dict, Any

import cv2
import numpy as np

# Import backend modules
from src.detector import Detector, DetectorConfig
from src.counter import Counter, CounterState
from src.occupancy import Occupancy, OccupancyState
from src.risk import Risk, RiskState
from src.alerts import Alerts, AlertState
from src.logger import Logger, LoggerState

# Import new modules
from src.camera_processor import CameraProcessor, CameraProcessorConfig, QualityMetrics
from src.dashboard_adapter import DashboardAdapter, AdapterConfig

# Import dashboard (only for type hints, adapter handles the rest)
from src.dashboard import Dashboard, Person, CameraHealth, PerformanceMetrics

# Configure logging
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class AppConfig:
    """
    Application configuration for the orchestrator.
    
    Contains settings for the camera, pipeline, and performance.
    Backend modules have their own configurations.
    
    Attributes:
        camera_source: Camera index or RTSP URL
        camera_width: Desired camera frame width
        camera_height: Desired camera frame height
        camera_fps: Target camera FPS
        camera_retry_interval: Seconds between camera reconnection attempts
        max_queue_size: Maximum size for thread-safe queues
        dashboard_refresh_rate: Dashboard update frequency (Hz)
        enable_logging: Whether to enable logging
        log_directory: Directory for log files
    """
    # Camera settings
    camera_source: str = "0"
    camera_width: int = 1280
    camera_height: int = 720
    camera_fps: int = 30
    camera_retry_interval: float = 2.0
    
    # Pipeline settings
    max_queue_size: int = 2
    dashboard_refresh_rate: int = 30
    
    # Logging
    enable_logging: bool = True
    log_directory: str = "logs"
    
    # Venue settings
    venue_name: str = "Main Hall"
    max_capacity: int = 120
    
    # Detector settings
    detector_model: str = "yolov8n.pt"
    detector_confidence: float = 0.5
    detector_device: str = "cpu"
    
    # Tracker settings
    tracker_timeout_frames: int = 30
    
    # Alert settings
    alert_cooldown_frames: int = 150
    
    # Logger settings
    logger_heartbeat_interval: int = 300
    
    # Camera processor settings
    camera_processor: CameraProcessorConfig = field(default_factory=CameraProcessorConfig)
    
    # Adapter settings
    adapter: AdapterConfig = field(default_factory=AdapterConfig)
    
    @classmethod
    def from_file(cls, path: str) -> AppConfig:
        """Load configuration from a JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)
    
    def to_file(self, path: str) -> None:
        """Save configuration to a JSON file."""
        with open(path, 'w') as f:
            json.dump(self.__dict__, f, indent=2, default=str)


# ============================================================================
# Application State
# ============================================================================

@dataclass
class AppState:
    """
    Runtime state of the application.
    
    Tracks the current status of the orchestrator.
    """
    is_running: bool = False
    frame_count: int = 0
    processed_count: int = 0
    start_time: datetime = field(default_factory=datetime.now)
    last_frame_time: float = 0.0
    fps_instant: float = 0.0
    fps_average: float = 0.0
    fps_samples: List[float] = field(default_factory=list)
    processing_time_ms: float = 0.0
    peak_processing_time_ms: float = 0.0
    error_count: int = 0
    
    # Module health
    detector_healthy: bool = True
    tracker_healthy: bool = True
    logger_healthy: bool = True


# ============================================================================
# Pipeline Data (Backend-Only)
# ============================================================================

@dataclass
class PipelineData:
    """
    Complete pipeline data for a single frame.
    
    Contains ONLY backend states. Dashboard conversion is handled by the adapter.
    
    Attributes:
        frame: Original frame (for display)
        people: List of detected Person objects
        counter_state: CounterState from Counter
        occupancy_state: OccupancyState from Occupancy
        risk_state: RiskState from Risk
        alert_state: AlertState from Alerts
        logger_state: LoggerState from Logger
        quality_metrics: QualityMetrics from CameraProcessor
        timestamp: When the data was produced
        frame_number: Frame number
    """
    frame: Optional[np.ndarray] = None
    people: List[Any] = field(default_factory=list)
    counter_state: Optional[CounterState] = None
    occupancy_state: Optional[OccupancyState] = None
    risk_state: Optional[RiskState] = None
    alert_state: Optional[AlertState] = None
    logger_state: Optional[LoggerState] = None
    quality_metrics: Optional[QualityMetrics] = None
    timestamp: datetime = field(default_factory=datetime.now)
    frame_number: int = 0


# ============================================================================
# Application Orchestrator
# ============================================================================

class Application:
    """
    Main application orchestrator for the Smart Crowd Management System.
    
    This class coordinates all backend modules, the camera processor,
    the dashboard adapter, and the dashboard.
    
    Architecture:
        - Thread 1: CameraProcessor (handles capture internally)
        - Thread 2: Processing pipeline → Dashboard Queue
        - Thread 3: Dashboard rendering (Tkinter main loop)
    
    Attributes:
        config: Application configuration
        state: Runtime state
        modules: Dictionary of backend modules
        frame_queue: Queue for processed frames
        dashboard_queue: Queue for pipeline data
        process_thread: Processing pipeline thread
        shutdown_event: Event for graceful shutdown
    """
    
    def __init__(self, config: Optional[AppConfig] = None) -> None:
        """
        Initialize the application orchestrator.
        
        Args:
            config: Application configuration (uses defaults if None)
        """
        self.config = config or AppConfig()
        self.state = AppState()
        self.modules: Dict[str, Any] = {}
        
        # Thread-safe queues
        self.frame_queue: Queue[np.ndarray] = Queue(maxsize=self.config.max_queue_size)
        self.dashboard_queue: Queue[PipelineData] = Queue(maxsize=self.config.max_queue_size)
        
        # Threading
        self.process_thread: Optional[threading.Thread] = None
        self.shutdown_event = threading.Event()
        
        # Dashboard
        self.dashboard: Optional[Dashboard] = None
        
        # FPS tracking
        self._fps_start_time: float = time.time()
        self._fps_frame_count: int = 0
        
        # Initialize modules
        self._initialize_modules()
        
        logger.info("Application orchestrator initialized successfully")
    
    # ========================================================================
    # Module Initialization
    # ========================================================================
    
    def _initialize_modules(self) -> None:
        """
        Initialize all backend modules.
        
        This is called ONCE at application startup.
        Modules are kept in memory for the entire application lifetime.
        """
        logger.info("Initializing backend modules...")
        
        try:
            # 0. Camera Processor - uses its own internal thread for capture
            camera_config = self.config.camera_processor
            camera_config.source = self.config.camera_source
            camera_config.width = self.config.camera_width
            camera_config.height = self.config.camera_height
            camera_config.fps = self.config.camera_fps
            camera_config.retry_interval = self.config.camera_retry_interval
            camera_config.max_queue_size = self.config.max_queue_size
            
            self.modules['camera_processor'] = CameraProcessor(camera_config)
            logger.info("✓ Camera Processor initialized")
            
            # 1. Detector
            detector_config = DetectorConfig(
                model_path=self.config.detector_model,
                confidence_threshold=self.config.detector_confidence,
                device=self.config.detector_device
            )
            self.modules['detector'] = Detector(detector_config)
            logger.info("✓ Detector initialized")
            
            # 2. Counter
            self.modules['counter'] = Counter(
                timeout_frames=self.config.tracker_timeout_frames
            )
            logger.info("✓ Counter initialized")
            
            # 3. Occupancy
            self.modules['occupancy'] = Occupancy(
                capacity=self.config.max_capacity,
                venue_name=self.config.venue_name
            )
            logger.info("✓ Occupancy initialized")
            
            # 4. Risk
            self.modules['risk'] = Risk()
            logger.info("✓ Risk initialized")
            
            # 5. Alerts
            self.modules['alerts'] = Alerts(
                cooldown_frames=self.config.alert_cooldown_frames
            )
            logger.info("✓ Alerts initialized")
            
            # 6. Logger
            self.modules['logger'] = Logger(
                log_dir=self.config.log_directory,
                heartbeat_interval=self.config.logger_heartbeat_interval
            )
            logger.info("✓ Logger initialized")
            
            # 7. Dashboard Adapter
            adapter_config = self.config.adapter
            adapter_config.default_capacity = self.config.max_capacity
            adapter_config.default_resolution = (self.config.camera_width, self.config.camera_height)
            adapter_config.enable_logging = self.config.enable_logging
            
            self.modules['adapter'] = DashboardAdapter(adapter_config)
            logger.info("✓ Dashboard Adapter initialized")
            
            # 8. Dashboard
            self.dashboard = Dashboard()
            self.dashboard.set_start_callback(self._on_start_monitoring)
            logger.info("✓ Dashboard initialized")
            
        except Exception as e:
            logger.error(f"Module initialization failed: {e}")
            raise
    
    # ========================================================================
    # Callback Methods
    # ========================================================================
    
    def _on_start_monitoring(self, venue_name: str, capacity: int) -> None:
        """
        Callback from the dashboard's Start Monitoring button.
        
        This starts the camera and processing pipeline.
        
        Args:
            venue_name: Venue name from dashboard
            capacity: Maximum capacity from dashboard
        """
        logger.info(f"Starting monitoring: venue='{venue_name}', capacity={capacity}")
        
        # Update configuration
        self.config.venue_name = venue_name
        self.config.max_capacity = capacity
        
        # Update modules
        self.modules['occupancy'].set_capacity(capacity)
        self.modules['occupancy'].venue_name = venue_name
        
        # Update adapter default capacity
        self.modules['adapter'].config.default_capacity = capacity
        
        # Start the pipeline
        self._start_pipeline()
        
        print("2 Callback reached")
    
    # ========================================================================
    # Pipeline Control
    # ========================================================================
    
    def _start_pipeline(self) -> None:
        """
        Start the camera capture and processing pipeline.
        
        CameraProcessor runs in its own thread.
        This starts the processing thread and the dashboard.
        """
        if self.state.is_running:
            logger.warning("Pipeline is already running")
            return
        
        self.state.is_running = True
        self.state.start_time = datetime.now()
        self.shutdown_event.clear()
        
        # Start camera processor (runs in its own thread)
        print("3 Starting camera processor")
        self.modules['camera_processor'].start()
        logger.info("Camera processor started")
        
        # Start processing thread (gets frames from camera processor)
        self.process_thread = threading.Thread(
            target=self._processing_loop,
            name="ProcessingPipeline",
            daemon=True
        )
        self.process_thread.start()
        logger.info("Processing pipeline thread started")
        
        # NOTE: Dashboard.run() is called from Application.run()
        # Not here - avoids starting the Tkinter main loop twice
        logger.info("Pipeline started. Waiting for dashboard to be ready...")
    
    def _stop_pipeline(self) -> None:
        """
        Stop the camera capture and processing pipeline.
        
        This signals threads to stop and waits for them to finish.
        """
        if not self.state.is_running:
            return
        
        logger.info("Stopping pipeline...")
        
        # Signal shutdown
        self.shutdown_event.set()
        self.state.is_running = False
        
        # Stop camera processor
        if 'camera_processor' in self.modules:
            self.modules['camera_processor'].stop()
        
        # Wait for processing thread to finish
        if self.process_thread and self.process_thread.is_alive():
            self.process_thread.join(timeout=2.0)
        
        # End logging session
        if 'logger' in self.modules:
            self.modules['logger'].end_session()
        
        logger.info("Pipeline stopped successfully")
    
    # ========================================================================
    # Processing Loop (Thread 2)
    # ========================================================================
    
    def _processing_loop(self) -> None:
        """
        Processing pipeline loop running in a dedicated thread.
        
        Gets frames from CameraProcessor and runs them through the pipeline:
        Detector → Counter → Occupancy → Risk → Alerts → Logger
        """
        camera_processor = self.modules['camera_processor']
        
        while not self.shutdown_event.is_set():
            try:
                # Get enhanced frame from camera processor
                frame = camera_processor.get_frame(timeout=0.5)
                
                if frame is None:
                    continue
                
                # Update state
                self.state.frame_count += 1
                self.state.last_frame_time = time.time()
                
                # Process the frame through the pipeline
                pipeline_start = time.perf_counter()
                pipeline_data = self._process_frame(frame)
                
                # Queue for dashboard
                if pipeline_data:
                    self._queue_dashboard_data(pipeline_data)
                    self.state.processed_count += 1
                pipeline_end = time.perf_counter()
                print(
                    f"Pipeline: {(pipeline_end - pipeline_start) * 1000:.1f} ms"
                    )
                
                # FPS tracking
                self._update_fps()
                
            except Empty:
                # No frame available, continue
                continue
            except Exception as e:
                logger.error(f"Processing loop error: {e}")
                self.state.error_count += 1
                self.state.detector_healthy = False
                
                # Continue processing (don't crash)
                time.sleep(0.1)
        
        logger.info("Processing loop stopped")
    
    def _queue_dashboard_data(self, data: PipelineData) -> None:
        """
        Queue pipeline data for dashboard with latest-wins policy.
        
        Args:
            data: PipelineData to queue
        """
        try:
            self.dashboard_queue.put(data, block=False)
        except Full:
            # Queue is full, remove oldest and add new
            try:
                self.dashboard_queue.get_nowait()
                self.dashboard_queue.put(data, block=False)
            except (Empty, Full):
                # If this fails, just log it
                logger.debug("Failed to queue dashboard data")
    
    def _process_frame(self, frame: np.ndarray) -> Optional[PipelineData]:
        """
        Process a single frame through the entire pipeline.
        
        Pipeline:
            1. Detector.detect() → List[Person]
            2. Counter.update() → CounterState
            3. Occupancy.update() → OccupancyState
            4. Risk.update() → RiskState
            5. Alerts.update() → AlertState
            6. Logger.log() → LoggerState
        
        Args:
            frame: Camera frame (BGR format)
            
        Returns:
            PipelineData containing all backend states for the frame
        """
        start_time = time.time()
        
        try:
            # Get camera quality metrics
            camera_processor = self.modules['camera_processor']
            quality_metrics = camera_processor.get_quality(timeout=0.0)
            
            # 1. Detection
            detect_start = time.time()
            _, people = self.modules['detector'].detect(frame)
            detect_time = (time.time() - detect_start) * 1000
            
            # Update detector health
            self.state.detector_healthy = True
            
            # 2. Counting
            counter_state = self.modules['counter'].update(people)
            self.state.tracker_healthy = True
            
            # 3. Occupancy
            occupancy_state = self.modules['occupancy'].update(counter_state)
            
            # 4. Risk
            risk_state = self.modules['risk'].update(
                counter_state,
                occupancy_state
            )
            
            # 5. Alerts
            alert_state = self.modules["alerts"].update(
                risk_state,
                frame=frame
                )
            
            # 6. Logger
            fps = self.state.fps_instant
            processing_time = (time.time() - start_time) * 1000
            
            logger_state = self.modules['logger'].log(
                counter_state=counter_state,
                occupancy_state=occupancy_state,
                risk_state=risk_state,
                alert_state=alert_state,
                fps=fps,
                processing_time=processing_time
            )
            self.state.logger_healthy = True
            
            # 7. Build pipeline data (backend-only)
            pipeline_data = PipelineData(
                frame=frame.copy(),
                people=people,
                counter_state=counter_state,
                occupancy_state=occupancy_state,
                risk_state=risk_state,
                alert_state=alert_state,
                logger_state=logger_state,
                quality_metrics=quality_metrics,
                timestamp=datetime.now(),
                frame_number=self.state.frame_count
            )
            
            # 8. Update state
            self.state.processing_time_ms = processing_time
            if processing_time > self.state.peak_processing_time_ms:
                self.state.peak_processing_time_ms = processing_time
            
            return pipeline_data
            
        except Exception as e:
            logger.error(f"Frame processing error: {e}")
            self.state.error_count += 1
            self.state.detector_healthy = False
            return None
    
    # ========================================================================
    # Dashboard Updater (Thread 3 is Tkinter main loop)
    # ========================================================================
    
    def _drain_dashboard_queue(self) -> None:
        """
        Drain the dashboard queue and update the dashboard.
        
        This runs in the Tkinter main loop.
        Uses the DashboardAdapter for all state translations.
        """
        try:
            while True:
                data = self.dashboard_queue.get_nowait()
                self._push_to_dashboard(data)
        except Empty:
            pass
        
        # Schedule next drain
        if self.dashboard:
            refresh_ms = int(1000 / self.config.dashboard_refresh_rate)
            self.dashboard.root.after(refresh_ms, self._drain_dashboard_queue)
            
    
    def _push_to_dashboard(self, data: PipelineData) -> None:
        """
        Push pipeline data to the dashboard using the adapter.
        
        Args:
            data: PipelineData from processing
        """
        if not self.dashboard:
            return
        
        adapter = self.modules['adapter']
        camera_processor = self.modules['camera_processor']
        
        # Get camera health from processor
        camera_health = camera_processor.get_camera_health()
        
        # Map camera quality to dashboard format using constants
        quality_map = {
            'EXCELLENT': 'GOOD',
            'GOOD': 'GOOD',
            'FAIR': 'GOOD',
            'POOR': 'BLUR',
            'VERY POOR': 'OFFLINE',
            'OFFLINE': 'OFFLINE'
        }
        camera_quality = quality_map.get(camera_health['overall_quality'], 'OFFLINE')
        
        # Use the adapter to convert all states
        frame_to_display = data.frame.copy() if isinstance(data.frame, np.ndarray) else data.frame

        self.dashboard.update(
            frame=frame_to_display,
            **adapter.dashboard_update_kwargs(
                counter=data.counter_state,
                occupancy=data.occupancy_state,
                risk=data.risk_state,
                alert=data.alert_state,
                people=data.people,
                camera_online=camera_health['online'],
                camera_quality=camera_quality,
                camera_resolution=(self.config.camera_width, self.config.camera_height),
                fps=self.state.fps_instant,
                detector_ready=self.state.detector_healthy,
                tracker_ready=self.state.tracker_healthy,
                logger_active=self.state.logger_healthy
            )
        )
    
    # ========================================================================
    # FPS Tracking
    # ========================================================================
    
    def _update_fps(self) -> None:
        """
        Update FPS tracking.
        
        Calculates both instant and average FPS.
        """
        self._fps_frame_count += 1
        
        current_time = time.time()
        
        # Calculate instant FPS every second
        if current_time - self._fps_start_time >= 1.0:
            instant_fps = self._fps_frame_count / (current_time - self._fps_start_time)
            self.state.fps_instant = instant_fps
            
            # Update average FPS (rolling average)
            self.state.fps_samples.append(instant_fps)
            if len(self.state.fps_samples) > 60:  # Keep last 60 samples
                self.state.fps_samples.pop(0)
            
            self.state.fps_average = sum(self.state.fps_samples) / len(self.state.fps_samples)
            
            # Reset counter
            self._fps_frame_count = 0
            self._fps_start_time = current_time
    
    # ========================================================================
    # Application Lifecycle
    # ========================================================================
    
    def run(self) -> None:
        """
        Run the application.
        
        This is the MAIN ENTRY POINT.
        It starts the dashboard and blocks until the application exits.
        """
        logger.info("Starting Smart Crowd Management System...")
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Start the dashboard (shows startup screen)
        if self.dashboard:
            # Start the dashboard queue drain in Tkinter main loop
            self.dashboard.root.after(100, self._drain_dashboard_queue)
            # This blocks until the window is closed
            self.dashboard.run()
        
        # Application exit
        self.shutdown()
    
    def _signal_handler(self, signum: int, frame) -> None:
        """
        Handle system signals for graceful shutdown.
        
        Args:
            signum: Signal number
            frame: Current stack frame
        """
        logger.info(f"Received signal {signum}, shutting down...")
        self.shutdown()
    
    def shutdown(self) -> None:
        """
        Gracefully shutdown the application.
        
        Stops all threads, releases resources, and cleans up.
        """
        logger.info("Shutting down application...")
        
        # Stop the pipeline
        self._stop_pipeline()
        
        # Clean up dashboard
        if self.dashboard:
            try:
                self.dashboard.root.quit()
                self.dashboard.root.destroy()
            except Exception as e:
                logger.debug(f"Dashboard cleanup: {e}")
        
        # Print final statistics
        self._print_statistics()
        
        logger.info("Application shutdown complete")
        sys.exit(0)
    
    def _print_statistics(self) -> None:
        """Print final runtime statistics."""
        elapsed = (datetime.now() - self.state.start_time).total_seconds()
        
        print("\n" + "="*60)
        print("APPLICATION STATISTICS")
        print("="*60)
        print(f"Total Frames Captured: {self.state.frame_count}")
        print(f"Total Frames Processed: {self.state.processed_count}")
        print(f"Total Errors: {self.state.error_count}")
        print(f"Average FPS: {self.state.fps_average:.1f}")
        print(f"Peak Processing Time: {self.state.peak_processing_time_ms:.1f}ms")
        print(f"Average Processing Time: {self.state.processing_time_ms:.1f}ms")
        print(f"Elapsed Time: {elapsed:.1f} seconds")
        print("="*60)


# ============================================================================
# Main Entry Point
# ============================================================================

def main() -> None:
    """
    Main entry point for the application.
    
    Sets up logging, creates the application, and runs it.
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('app.log')
        ]
    )
    
    # Suppress verbose third-party logs
    logging.getLogger('ultralytics').setLevel(logging.WARNING)
    logging.getLogger('cv2').setLevel(logging.WARNING)
    
    # Create application configuration
    config = AppConfig(
        camera_source="0",
        camera_width=1280,
        camera_height=720,
        camera_fps=30,
        venue_name="Main Hall",
        max_capacity=120,
        detector_model="yolov8n.pt",
        detector_confidence=0.5,
        detector_device="cpu",
        tracker_timeout_frames=30,
        alert_cooldown_frames=150
    )
    
    # Create and run application
    try:
        app = Application(config)
        app.run()
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    except Exception as e:
        logger.error(f"Application error: {e}")
        raise


if __name__ == "__main__":
    main()