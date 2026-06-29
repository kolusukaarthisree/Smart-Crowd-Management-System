"""
app.py - Application Controller Module

This module is the SYSTEM CONTROLLER of the Smart Crowd Management System.
Its purpose is to initialize, coordinate, and manage the complete lifecycle
of the application.

Author: System Architect
Version: 2.2.0 (Optimized)
"""

from __future__ import annotations

import argparse
import cv2
import logging
import sys
import time
import signal
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple, Dict, Any

from src.detector import Detector, DetectorConfig
from src.counter import Counter
from src.occupancy import Occupancy
from src.risk import Risk
from src.alerts import Alerts
from src.logger import Logger
from src.dashboard import Dashboard

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CameraStatus(Enum):
    CONNECTED = "CONNECTED"
    DISCONNECTED = "DISCONNECTED"
    RECONNECTING = "RECONNECTING"
    FROZEN = "FROZEN"
    OK = "OK"


@dataclass
class AppConfig:
    camera_index: int = 0
    camera_width: int = 1280
    camera_height: int = 720
    camera_fps: int = 30
    use_mjpeg: bool = True
    buffer_size: int = 1
    window_title: str = "Smart Crowd Management System"
    max_capacity: int = 50
    confidence_threshold: float = 0.5
    timeout_frames: int = 30
    log_dir: str = "logs"
    show_fps: bool = True
    show_frame_number: bool = False
    verbose: bool = False
    model_path: str = "yolov8n.pt"
    device: str = "cpu"
    venue_name: str = "Main Hall"
    enable_enhancement: bool = True
    sharpening_strength: float = 0.5
    clahe_clip_limit: float = 2.0
    clahe_grid_size: int = 8
    denoise_strength: float = 0.0
    brightness_correction: float = 0.0
    max_reconnect_attempts: int = 10
    reconnect_delay: float = 2.0
    sharpness_threshold: float = 100.0


@dataclass
class CameraHealth:
    status: CameraStatus = CameraStatus.CONNECTED
    resolution: Tuple[int, int] = (0, 0)
    fps: float = 0.0
    frame_count: int = 0
    last_frame_time: float = 0.0
    frozen_frames: int = 0
    sharpness_score: float = 0.0
    brightness_score: float = 0.0
    is_healthy: bool = True
    reconnect_attempts: int = 0
    is_blurry: bool = False


@dataclass
class PerformanceMetrics:
    frame_count: int = 0
    total_time: float = 0.0
    fps: float = 0.0
    avg_fps: float = 0.0
    processing_time: float = 0.0
    avg_processing_time: float = 0.0
    dropped_frames: int = 0
    last_frame_time: float = 0.0
    start_time: float = field(default_factory=time.time)


class SmartCrowdManagementSystem:
    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or AppConfig()
        self.modules = {}
        self.camera = None
        self.running = False
        self.performance = PerformanceMetrics()
        self.health = CameraHealth()
        
        signal.signal(signal.SIGINT, lambda s, f: self.stop())
        signal.signal(signal.SIGTERM, lambda s, f: self.stop())
        
        self.clahe = cv2.createCLAHE(self.config.clahe_clip_limit, (self.config.clahe_grid_size,) * 2)
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]]) * self.config.sharpening_strength
        kernel[1,1] = 1 + self.config.sharpening_strength * 8
        self.sharpening_kernel = kernel
        logger.info("Application Controller initialized")

    def initialize(self) -> bool:
        logger.info("Initializing Smart Crowd Management System...")
        try:
            self._init_modules()
            self._init_camera()
            self.modules['logger'].start_session()
            self._validate()
            logger.info("System initialization complete")
            return True
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            self.cleanup()
            return False

    def _init_modules(self):
        logger.info("Initializing modules...")
        self.modules['detector'] = Detector(DetectorConfig(
            model_path=self.config.model_path,
            confidence_threshold=self.config.confidence_threshold,
            device=self.config.device
        ))
        self.modules['counter'] = Counter(timeout_frames=self.config.timeout_frames)
        self.modules['occupancy'] = Occupancy(capacity=self.config.max_capacity, venue_name=self.config.venue_name)
        self.modules['risk'] = Risk()
        self.modules['alerts'] = Alerts()
        self.modules['logger'] = Logger(log_dir=self.config.log_dir)
        self.modules['dashboard'] = Dashboard(
            show_fps=self.config.show_fps,
            show_frame_number=self.config.show_frame_number,
            camera_name=self.config.venue_name
        )
        for name in self.modules:
            logger.info(f"  ✓ {name.capitalize()} initialized")

    def _init_camera(self):
        logger.info(f"Initializing camera {self.config.camera_index}...")
        if not self._open_camera():
            raise RuntimeError("Failed to open camera")
        logger.info("  ✓ Camera initialization complete")

    def _open_camera(self) -> bool:
        try:
            self.camera = cv2.VideoCapture(self.config.camera_index, cv2.CAP_DSHOW)
            if not self.camera.isOpened():
                return False
            
            if self.config.use_mjpeg:
                self.camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            
            for w, h in [(self.config.camera_width, self.config.camera_height), (1280, 720)]:
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, w)
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
                aw, ah = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
                if aw == w and ah == h:
                    break
            
            self.health.resolution = (aw, ah)
            self.health.status = CameraStatus.CONNECTED
            logger.info(f"  ✓ Resolution: {aw}x{ah}")
            
            if self.config.camera_fps > 0:
                self.camera.set(cv2.CAP_PROP_FPS, self.config.camera_fps)
            
            self.camera.set(cv2.CAP_PROP_BUFFERSIZE, self.config.buffer_size)
            logger.info(f"  ✓ Buffer size: {self.config.buffer_size}")
            
            ret, frame = self.camera.read()
            if not ret or frame is None or frame.size == 0:
                raise RuntimeError("Camera validation failed")
            
            logger.info("  ✓ Camera validation passed")
            return True
        except Exception as e:
            logger.error(f"Camera open failed: {e}")
            return False

    def _validate(self):
        if self.camera is None or not self.camera.isOpened():
            raise RuntimeError("Camera not available")
        required = ['detector', 'counter', 'occupancy', 'risk', 'alerts', 'logger', 'dashboard']
        for name in required:
            if name not in self.modules:
                raise RuntimeError(f"Module {name} not initialized")

    def _get_frame(self) -> Optional[np.ndarray]:
        if self.camera is None:
            return None
        
        ret, frame = self.camera.read()
        
        if not ret or frame is None:
            self.performance.dropped_frames += 1
            self.health.status = CameraStatus.DISCONNECTED
            self.health.is_healthy = False
            logger.warning("Camera disconnected - attempting to reconnect...")
            if self._attempt_reconnect():
                logger.info("Camera reconnected successfully")
                ret, frame = self.camera.read()
                if not ret or frame is None:
                    return None
            else:
                return None
        
        self.health.frame_count += 1
        self.health.status = CameraStatus.OK
        self.health.is_healthy = True
        self.health.reconnect_attempts = 0
        
        # Check frozen
        now = time.time()
        if self.health.last_frame_time > 0 and (now - self.health.last_frame_time) > 2.0:
            self.health.frozen_frames += 1
            if self.health.frozen_frames > 5:
                self.health.status = CameraStatus.FROZEN
                self.health.is_healthy = False
                logger.warning("Camera frozen - attempting to reconnect...")
                if self._attempt_reconnect():
                    logger.info("Camera reconnected after freeze")
                    ret, frame = self.camera.read()
                    if ret and frame is not None:
                        self.health.frozen_frames = 0
                    else:
                        return None
                else:
                    return None
        self.health.last_frame_time = now
        
        # Quality analysis
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.health.sharpness_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        self.health.brightness_score = np.mean(gray)
        self.health.is_blurry = self.health.sharpness_score < self.config.sharpness_threshold
        
        # Enhancement
        if self.config.enable_enhancement:
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l = self.clahe.apply(l)
            frame = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
            if self.config.sharpening_strength > 0:
                frame = cv2.filter2D(frame, -1, self.sharpening_kernel)
            if self.config.denoise_strength > 0:
                frame = cv2.fastNlMeansDenoisingColored(frame, None, self.config.denoise_strength, 
                                                       self.config.denoise_strength * 0.5, 7, 21)
            if self.config.brightness_correction != 0:
                frame = cv2.add(frame, np.full(frame.shape, self.config.brightness_correction, dtype=np.uint8))
        
        return frame

    def _attempt_reconnect(self) -> bool:
        if self.health.reconnect_attempts >= self.config.max_reconnect_attempts:
            logger.error(f"Max reconnection attempts ({self.config.max_reconnect_attempts}) reached")
            return False
        
        self.health.status = CameraStatus.RECONNECTING
        for attempt in range(1, self.config.max_reconnect_attempts + 1):
            logger.info(f"Reconnection attempt {attempt}/{self.config.max_reconnect_attempts}...")
            if self.camera:
                self.camera.release()
                self.camera = None
            time.sleep(self.config.reconnect_delay)
            if self._open_camera():
                self.health.reconnect_attempts = 0
                self.health.status = CameraStatus.CONNECTED
                self.health.is_healthy = True
                return True
            self.health.reconnect_attempts = attempt
        logger.error("Failed to reconnect to camera")
        self.health.status = CameraStatus.DISCONNECTED
        return False

    def _process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        m = self.modules
        _, people = m['detector'].detect(frame)
        counter_state = m['counter'].update(people)
        occupancy_state = m['occupancy'].update(counter_state)
        risk_state = m['risk'].update(counter_state, occupancy_state)
        alert_state = m['alerts'].update(risk_state)
        
        annotated = m['dashboard'].render(
            frame=frame, people=people, counter_state=counter_state,
            occupancy_state=occupancy_state, risk_state=risk_state,
            alert_state=alert_state, fps=self.performance.fps
        )
        
        m['logger'].log(counter_state, occupancy_state, risk_state, alert_state,
                       self.performance.fps, self.performance.processing_time)
        
        return annotated, {'people': people, 'counter': counter_state, 
                          'occupancy': occupancy_state, 'risk': risk_state, 'alerts': alert_state}

    def _update_performance(self, elapsed: float):
        self.performance.frame_count += 1
        self.performance.processing_time = elapsed * 1000
        self.performance.total_time += elapsed
        self.performance.avg_processing_time = (self.performance.total_time / self.performance.frame_count) * 1000
        
        now = time.time()
        if self.performance.last_frame_time > 0:
            self.performance.fps = 1.0 / (now - self.performance.last_frame_time) if (now - self.performance.last_frame_time) > 0 else 0
        self.performance.last_frame_time = now
        self.performance.avg_fps = self.performance.frame_count / self.performance.total_time if self.performance.total_time > 0 else 0

    def _display_performance(self):
        if self.config.verbose:
            blur = "🔴BLUR" if self.health.is_blurry else "✅"
            print(f"\rFPS: {self.performance.fps:.1f} | Avg: {self.performance.avg_fps:.1f} | "
                  f"Process: {self.performance.processing_time:.1f}ms | Frames: {self.performance.frame_count} | "
                  f"Sharp: {self.health.sharpness_score:.1f} {blur}", end="")

    def run(self):
        if not self.running:
            self.running = True
            logger.info("Main processing loop started")
        
        while self.running:
            try:
                start = time.time()
                frame = self._get_frame()
                if frame is None:
                    time.sleep(0.1)
                    self.performance.dropped_frames += 1
                    continue
                
                annotated, _ = self._process_frame(frame)
                cv2.imshow(self.config.window_title, annotated)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logger.info("Exit requested by user")
                    self.stop()
                    break
                elif key == ord('r'):
                    for mod in ['counter', 'occupancy', 'risk', 'alerts']:
                        if mod in self.modules:
                            self.modules[mod].reset()
                    self.performance = PerformanceMetrics()
                    logger.info("System reset complete")
                elif key == ord('e'):
                    self.config.enable_enhancement = not self.config.enable_enhancement
                    logger.info(f"Enhancement: {'ON' if self.config.enable_enhancement else 'OFF'}")
                
                self._update_performance(time.time() - start)
                self._display_performance()
                
            except Exception as e:
                if not self._handle_error(e):
                    self.stop()
                    break

    def _handle_error(self, error: Exception) -> bool:
        logger.error(f"Error: {error}")
        if 'logger' in self.modules:
            self.modules['logger'].log_error(str(error), "ERROR", error)
        
        err = str(error).lower()
        if any(x in err for x in ['camera', 'disconnect']):
            logger.critical("Camera error - attempting to reconnect...")
            return self._attempt_reconnect()
        if any(x in err for x in ['memory', 'out of memory']):
            logger.critical("Memory error - shutting down")
            return False
        return True

    def stop(self):
        if self.running:
            logger.info("Stopping processing loop...")
            self.running = False
            self.cleanup()

    def cleanup(self):
        logger.info("Cleaning up system...")
        self.running = False
        if self.camera:
            self.camera.release()
            self.camera = None
        if 'logger' in self.modules:
            try:
                self.modules['logger'].end_session()
                self.modules['logger'].flush()
            except Exception as e:
                logger.error(f"Logger error: {e}")
        cv2.destroyAllWindows()
        self._display_final_stats()
        logger.info("System cleanup complete")

    def _display_final_stats(self):
        if self.performance.frame_count > 0:
            print("\n" + "="*50)
            print("Final Statistics:")
            print("="*50)
            stats = [
                f"Total Frames:      {self.performance.frame_count}",
                f"Average FPS:       {self.performance.avg_fps:.1f}",
                f"Avg Processing:    {self.performance.avg_processing_time:.1f}ms",
                f"Dropped Frames:    {self.performance.dropped_frames}",
                f"Resolution:        {self.health.resolution[0]}x{self.health.resolution[1]}",
                f"Reconnections:     {self.health.reconnect_attempts}",
                f"Blurry Frames:     {'Yes' if self.health.is_blurry else 'No'}"
            ]
            if 'logger' in self.modules:
                s = self.modules['logger'].get_session_summary()
                stats.extend([f"Events Logged:     {s['events_logged']}", 
                             f"Alerts Logged:     {s['alerts_logged']}",
                             f"Errors Logged:     {s['errors_logged']}"])
            print("\n".join(stats))
            print("="*50)


def main():
    parser = argparse.ArgumentParser(description="Smart Crowd Management System")
    parser.add_argument("--camera", "-c", type=int, default=0)
    parser.add_argument("--width", "-W", type=int, default=1280)
    parser.add_argument("--height", "-H", type=int, default=720)
    parser.add_argument("--fps", "-f", type=int, default=30)
    parser.add_argument("--no-mjpeg", action="store_true")
    parser.add_argument("--capacity", "-cap", type=int, default=50)
    parser.add_argument("--model", "-m", type=str, default="yolov8n.pt")
    parser.add_argument("--device", "-d", type=str, default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--confidence", "-conf", type=float, default=0.5)
    parser.add_argument("--log-dir", "-l", type=str, default="logs")
    parser.add_argument("--venue", "-v", type=str, default="Main Hall")
    parser.add_argument("--no-enhance", action="store_true")
    parser.add_argument("--reconnect-attempts", type=int, default=10)
    parser.add_argument("--reconnect-delay", type=float, default=2.0)
    parser.add_argument("--sharpness-threshold", type=float, default=100.0)
    parser.add_argument("--verbose", "-V", action="store_true")
    args = parser.parse_args()

    config = AppConfig(
        camera_index=args.camera, camera_width=args.width, camera_height=args.height,
        camera_fps=args.fps, use_mjpeg=not args.no_mjpeg, max_capacity=args.capacity,
        model_path=args.model, device=args.device, confidence_threshold=args.confidence,
        log_dir=args.log_dir, venue_name=args.venue, enable_enhancement=not args.no_enhance,
        max_reconnect_attempts=args.reconnect_attempts, reconnect_delay=args.reconnect_delay,
        sharpness_threshold=args.sharpness_threshold, verbose=args.verbose
    )

    system = SmartCrowdManagementSystem(config)
    if system.initialize():
        try:
            system.run()
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            system.cleanup()
    else:
        logger.error("Failed to initialize system")
        sys.exit(1)


if __name__ == "__main__":
    main()