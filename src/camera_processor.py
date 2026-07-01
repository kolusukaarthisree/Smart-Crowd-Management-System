"""
camera_processor.py - Professional Camera Pipeline Module

This module handles ALL camera-related operations for the Smart Crowd Management System.
It has exactly ONE responsibility:

    Transform raw camera frames into enhanced, quality-assessed frames ready for detection.

CameraProcessor does NOT:
    - Perform detection (Detector does that)
    - Count people (Counter does that)
    - Render UI (Dashboard does that)

CameraProcessor ONLY:
    - Captures frames
    - Assesses image quality (blur, brightness, contrast, noise, exposure)
    - Enhances frames (CLAHE, gamma correction, sharpening, noise reduction)
    - Manages camera reconnection
    - Provides quality scoring

Architecture:
    Raw Frame → Quality Assessment → Enhancement → Enhanced Frame + Quality Report
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from queue import Queue, Empty
from typing import Optional, Tuple, List, Dict, Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# Enums
# ============================================================================

class CameraQuality(Enum):
    """
    Camera quality classification.
    
    Based on comprehensive quality metrics including blur, brightness,
    contrast, noise, and exposure.
    """
    EXCELLENT = "EXCELLENT"  # 90-100
    GOOD = "GOOD"           # 70-89
    FAIR = "FAIR"           # 50-69
    POOR = "POOR"           # 30-49
    VERY_POOR = "VERY POOR" # 10-29
    OFFLINE = "OFFLINE"     # 0-9


class EnhancementType(Enum):
    """Types of image enhancement operations."""
    CLAHE = "CLAHE"
    GAMMA = "GAMMA"
    SHARPEN = "SHARPEN"
    DENOISE = "DENOISE"
    CONTRAST = "CONTRAST"


# ============================================================================
# Quality Metrics
# ============================================================================

@dataclass
class QualityMetrics:
    """
    Comprehensive image quality assessment results.
    
    Attributes:
        blur_score: Variance of Laplacian (higher = sharper)
        blur_status: 'Sharp', 'Moderate', or 'Blurry'
        brightness_mean: Average pixel intensity (0-255)
        brightness_status: 'Too Dark', 'Dark', 'Normal', 'Bright', or 'Too Bright'
        contrast_std: Standard deviation of pixel intensities
        contrast_status: 'Low', 'Normal', or 'High'
        noise_level: Estimated noise level (lower = less noise)
        noise_status: 'Low', 'Moderate', or 'High'
        exposure_score: Histogram-based exposure (0-100)
        exposure_status: 'Under', 'Normal', or 'Over'
        overall_score: Combined quality score (0-100)
        overall_quality: CameraQuality enum
        frame_number: Frame number when metrics were computed
        timestamp: When metrics were computed
    """
    blur_score: float = 0.0
    blur_status: str = "Unknown"
    brightness_mean: float = 0.0
    brightness_status: str = "Unknown"
    contrast_std: float = 0.0
    contrast_status: str = "Unknown"
    noise_level: float = 0.0
    noise_status: str = "Unknown"
    exposure_score: float = 0.0
    exposure_status: str = "Unknown"
    overall_score: float = 0.0
    overall_quality: CameraQuality = CameraQuality.GOOD
    frame_number: int = 0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class EnhancementReport:
    """
    Report of enhancement operations applied to a frame.
    
    Attributes:
        operations: List of enhancement types applied
        processing_time_ms: Time taken for enhancement
        original_dims: Original frame dimensions
        enhanced_dims: Enhanced frame dimensions
    """
    operations: List[EnhancementType] = field(default_factory=list)
    processing_time_ms: float = 0.0
    original_dims: Tuple[int, int] = (0, 0)
    enhanced_dims: Tuple[int, int] = (0, 0)


# ============================================================================
# Camera Processor Configuration
# ============================================================================

@dataclass
class CameraProcessorConfig:
    """
    Configuration for the camera processor.
    
    Attributes:
        source: Camera index or RTSP URL
        width: Desired frame width
        height: Desired frame height
        fps: Target FPS
        retry_interval: Seconds between reconnection attempts
        max_queue_size: Maximum frame queue size
        
        # Enhancement settings (disabled by default for performance)
        enable_enhancement: Whether to apply enhancement (default False)
        use_clahe: Apply CLAHE contrast enhancement
        use_gamma: Apply gamma correction
        use_sharpen: Apply sharpening
        use_denoise: Apply denoising
        gamma_value: Gamma correction value (0.1-2.0)
        clahe_clip_limit: CLAHE clip limit
        clahe_grid_size: CLAHE grid size
        sharpen_strength: Sharpening strength
        denoise_strength: Denoising strength
        
        # Quality assessment thresholds
        blur_threshold: Below this is considered blurry
        brightness_min: Minimum acceptable brightness
        brightness_max: Maximum acceptable brightness
        contrast_min: Minimum acceptable contrast
        noise_max: Maximum acceptable noise level
        exposure_good_min: Minimum good exposure score
        exposure_good_max: Maximum good exposure score
        
        # Performance optimizations
        quality_assess_interval: Compute quality every N frames (default 10)
        skip_frames: Drop every N-th frame to reduce load (0 = no skip)
        enhance_only_if_poor: Only enhance if quality is below GOOD (default True)
    """
    # Camera settings
    source: str = "0"
    width: int = 640          # lowered from 1280 for performance
    height: int = 480         # lowered from 720 for performance
    fps: int = 30
    retry_interval: float = 2.0
    max_queue_size: int = 2
    
    # Enhancement settings - DISABLED by default
    enable_enhancement: bool = False   # ← OFF for speed
    use_clahe: bool = True
    use_gamma: bool = True
    use_sharpen: bool = True
    use_denoise: bool = True
    gamma_value: float = 1.2
    clahe_clip_limit: float = 2.0
    clahe_grid_size: int = 8
    sharpen_strength: float = 0.3
    denoise_strength: float = 10.0
    
    # Quality thresholds
    blur_threshold: float = 100.0
    brightness_min: float = 50.0
    brightness_max: float = 200.0
    contrast_min: float = 30.0
    noise_max: float = 50.0
    exposure_good_min: float = 30.0
    exposure_good_max: float = 70.0

    # Performance tuning
    quality_assess_interval: int = 10    # compute quality every 10 frames
    skip_frames: int = 0                 # 0 = no skip, 1 = skip every other frame
    enhance_only_if_poor: bool = True    # avoid enhancement on already good frames


# ============================================================================
# Camera Processor
# ============================================================================

class CameraProcessor:
    """
    Professional Camera Pipeline Processor.
    
    Handles camera capture, frame enhancement, and quality assessment.
    Runs in its own thread and provides enhanced frames to the pipeline.
    
    Architecture:
        Camera → Raw Frame → Quality Assessment → Enhancement → Queue
        
    Attributes:
        config: Processor configuration
        frame_queue: Thread-safe queue for enhanced frames
        quality_queue: Thread-safe queue for quality metrics
        is_running: Whether the processor is running
        frame_count: Total frames processed
        capture_thread: Camera capture thread
        shutdown_event: Signal for graceful shutdown
    """
    
    def __init__(self, config: Optional[CameraProcessorConfig] = None) -> None:
        """
        Initialize the camera processor.
        
        Args:
            config: Processor configuration (uses defaults if None)
        """
        self.config = config or CameraProcessorConfig()
        self.frame_queue: Queue[np.ndarray] = Queue(maxsize=self.config.max_queue_size)
        self.quality_queue: Queue[QualityMetrics] = Queue(maxsize=self.config.max_queue_size)
        
        self.is_running = False
        self.frame_count = 0
        self.fps_samples: List[float] = []
        self.current_fps: float = 0.0
        
        # Threading
        self.capture_thread: Optional[threading.Thread] = None
        self.shutdown_event = threading.Event()
        
        # Camera state
        self.is_camera_online = False
        self.reconnect_count = 0
        self.last_frame_time = 0.0
        
        # Quality tracking
        self.last_quality: Optional[QualityMetrics] = None
        self.quality_history: List[QualityMetrics] = []
        
        # For skipping frames
        self._frame_skip_counter = 0
        
        logger.info(f"CameraProcessor initialized: source='{self.config.source}'")
        logger.info(f"Enhancement: {'ON' if self.config.enable_enhancement else 'OFF'}, "
                    f"Resolution: {self.config.width}x{self.config.height}, "
                    f"Quality interval: {self.config.quality_assess_interval}")
    
    # ========================================================================
    # Public Interface
    # ========================================================================
    
    def start(self) -> None:
        """
        Start the camera capture thread.
        
        This begins capturing, processing, and enhancing frames.
        """
        if self.is_running:
            logger.warning("Camera processor is already running")
            return
        
        self.is_running = True
        self.shutdown_event.clear()
        
        self.capture_thread = threading.Thread(
            target=self._camera_loop,
            name="CameraProcessor",
            daemon=True
        )
        self.capture_thread.start()
        
        logger.info("Camera processor started")
    
    def stop(self) -> None:
        """
        Stop the camera capture thread.
        
        Signals the thread to exit and waits for it to finish.
        """
        if not self.is_running:
            return
        
        logger.info("Stopping camera processor...")
        self.shutdown_event.set()
        self.is_running = False
        
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=3.0)
        
        logger.info("Camera processor stopped")
    
    def get_frame(self, timeout: float = 0.033) -> Optional[np.ndarray]:
        """
        Get the latest enhanced frame.
        
        Args:
            timeout: Maximum time to wait for a frame (seconds)
            
        Returns:
            Enhanced frame or None if timeout
        """
        try:
            return self.frame_queue.get(timeout=timeout)
        except Empty:
            return None
    
    def get_quality(self, timeout: float = 0.033) -> Optional[QualityMetrics]:
        """
        Get the latest quality metrics.
        
        Args:
            timeout: Maximum time to wait for quality metrics
            
        Returns:
            QualityMetrics or None if timeout
        """
        try:
            return self.quality_queue.get(timeout=timeout)
        except Empty:
            return None
    
    def get_camera_health(self) -> Dict[str, Any]:
        """
        Get comprehensive camera health information.
        
        Returns:
            Dictionary with camera health status
        """
        quality = self.last_quality
        
        return {
            'online': self.is_camera_online,
            'reconnect_count': self.reconnect_count,
            'frame_count': self.frame_count,
            'fps': self.current_fps,
            'overall_quality': quality.overall_quality.value if quality else "UNKNOWN",
            'overall_score': quality.overall_score if quality else 0.0,
            'blur_score': quality.blur_score if quality else 0.0,
            'blur_status': quality.blur_status if quality else "Unknown",
            'brightness_mean': quality.brightness_mean if quality else 0.0,
            'brightness_status': quality.brightness_status if quality else "Unknown",
            'contrast_std': quality.contrast_std if quality else 0.0,
            'contrast_status': quality.contrast_status if quality else "Unknown",
            'noise_level': quality.noise_level if quality else 0.0,
            'noise_status': quality.noise_status if quality else "Unknown",
            'exposure_score': quality.exposure_score if quality else 0.0,
            'exposure_status': quality.exposure_status if quality else "Unknown"
        }
    
    # ========================================================================
    # Camera Loop (Main Thread)
    # ========================================================================
    
    def _camera_loop(self) -> None:
        """
        Main camera capture and processing loop.
        
        Runs in a dedicated thread and continuously:
            1. Captures frames from camera
            2. Assesses image quality (every N frames)
            3. Enhances frames (if enabled and needed)
            4. Puts frames in queue
            
        Implements automatic reconnection on failure.
        """
        cap = None
        fps_timer = time.time()
        fps_counter = 0
        
        while not self.shutdown_event.is_set():
            try:
                # Open camera if needed
                if cap is None or not cap.isOpened():
                    cap = self._open_camera()
                    if cap is None:
                        self.is_camera_online = False
                        time.sleep(self.config.retry_interval)
                        continue
                
                # Capture frame
                ret, frame = cap.read()
                
                if not ret or frame is None:
                    logger.warning("Frame read failed, reconnecting...")
                    self.is_camera_online = False
                    self.reconnect_count += 1
                    cap.release()
                    cap = None
                    time.sleep(self.config.retry_interval)
                    continue
                
                # Camera is online
                if not self.is_camera_online:
                    self.is_camera_online = True
                    logger.info("Camera reconnected successfully")
                
                # Frame skipping (reduce load)
                if self.config.skip_frames > 0:
                    self._frame_skip_counter += 1
                    if self._frame_skip_counter % (self.config.skip_frames + 1) == 0:
                        continue   # skip this frame entirely
                
                self.frame_count += 1
                
                # 1. Resize frame
                frame = self._resize_frame(frame)
                
                # 2. Assess quality (only every N frames)
                if self.frame_count % self.config.quality_assess_interval == 0:
                    quality = self._assess_quality(frame)
                    self.last_quality = quality
                    # Update quality history
                    self.quality_history.append(quality)
                    if len(self.quality_history) > 100:
                        self.quality_history.pop(0)
                else:
                    # Reuse last quality (or a default if None)
                    if self.last_quality is None:
                        # Compute once if we haven't yet
                        quality = self._assess_quality(frame)
                        self.last_quality = quality
                        self.quality_history.append(quality)
                    else:
                        # Update frame number and timestamp for consistency
                        quality = self.last_quality
                        # We can optionally create a shallow copy to update frame_number
                        # but for simplicity we reuse the object; it's acceptable.
                
                # 3. Enhance frame (if enabled)
                if self.config.enable_enhancement:
                    # Optional: skip enhancement if quality is already GOOD or better
                    if self.config.enhance_only_if_poor and quality.overall_quality in (CameraQuality.GOOD, CameraQuality.EXCELLENT):
                        enhanced_frame = frame
                        report = EnhancementReport(
                            original_dims=frame.shape[:2],
                            enhanced_dims=frame.shape[:2]
                        )
                    else:
                        enhanced_frame, report = self._enhance_frame(frame)
                else:
                    enhanced_frame = frame
                    report = EnhancementReport(
                        original_dims=frame.shape[:2],
                        enhanced_dims=frame.shape[:2]
                    )
                
                # 4. Queue frame (latest-wins)
                self._queue_frame(enhanced_frame)
                
                # 5. Queue quality metrics
                self._queue_quality(quality)
                
                # 6. FPS tracking
                fps_counter += 1
                if time.time() - fps_timer >= 1.0:
                    self.current_fps = fps_counter / (time.time() - fps_timer)
                    fps_counter = 0
                    fps_timer = time.time()
                    self.fps_samples.append(self.current_fps)
                    
                    # Keep last 60 samples
                    if len(self.fps_samples) > 60:
                        self.fps_samples.pop(0)
                
                # 7. Update timing
                self.last_frame_time = time.time()
                
            except Exception as e:
                logger.error(f"Camera loop error: {e}")
                self.is_camera_online = False
                
                if cap:
                    cap.release()
                    cap = None
                
                time.sleep(self.config.retry_interval)
        
        # Cleanup
        if cap:
            cap.release()
        
        logger.info("Camera loop stopped")
    
    # ========================================================================
    # Camera Operations
    # ========================================================================
    
    def _open_camera(self) -> Optional[cv2.VideoCapture]:
        """
        Open the camera with configured settings.
        
        Returns:
            cv2.VideoCapture or None if opening fails
        """
        try:
            # Convert source to int if numeric string
            source = self.config.source
            if source.isdigit():
                source = int(source)
            
            cap = cv2.VideoCapture(source)
            
            if not cap.isOpened():
                logger.error(f"Failed to open camera: {self.config.source}")
                return None
            
            # Set properties
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
            cap.set(cv2.CAP_PROP_FPS, self.config.fps)
            
            # Verify actual resolution
            actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            logger.info(f"Camera opened: {actual_w}x{actual_h} (requested: {self.config.width}x{self.config.height})")
            
            return cap
            
        except Exception as e:
            logger.error(f"Camera open error: {e}")
            return None
    
    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Resize frame to configured dimensions.
        
        Args:
            frame: Input frame
            
        Returns:
            Resized frame
        """
        if frame.shape[:2] != (self.config.height, self.config.width):
            return cv2.resize(
                frame,
                (self.config.width, self.config.height),
                interpolation=cv2.INTER_AREA
            )
        return frame
    
    # ========================================================================
    # Quality Assessment
    # ========================================================================
    
    def _assess_quality(self, frame: np.ndarray) -> QualityMetrics:
        """
        Comprehensive image quality assessment.
        
        Evaluates:
            - Blur (Variance of Laplacian)
            - Brightness (Mean intensity)
            - Contrast (Std deviation)
            - Noise (Median absolute deviation)
            - Exposure (Histogram analysis)
        
        Args:
            frame: Input frame (BGR)
            
        Returns:
            QualityMetrics with all assessments
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 1. Blur detection (Variance of Laplacian)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        if blur_score > 200:
            blur_status = "Sharp"
        elif blur_score > 100:
            blur_status = "Moderate"
        else:
            blur_status = "Blurry"
        
        # 2. Brightness
        brightness_mean = float(gray.mean())
        if brightness_mean < 30:
            brightness_status = "Too Dark"
        elif brightness_mean < 80:
            brightness_status = "Dark"
        elif brightness_mean < 180:
            brightness_status = "Normal"
        elif brightness_mean < 220:
            brightness_status = "Bright"
        else:
            brightness_status = "Too Bright"
        
        # 3. Contrast
        contrast_std = float(gray.std())
        if contrast_std < 30:
            contrast_status = "Low"
        elif contrast_std < 80:
            contrast_status = "Normal"
        else:
            contrast_status = "High"
        
        # 4. Noise estimation (median absolute deviation)
        noise_level = self._estimate_noise(gray)
        if noise_level < 20:
            noise_status = "Low"
        elif noise_level < 50:
            noise_status = "Moderate"
        else:
            noise_status = "High"
        
        # 5. Exposure (histogram analysis)
        exposure_score = self._calculate_exposure(gray)
        if exposure_score < 20:
            exposure_status = "Under"
        elif exposure_score > 80:
            exposure_status = "Over"
        else:
            exposure_status = "Normal"
        
        # 6. Overall score (weighted combination)
        overall_score = self._calculate_overall_score(
            blur_score=blur_score,
            brightness=brightness_mean,
            contrast=contrast_std,
            noise=noise_level,
            exposure=exposure_score
        )
        
        # 7. Overall quality classification
        overall_quality = self._classify_quality(overall_score)
        
        return QualityMetrics(
            blur_score=blur_score,
            blur_status=blur_status,
            brightness_mean=brightness_mean,
            brightness_status=brightness_status,
            contrast_std=contrast_std,
            contrast_status=contrast_status,
            noise_level=noise_level,
            noise_status=noise_status,
            exposure_score=exposure_score,
            exposure_status=exposure_status,
            overall_score=overall_score,
            overall_quality=overall_quality,
            frame_number=self.frame_count,
            timestamp=datetime.now()
        )
    
    def _estimate_noise(self, gray: np.ndarray) -> float:
        """
        Estimate noise level using median absolute deviation.
        
        Args:
            gray: Grayscale image
            
        Returns:
            Noise estimate
        """
        # Simple noise estimation using Laplacian
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        # Median absolute deviation
        mad = np.median(np.abs(laplacian - np.median(laplacian)))
        return float(mad)
    
    def _calculate_exposure(self, gray: np.ndarray) -> float:
        """
        Calculate exposure score from histogram.
        
        Args:
            gray: Grayscale image
            
        Returns:
            Exposure score (0-100)
        """
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist / hist.sum()  # Normalize
        
        # Weighted exposure: ideally histogram is spread across range
        # This is a simplified version
        cumulative = np.cumsum(hist)
        
        # Check for clipping at dark and bright ends
        dark_clip = cumulative[10]  # Percent of pixels below 10
        bright_clip = 1 - cumulative[245]  # Percent of pixels above 245
        
        exposure_score = 100 - (dark_clip * 100 + bright_clip * 100) / 2
        
        return max(0, min(100, exposure_score))
    
    def _calculate_overall_score(
        self,
        blur_score: float,
        brightness: float,
        contrast: float,
        noise: float,
        exposure: float
    ) -> float:
        """
        Calculate overall quality score from individual metrics.
        
        Args:
            blur_score: Variance of Laplacian
            brightness: Mean intensity
            contrast: Std deviation
            noise: Noise estimate
            exposure: Exposure score
            
        Returns:
            Overall score (0-100)
        """
        # Normalize each metric to 0-100
        # Blur: 0-500 range, higher is better
        blur_norm = min(100, (blur_score / 500) * 100)
        
        # Brightness: optimal around 128
        brightness_norm = 100 - min(100, abs(brightness - 128) * 1.5)
        
        # Contrast: optimal around 60-80
        contrast_norm = min(100, (contrast / 80) * 100)
        contrast_norm = min(100, max(0, contrast_norm))
        
        # Noise: lower is better, 0-100 range
        noise_norm = max(0, 100 - (noise / 2))
        
        # Exposure: already 0-100
        exposure_norm = exposure
        
        # Weighted average
        weights = {
            'blur': 0.25,
            'brightness': 0.20,
            'contrast': 0.20,
            'noise': 0.20,
            'exposure': 0.15
        }
        
        score = (
            blur_norm * weights['blur'] +
            brightness_norm * weights['brightness'] +
            contrast_norm * weights['contrast'] +
            noise_norm * weights['noise'] +
            exposure_norm * weights['exposure']
        )
        
        return round(score, 1)
    
    def _classify_quality(self, score: float) -> CameraQuality:
        """
        Classify overall quality based on score.
        
        Args:
            score: Quality score (0-100)
            
        Returns:
            CameraQuality enum
        """
        if score >= 90:
            return CameraQuality.EXCELLENT
        elif score >= 70:
            return CameraQuality.GOOD
        elif score >= 50:
            return CameraQuality.FAIR
        elif score >= 30:
            return CameraQuality.POOR
        elif score >= 10:
            return CameraQuality.VERY_POOR
        else:
            return CameraQuality.OFFLINE
    
    # ========================================================================
    # Image Enhancement
    # ========================================================================
    
    def _enhance_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, EnhancementReport]:
        """
        Apply image enhancement operations to a frame.
        
        Enhancement operations:
            1. CLAHE: Contrast Limited Adaptive Histogram Equalization
            2. Gamma Correction: Adjust brightness
            3. Sharpening: Enhance edges
            4. Denoising: Reduce noise
            
        Args:
            frame: Input frame (BGR)
            
        Returns:
            Tuple of (enhanced frame, enhancement report)
        """
        if not self.config.enable_enhancement:
            return frame, EnhancementReport(
                original_dims=frame.shape[:2],
                enhanced_dims=frame.shape[:2]
            )
        
        start_time = time.time()
        enhanced = frame.copy()
        operations = []
        
        # 1. Denoise first (before other operations)
        if self.config.use_denoise:
            enhanced = cv2.fastNlMeansDenoisingColored(
                enhanced,
                None,
                self.config.denoise_strength,
                self.config.denoise_strength,
                7,
                21
            )
            operations.append(EnhancementType.DENOISE)
        
        # Convert to LAB for CLAHE (applies to L channel only)
        lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # 2. CLAHE (Contrast enhancement)
        if self.config.use_clahe:
            clahe = cv2.createCLAHE(
                clipLimit=self.config.clahe_clip_limit,
                tileGridSize=(self.config.clahe_grid_size, self.config.clahe_grid_size)
            )
            l = clahe.apply(l)
            operations.append(EnhancementType.CLAHE)
        
        # Merge back
        lab = cv2.merge((l, a, b))
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # 3. Gamma Correction
        if self.config.use_gamma:
            enhanced = self._apply_gamma(enhanced, self.config.gamma_value)
            operations.append(EnhancementType.GAMMA)
        
        # 4. Sharpening
        if self.config.use_sharpen:
            kernel = np.array([
                [-1, -1, -1],
                [-1, 9, -1],
                [-1, -1, -1]
            ]) * (1 + self.config.sharpen_strength)
            
            # Apply unsharp mask
            gaussian = cv2.GaussianBlur(enhanced, (0, 0), 2.0)
            enhanced = cv2.addWeighted(
                enhanced,
                1.0 + self.config.sharpen_strength,
                gaussian,
                -self.config.sharpen_strength,
                0
            )
            operations.append(EnhancementType.SHARPEN)
        
        # 5. Contrast enhancement (if needed)
        # Clip values to valid range
        enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
        
        processing_time = (time.time() - start_time) * 1000
        
        report = EnhancementReport(
            operations=operations,
            processing_time_ms=processing_time,
            original_dims=frame.shape[:2],
            enhanced_dims=enhanced.shape[:2]
        )
        
        return enhanced, report
    
    def _apply_gamma(self, frame: np.ndarray, gamma: float) -> np.ndarray:
        """
        Apply gamma correction to a frame.
        
        Args:
            frame: Input frame (BGR)
            gamma: Gamma value (0.1-2.0)
            
        Returns:
            Gamma-corrected frame
        """
        inv_gamma = 1.0 / gamma
        table = np.array([
            ((i / 255.0) ** inv_gamma) * 255
            for i in range(256)
        ]).astype(np.uint8)
        
        return cv2.LUT(frame, table)
    
    # ========================================================================
    # Queue Management
    # ========================================================================
    
    def _queue_frame(self, frame: np.ndarray) -> None:
        """
        Queue a frame with latest-wins policy.
        
        If the queue is full, removes the oldest frame and adds the new one.
        
        Args:
            frame: Frame to queue
        """
        try:
            # Try to put without blocking
            self.frame_queue.put(frame, block=False)
        except:
            # Queue is full, remove oldest and add new
            try:
                self.frame_queue.get_nowait()
                self.frame_queue.put(frame, block=False)
            except:
                # If this fails, just log it
                logger.debug("Failed to queue frame")
    
    def _queue_quality(self, quality: QualityMetrics) -> None:
        """
        Queue quality metrics with latest-wins policy.
        
        Args:
            quality: QualityMetrics to queue
        """
        try:
            self.quality_queue.put(quality, block=False)
        except:
            try:
                self.quality_queue.get_nowait()
                self.quality_queue.put(quality, block=False)
            except:
                pass
    
    # ========================================================================
    # Utility Methods
    # ========================================================================
    
    def get_average_fps(self) -> float:
        """Get the average FPS over the last 60 samples."""
        if not self.fps_samples:
            return 0.0
        return sum(self.fps_samples) / len(self.fps_samples)
    
    def get_quality_history(self, limit: int = 100) -> List[QualityMetrics]:
        """Get recent quality history."""
        return self.quality_history[-limit:]
    
    def reset(self) -> None:
        """Reset the processor state."""
        self.frame_count = 0
        self.fps_samples.clear()
        self.current_fps = 0.0
        self.reconnect_count = 0
        self.quality_history.clear()
        self.last_quality = None
        logger.info("Camera processor reset")
    
    def __repr__(self) -> str:
        """String representation."""
        return (f"CameraProcessor(source='{self.config.source}', "
                f"online={self.is_camera_online}, "
                f"frames={self.frame_count}, "
                f"fps={self.current_fps:.1f})")