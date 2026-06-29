"""
counter.py - Track-Based Crowd Counter Module

This module maintains an accurate estimate of the number of unique people
currently present in the monitored scene using stable tracked identities.

It operates on tracks (not raw detections) and handles temporary occlusions
through track timeout management.

Architecture:
    List<Person> → Track Extraction → Track Update → Track Timeout → Count

Author: System Architect
Version: 2.2.0 (FINAL - Locked)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Set

from .models.person import Person


@dataclass
class TrackInfo:
    """
    Information about a single tracked person.
    
    Kept minimal - only what's needed for crowd counting.
    active_track_ids is the single source of truth for active status.
    
    Attributes:
        track_id: Unique track ID
        first_seen_frame: Frame number when first detected
        last_seen_frame: Frame number when last detected
    """
    track_id: int
    first_seen_frame: int
    last_seen_frame: int
    
    def update(self, frame_number: int) -> None:
        """Update the track with a new detection."""
        self.last_seen_frame = frame_number


@dataclass
class CounterState:
    """
    State of the counter at any point.
    
    This is the PRIMARY interface for other modules.
    All counter information is exposed through this object.
    
    Attributes:
        current_count: Number of active tracks
        peak_count: Maximum count observed so far
        average_count: Running average count
        total_unique: Total unique tracks ever seen
        active_ids: Set of currently active track IDs
        frame_number: Current frame number
        timestamp: When the state was computed
        change_detected: Whether count changed since last update
        last_change: Last count change (positive for increase, negative for decrease)
    """
    current_count: int = 0
    peak_count: int = 0
    average_count: float = 0.0
    total_unique: int = 0
    active_ids: Set[int] = field(default_factory=set)
    frame_number: int = 0
    timestamp: datetime = field(default_factory=datetime.now)
    change_detected: bool = False
    last_change: int = 0


class Counter:
    """
    Track-Based Crowd Counter Engine.
    
    Maintains accurate occupancy estimates using stable tracked identities.
    Handles temporary occlusions through configurable track timeout.
    
    Attributes:
        timeout_frames: Number of frames to keep a track active after last seen
        tracks: Dictionary of tracks (track_id → TrackInfo)
        active_track_ids: Set of currently active track IDs (single source of truth)
        state: Current counter state
        frame_number: Current frame number
    """
    
    def __init__(self, timeout_frames: int = 30) -> None:
        """
        Initialize the Track-Based Crowd Counter.
        
        Args:
            timeout_frames: Number of frames to keep a track active after last seen
        """
        self.timeout_frames = timeout_frames
        
        # Track management
        self.tracks: Dict[int, TrackInfo] = {}
        self.active_track_ids: Set[int] = set()  # Single source of truth
        
        # Counter state
        self.frame_number = 0
        self.state = CounterState()
        
        # Statistics
        self.peak_count = 0
        self.total_unique = 0
        self.cumulative_count = 0
        self.observed_frames = 0
    
    @property
    def current_count(self) -> int:
        """
        Get the current active track count.
        
        Returns:
            Number of active tracks
        """
        return len(self.active_track_ids)
    
    def update(self, people: List[Person]) -> CounterState:
        """
        Update counter with new detections from a frame.
        
        This is the MAIN ENTRY POINT of the counter.
        
        Algorithm:
            1. Extract track IDs from people
            2. Update existing tracks with new detections
            3. Add new tracks for unseen IDs
            4. Remove tracks that have timed out
            5. Compute current count
            6. Update statistics
        
        Args:
            people: List of Person objects from the detector
            
        Returns:
            CounterState containing current count and statistics
        """
        self.frame_number += 1
        
        # Step 1: Extract track IDs
        current_ids = {person.track_id for person in people}
        
        # Step 2: Update existing tracks
        self._update_tracks(current_ids)
        
        # Step 3: Add new tracks
        self._add_new_tracks(current_ids)
        
        # Step 4: Remove expired tracks
        self._remove_expired_tracks()
        
        # Step 5: Compute current count
        current_count = self.current_count
        
        # Step 6: Update statistics
        self._update_statistics(current_count)
        
        # Step 7: Build and return state
        self.state = self._build_state(current_count)
        
        return self.state
    
    def _update_tracks(self, current_ids: Set[int]) -> None:
        """
        Update existing tracks with new detections.
        
        Args:
            current_ids: Set of track IDs in current frame
        """
        for track_id in current_ids:
            if track_id in self.tracks:
                track = self.tracks[track_id]
                track.update(self.frame_number)
                self.active_track_ids.add(track_id)
    
    def _add_new_tracks(self, current_ids: Set[int]) -> None:
        """
        Add new tracks for IDs not seen before.
        
        Args:
            current_ids: Set of track IDs in current frame
        """
        for track_id in current_ids:
            if track_id not in self.tracks:
                # Create new track
                track = TrackInfo(
                    track_id=track_id,
                    first_seen_frame=self.frame_number,
                    last_seen_frame=self.frame_number
                )
                self.tracks[track_id] = track
                self.active_track_ids.add(track_id)
                self.total_unique += 1
    
    def _remove_expired_tracks(self) -> None:
        """
        Remove tracks that have exceeded the timeout.
        
        A track is considered expired if it hasn't been seen for
        `timeout_frames` frames.
        
        Using a separate list for expired tracks improves readability
        by separating detection from removal.
        """
        expired_tracks = []
        
        # Step 1: Detect expired tracks
        for track_id in self.active_track_ids:
            track = self.tracks.get(track_id)
            if track:
                frames_missing = self.frame_number - track.last_seen_frame
                if frames_missing > self.timeout_frames:
                    expired_tracks.append(track_id)
        
        # Step 2: Remove expired tracks
        for track_id in expired_tracks:
            self.active_track_ids.remove(track_id)
    
    def _update_statistics(self, current_count: int) -> None:
        """
        Update counter statistics.
        
        Args:
            current_count: Current active track count
        """
        # Update peak count
        if current_count > self.peak_count:
            self.peak_count = current_count
        
        # Update running average
        self.observed_frames += 1
        self.cumulative_count += current_count
    
    def _build_state(self, current_count: int) -> CounterState:
        """
        Build the current counter state.
        
        Args:
            current_count: Current active track count
            
        Returns:
            CounterState object
        """
        # Calculate average
        avg_count = self.cumulative_count / self.observed_frames if self.observed_frames > 0 else 0.0
        
        # Check if count changed
        previous_count = self.state.current_count
        change_detected = current_count != previous_count
        last_change = current_count - previous_count if change_detected else 0
        
        return CounterState(
            current_count=current_count,
            peak_count=self.peak_count,
            average_count=avg_count,
            total_unique=self.total_unique,
            active_ids=self.active_track_ids.copy(),
            frame_number=self.frame_number,
            timestamp=datetime.now(),
            change_detected=change_detected,
            last_change=last_change
        )
    
    def reset(self) -> None:
        """
        Reset the counter state.
        
        Useful when switching cameras or restarting monitoring.
        """
        self.tracks.clear()
        self.active_track_ids.clear()
        self.frame_number = 0
        self.peak_count = 0
        self.total_unique = 0
        self.cumulative_count = 0
        self.observed_frames = 0
        self.state = CounterState()