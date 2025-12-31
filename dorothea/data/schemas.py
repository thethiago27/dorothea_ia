"""
Data schemas for the Dorothea AI system.

This module contains dataclasses that define the structure of data
used throughout the system, from raw Spotify data to processed datasets.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, List
import numpy as np


@dataclass
class SpotifyTrackData:
    """Raw data from Spotify CSV"""
    id: str
    name: str
    album: str
    release_date: str
    track_number: int
    uri: str
    acousticness: float
    danceability: float
    energy: float
    instrumentalness: float
    liveness: float
    loudness: float
    speechiness: float
    tempo: float
    valence: float
    popularity: int
    duration_ms: int


@dataclass
class MoodVector:
    """Mood vector representation for tracks"""
    valence: float      # Happiness/sadness
    energy: float       # Intensity
    danceability: float # Rhythm suitability
    acousticness: float # Acoustic vs electronic
    tempo_normalized: float  # Normalized tempo
    
    def to_array(self) -> np.ndarray:
        """Convert mood vector to numpy array"""
        return np.array([
            self.valence,
            self.energy,
            self.danceability,
            self.acousticness,
            self.tempo_normalized
        ])
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'MoodVector':
        """Create mood vector from numpy array"""
        if len(arr) != 5:
            raise ValueError(f"Expected array of length 5, got {len(arr)}")
        return cls(
            valence=float(arr[0]),
            energy=float(arr[1]),
            danceability=float(arr[2]),
            acousticness=float(arr[3]),
            tempo_normalized=float(arr[4])
        )


@dataclass
class ProcessedTrack:
    """Processed track with mood classification"""
    id: str
    name: str
    album: str
    uri: str
    mood_vector: MoodVector
    mood_label: str
    raw_features: Dict[str, float]


@dataclass
class ProcessedDataset:
    """Dataset after processing with normalization parameters"""
    features: np.ndarray
    labels: np.ndarray
    track_ids: List[str]
    scaler_params: Dict[str, np.ndarray]
    metadata: Dict[str, Any]


@dataclass
class ValidationResult:
    """Result of data validation operations"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    metadata: Optional[Dict[str, Any]] = None
    
    def add_error(self, error: str) -> None:
        """Add an error to the validation result"""
        self.errors.append(error)
        self.is_valid = False
    
    def add_warning(self, warning: str) -> None:
        """Add a warning to the validation result"""
        self.warnings.append(warning)
    
    def has_errors(self) -> bool:
        """Check if validation has any errors"""
        return len(self.errors) > 0
    
    def has_warnings(self) -> bool:
        """Check if validation has any warnings"""
        return len(self.warnings) > 0