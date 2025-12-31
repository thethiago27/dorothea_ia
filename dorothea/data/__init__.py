"""
Data processing module for Dorothea AI.

This module provides data loading, validation, and processing capabilities
for Spotify track data used in the mood-based recommendation system.
"""

from .schemas import (
    SpotifyTrackData,
    ProcessedTrack,
    ProcessedDataset,
    ValidationResult,
    MoodVector
)
from .validator import DataValidator
from .processor import DataProcessor

__all__ = [
    'SpotifyTrackData',
    'ProcessedTrack', 
    'ProcessedDataset',
    'ValidationResult',
    'MoodVector',
    'DataValidator',
    'DataProcessor'
]