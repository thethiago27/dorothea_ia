"""
Persistence layer module for Dorothea AI.

Handles track mapping, model storage, and data persistence operations.
"""

from .track_mapper import TrackInfo, TrackMapper

__all__ = ['TrackInfo', 'TrackMapper']