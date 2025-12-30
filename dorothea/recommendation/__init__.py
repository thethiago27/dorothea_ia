"""
Recommendation module for Dorothea AI.

This module provides mood-based song recommendation functionality,
including similarity calculation, filtering, and ranking.
"""

from .engine import RecommendationEngine
from .similarity import SimilarityCalculator
from .schemas import RecommendationRequest, RecommendationResponse, SongRecommendation

__all__ = [
    'RecommendationEngine',
    'SimilarityCalculator', 
    'RecommendationRequest',
    'RecommendationResponse',
    'SongRecommendation'
]