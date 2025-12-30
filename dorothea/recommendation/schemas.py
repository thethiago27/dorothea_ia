"""
Recommendation schemas for the Dorothea AI system.
"""
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from ..data.schemas import MoodVector
@dataclass
class RecommendationRequest:
    filters: Optional[Dict[str, Any]] = None
    limit: int = 10
    def __post_init__(self):
        if self.limit <= 0:
            raise ValueError("Limit must be positive")
        if self.limit > 100:
            raise ValueError("Limit cannot exceed 100")
@dataclass
class SongRecommendation:
    def __post_init__(self):
        if not (0.0 <= self.similarity_score <= 1.0):
            raise ValueError("Similarity score must be between 0.0 and 1.0")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("Confidence must be between 0.0 and 1.0")
@dataclass
class RecommendationResponse:
    def __post_init__(self):
        if self.processing_time_ms < 0:
            raise ValueError("Processing time cannot be negative")
        if self.total_candidates < 0:
            raise ValueError("Total candidates cannot be negative")
        if len(self.recommendations) > self.total_candidates:
            raise ValueError("Number of recommendations cannot exceed total candidates")