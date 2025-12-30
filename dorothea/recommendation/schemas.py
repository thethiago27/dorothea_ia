"""
Recommendation schemas for the Dorothea AI system.
"""
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from ..data.schemas import MoodVector


@dataclass
class RecommendationRequest:
    """Request for song recommendations based on mood."""
    mood_input: MoodVector
    filters: Optional[Dict[str, Any]] = None
    limit: int = 10
    
    def __post_init__(self):
        if self.limit <= 0:
            raise ValueError("Limit must be positive")
        if self.limit > 100:
            raise ValueError("Limit cannot exceed 100")


@dataclass
class SongRecommendation:
    """A single song recommendation with similarity scores."""
    track_id: str
    track_name: str
    album: str
    uri: str
    similarity_score: float
    confidence: float
    mood_vector: Optional[MoodVector] = None
    
    def __post_init__(self):
        if not (0.0 <= self.similarity_score <= 1.0):
            raise ValueError("Similarity score must be between 0.0 and 1.0")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("Confidence must be between 0.0 and 1.0")


@dataclass
class RecommendationResponse:
    """Response containing song recommendations."""
    recommendations: List[SongRecommendation]
    query_mood: MoodVector
    processing_time_ms: float
    total_candidates: int
    
    def __post_init__(self):
        if self.processing_time_ms < 0:
            raise ValueError("Processing time cannot be negative")
        if self.total_candidates < 0:
            raise ValueError("Total candidates cannot be negative")
        if len(self.recommendations) > self.total_candidates:
            raise ValueError("Number of recommendations cannot exceed total candidates")