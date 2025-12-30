"""
Recommendation engine for Dorothea AI.
"""
import time
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
from ..data.schemas import MoodVector
from ..persistence.track_mapper import TrackMapper, TrackInfo
from .similarity import SimilarityCalculator
from .schemas import RecommendationRequest, RecommendationResponse, SongRecommendation
class RecommendationEngine:
    """Main recommendation engine that orchestrates mood-based song recommendations."""
    
    def __init__(self, track_mapper: TrackMapper, similarity_calculator: Optional[SimilarityCalculator] = None):
        """Initialize the recommendation engine.
        
        Args:
            track_mapper: TrackMapper for accessing track data
            similarity_calculator: Calculator for similarity scores (optional)
        """
        self.track_mapper = track_mapper
        self.similarity_calculator = similarity_calculator or SimilarityCalculator()
        self.logger = logging.getLogger(__name__)
        self.min_similarity_threshold = 0.1
        self.fallback_limit = 5
    def recommend(self, request: RecommendationRequest) -> RecommendationResponse:
        """Generate song recommendations based on mood input and filters.
        
        Args:
            request: RecommendationRequest with mood input and filters
            
        Returns:
            RecommendationResponse with ranked song recommendations
            
        Raises:
            RuntimeError: If recommendation generation fails
        """
        start_time = time.time()
        try:
            track_mapping = self.track_mapper.load_mapping()
            if not track_mapping:
                self.logger.warning("No tracks available for recommendations")
                return RecommendationResponse(
                    recommendations=[],
                    query_mood=request.mood_input,
                    processing_time_ms=0.0,
                    total_candidates=0
                )
            candidates = []
            for track_info in track_mapping.values():
                if track_info.mood_vector is not None:
                    candidates.append(track_info)
            if not candidates:
                self.logger.warning("No tracks with mood vectors available")
                return RecommendationResponse(
                    recommendations=[],
                    query_mood=request.mood_input,
                    processing_time_ms=(time.time() - start_time) * 1000,
                    total_candidates=0
                )
            if request.filters:
                candidates = self.apply_filters(candidates, request.filters)
            total_candidates = len(candidates)
            if not candidates:
                self.logger.info("No candidates remain after filtering")
                return RecommendationResponse(
                    recommendations=[],
                    query_mood=request.mood_input,
                    processing_time_ms=(time.time() - start_time) * 1000,
                    total_candidates=0
                )
            ranked_recommendations = self.rank_by_similarity(request.mood_input, candidates)
            limited_recommendations = ranked_recommendations[:request.limit]
            if not limited_recommendations or (
                limited_recommendations and 
                limited_recommendations[0].similarity_score < self.min_similarity_threshold
            ):
                self.logger.info(
                    f"Low similarity scores detected (best: "
                    f"{limited_recommendations[0].similarity_score if limited_recommendations else 'N/A'}), "
                    f"using fallback recommendations"
                )
            processing_time = (time.time() - start_time) * 1000
            return RecommendationResponse(
                recommendations=limited_recommendations,
                query_mood=request.mood_input,
                processing_time_ms=processing_time,
                total_candidates=total_candidates
            )
        except Exception as e:
            error_msg = f"Failed to generate recommendations: {e}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    def apply_filters(self, candidates: List[TrackInfo], filters: Dict[str, Any]) -> List[TrackInfo]:
        """Apply filters to candidate tracks.
        
        Args:
            candidates: List of candidate tracks
            filters: Dictionary of filters to apply
            
        Returns:
            Filtered list of tracks
        """
        filtered = candidates.copy()
        if 'album' in filters:
            album_filter = filters['album']
            if isinstance(album_filter, str):
                filtered = [t for t in filtered if t.album.lower() == album_filter.lower()]
            elif isinstance(album_filter, list):
                album_set = {a.lower() for a in album_filter}
                filtered = [t for t in filtered if t.album.lower() in album_set]
            else:
                self.logger.warning(f"Invalid album filter type: {type(album_filter)}")
        if 'era' in filters:
            era_filter = filters['era']
            era_albums = self._get_era_albums(era_filter)
            if era_albums:
                filtered = [t for t in filtered if t.album in era_albums]
            else:
                self.logger.warning(f"Unknown era: {era_filter}")
        if 'year_range' in filters:
            year_range = filters['year_range']
            if isinstance(year_range, dict) and 'start' in year_range and 'end' in year_range:
                self.logger.warning("Year range filtering not implemented (requires release_date in TrackInfo)")
        if 'exclude_albums' in filters:
            exclude_albums = filters['exclude_albums']
            if isinstance(exclude_albums, list):
                exclude_set = {a.lower() for a in exclude_albums}
                filtered = [t for t in filtered if t.album.lower() not in exclude_set]
        self.logger.debug(f"Applied filters: {len(candidates)} -> {len(filtered)} candidates")
        return filtered
    def rank_by_similarity(self, query_mood: MoodVector, candidates: List[TrackInfo]) -> List[SongRecommendation]:
        """Rank candidates by similarity to query mood.
        
        Args:
            query_mood: Target mood vector
            candidates: List of candidate tracks
            
        Returns:
            List of SongRecommendation objects sorted by similarity (descending)
        """
        if not candidates:
            return []
        query_array = query_mood.to_array()
        candidate_arrays = np.array([track.mood_vector.to_array() for track in candidates])
        similarities = self.similarity_calculator.compute_batch_similarity(query_array, candidate_arrays)
        recommendations = []
        for i, (track, similarity) in enumerate(zip(candidates, similarities)):
            confidence = self._compute_confidence(similarity, query_mood, track.mood_vector)
            recommendation = SongRecommendation(
                track_id=str(track.internal_id),
                track_name=track.name,
                album=track.album,
                uri=track.uri,
                similarity_score=float(similarity),
                mood_vector=track.mood_vector,
                confidence=confidence
            )
            recommendations.append(recommendation)
        recommendations.sort(key=lambda x: x.similarity_score, reverse=True)
        return recommendations
    def _get_era_albums(self, era: str) -> Optional[List[str]]:
        """Get list of albums for a given Taylor Swift era.
        
        Args:
            era: Era name (e.g., "fearless", "red", "1989")
            
        Returns:
            List of album names for the era, or None if era not found
        """
        era_mappings = {
            'debut': ['Taylor Swift'],
            'fearless': ['Fearless', 'Fearless (Taylor\'s Version)'],
            'speak_now': ['Speak Now', 'Speak Now (Taylor\'s Version)'],
            'red': ['Red', 'Red (Taylor\'s Version)'],
            '1989': ['1989', '1989 (Taylor\'s Version)'],
            'reputation': ['reputation'],
            'lover': ['Lover'],
            'folklore': ['folklore'],
            'evermore': ['evermore'],
            'midnights': ['Midnights'],
            'tortured_poets': ['The Tortured Poets Department']
        }
        era_lower = era.lower().replace(' ', '_')
        return era_mappings.get(era_lower)
    def _compute_confidence(self, similarity_score: float, query_mood: MoodVector, 
                           track_mood: MoodVector) -> float:
        """Compute confidence score for a recommendation.
        
        Args:
            similarity_score: Cosine similarity score
            query_mood: User's target mood vector
            track_mood: Track's mood vector
            
        Returns:
            Confidence score in range [0, 1]
        """
        base_confidence = (similarity_score + 1.0) / 2.0
        if similarity_score > 0.8:
            base_confidence = min(1.0, base_confidence * 1.1)
        elif similarity_score > 0.6:
            base_confidence = min(1.0, base_confidence * 1.05)
        if similarity_score < 0.2:
            base_confidence *= 0.8
        return max(0.0, min(1.0, base_confidence))