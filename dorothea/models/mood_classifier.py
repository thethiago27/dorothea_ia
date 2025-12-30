"""
Mood classification module for Dorothea AI.
Provides MoodClassifier class for categorizing tracks by mood.
"""
from enum import Enum
from dataclasses import dataclass
from typing import Dict, Any, Optional
import numpy as np
import logging
from ..data.schemas import MoodVector
class MoodLevel(Enum):
    VERY_LOW = 0
    LOW = 1
    NEUTRAL = 2
    HIGH = 3
    VERY_HIGH = 4
@dataclass
class MoodClassification:
    """Result of mood classification containing mood vector and metadata."""
    mood_vector: MoodVector
    primary_mood: str
    confidence: float
    raw_features: Dict[str, float]
class MoodClassifier:
    """Classifies tracks into mood categories based on audio features."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the mood classifier with configuration.
        
        Args:
            config: Configuration dictionary with thresholds and weights
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        self.thresholds = self.config.get('mood_thresholds', {
            'low': 0.3,
            'neutral': 0.5,
            'high': 0.7
        })
        self.feature_weights = self.config.get('feature_weights', {
            'valence': 0.3,
            'energy': 0.25,
            'danceability': 0.2,
            'acousticness': 0.15,
            'tempo_normalized': 0.1
        })
        self.valid_ranges = {
            'valence': (0.0, 1.0),
            'energy': (0.0, 1.0),
            'danceability': (0.0, 1.0),
            'acousticness': (0.0, 1.0),
            'tempo': (0.0, 250.0)
        }
    def classify(self, features: Dict[str, float]) -> MoodClassification:
        """Classify a track's mood based on its audio features.
        
        Args:
            features: Dictionary of audio features (valence, energy, etc.)
            
        Returns:
            MoodClassification object with mood vector and metadata
        """
        normalized_features = self._normalize_features(features)
        self._check_extreme_values(features)
        mood_vector = MoodVector(
            valence=normalized_features['valence'],
            energy=normalized_features['energy'],
            danceability=normalized_features['danceability'],
            acousticness=normalized_features['acousticness'],
            tempo_normalized=normalized_features['tempo_normalized']
        )
        weighted_score = self.compute_weighted_score(normalized_features)
        mood_label = self.get_mood_label(mood_vector)
        confidence = self._compute_confidence(normalized_features, weighted_score)
        return MoodClassification(
            mood_vector=mood_vector,
            primary_mood=mood_label,
            confidence=confidence,
            raw_features=features.copy()
        )
    def get_mood_label(self, mood_vector: MoodVector) -> str:
        """Get a descriptive mood label based on the mood vector.
        
        Args:
            mood_vector: MoodVector containing normalized feature values
            
        Returns:
            String mood label (e.g., "melancholic", "energetic", "calm")
        """
        valence = mood_vector.valence
        energy = mood_vector.energy
        danceability = mood_vector.danceability
        acousticness = mood_vector.acousticness
        if valence < self.thresholds['low'] and energy < self.thresholds['low']:
            if acousticness > self.thresholds['high']:
                return "melancholic_acoustic"
            else:
                return "melancholic"
        elif valence < self.thresholds['low'] and energy > self.thresholds['high']:
            return "intense_sad"
        elif valence > self.thresholds['high'] and energy > self.thresholds['high']:
            if danceability > self.thresholds['high']:
                return "euphoric_dance"
            else:
                return "energetic_happy"
        elif valence > self.thresholds['high'] and energy < self.thresholds['low']:
            if acousticness > self.thresholds['high']:
                return "peaceful_acoustic"
            else:
                return "calm_happy"
        elif valence < self.thresholds['neutral'] and energy > self.thresholds['neutral']:
            return "aggressive"
        elif valence > self.thresholds['neutral'] and energy < self.thresholds['neutral']:
            return "serene"
        else:
            if danceability > self.thresholds['high']:
                return "neutral_dance"
            elif acousticness > self.thresholds['high']:
                return "neutral_acoustic"
            else:
                return "neutral"
    def compute_weighted_score(self, features: Dict[str, float]) -> float:
        """Compute a weighted score from normalized features.
        
        Args:
            features: Dictionary of normalized feature values
            
        Returns:
            Weighted score in range [0, 1]
        """
        score = 0.0
        total_weight = 0.0
        for feature, weight in self.feature_weights.items():
            if feature in features:
                score += features[feature] * weight
                total_weight += weight
        if total_weight > 0:
            score = score / total_weight
        return max(0.0, min(1.0, score))
    def _normalize_features(self, features: Dict[str, float]) -> Dict[str, float]:
        """Normalize features to [0, 1] range.
        
        Args:
            features: Raw feature values
            
        Returns:
            Dictionary of normalized features
        """
        normalized = {}
        for feature in ['valence', 'energy', 'danceability', 'acousticness']:
            if feature in features:
                normalized[feature] = max(0.0, min(1.0, features[feature]))
        if 'tempo' in features:
            tempo = features['tempo']
            normalized['tempo_normalized'] = max(0.0, min(1.0, tempo / 250.0))
        return normalized
    def _check_extreme_values(self, features: Dict[str, float]) -> None:
        """Check for extreme values outside expected ranges.
        
        Args:
            features: Feature values to validate
        """
        for feature, value in features.items():
            if feature in self.valid_ranges:
                min_val, max_val = self.valid_ranges[feature]
                if value < min_val or value > max_val:
                    self.logger.warning(
                        f"Extreme value detected for {feature}: {value} "
                        f"(expected range: {min_val}-{max_val})"
                    )
    def _compute_confidence(self, features: Dict[str, float], weighted_score: float) -> float:
        """Compute confidence score for the mood classification.
        
        Args:
            features: Normalized feature values
            weighted_score: Computed weighted score
            
        Returns:
            Confidence score in range [0, 1]
        """
        distance_from_neutral = abs(weighted_score - 0.5)
        base_confidence = distance_from_neutral * 2
        if 'valence' in features and 'energy' in features:
            valence = features['valence']
            energy = features['energy']
            if (valence > 0.5 and energy > 0.5) or (valence < 0.5 and energy < 0.5):
                base_confidence = min(1.0, base_confidence * 1.2)
        return max(0.0, min(1.0, base_confidence))