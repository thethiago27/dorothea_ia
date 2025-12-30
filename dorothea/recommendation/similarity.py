"""
Similarity calculation module for Dorothea AI.
"""
import numpy as np
from typing import List, Tuple
import logging
class SimilarityCalculator:
    """Calculates similarity scores between mood vectors."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors.
        
        Args:
            v1: First vector (numpy array)
            v2: Second vector (numpy array)
            
        Returns:
            Cosine similarity score in range [-1, 1], where 1 is identical
            
        Raises:
            ValueError: If vectors have different shapes or are not 1D
        """
        if v1.shape != v2.shape:
            raise ValueError(f"Vector dimensions must match: {v1.shape} vs {v2.shape}")
        if len(v1.shape) != 1:
            raise ValueError("Vectors must be 1-dimensional")
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            if norm1 == 0 and norm2 == 0:
                return 1.0
            else:
                return 0.0
        dot_product = np.dot(v1, v2)
        similarity = dot_product / (norm1 * norm2)
        return float(np.clip(similarity, -1.0, 1.0))
    def euclidean_distance(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Calculate Euclidean distance between two vectors.
        
        Args:
            v1: First vector (numpy array)
            v2: Second vector (numpy array)
            
        Returns:
            Euclidean distance (non-negative float)
            
        Raises:
            ValueError: If vectors have different shapes
        """
        if v1.shape != v2.shape:
            raise ValueError(f"Vector dimensions must match: {v1.shape} vs {v2.shape}")
        distance = np.linalg.norm(v1 - v2)
        return float(distance)
    def compute_batch_similarity(self, query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between a query vector and multiple candidates.
        
        Args:
            query: Query vector of shape (n_features,)
            candidates: Matrix of candidate vectors of shape (n_candidates, n_features)
            
        Returns:
            Array of similarity scores of shape (n_candidates,)
            
        Raises:
            ValueError: If input shapes are invalid or incompatible
        """
        if len(query.shape) != 1:
            raise ValueError("Query must be a 1-dimensional vector")
        if len(candidates.shape) != 2:
            raise ValueError("Candidates must be a 2-dimensional matrix")
        if query.shape[0] != candidates.shape[1]:
            raise ValueError(
                f"Feature dimensions must match: query={query.shape[0]}, "
                f"candidates={candidates.shape[1]}"
            )
        if candidates.shape[0] == 0:
            return np.array([])
        query_norm = np.linalg.norm(query)
        if query_norm == 0:
            candidate_norms = np.linalg.norm(candidates, axis=1)
            similarities = np.where(candidate_norms == 0, 1.0, 0.0)
            return similarities
        candidate_norms = np.linalg.norm(candidates, axis=1)
        dot_products = np.dot(candidates, query)
        zero_candidates = (candidate_norms == 0)
        similarities = np.zeros(candidates.shape[0])
        non_zero_mask = ~zero_candidates
        if np.any(non_zero_mask):
            similarities[non_zero_mask] = (
                dot_products[non_zero_mask] / 
                (query_norm * candidate_norms[non_zero_mask])
            )
        similarities[zero_candidates] = 0.0
        similarities = np.clip(similarities, -1.0, 1.0)
        return similarities
    def similarity_to_distance(self, similarity: float) -> float:
        """Convert cosine similarity to distance metric.
        
        Args:
            similarity: Cosine similarity in range [-1, 1]
            
        Returns:
            Distance in range [0, 2] where 0 is identical and 2 is opposite
        """
        return 1.0 - similarity
    def rank_by_similarity(self, query: np.ndarray, candidates: np.ndarray, 
                          descending: bool = True) -> List[Tuple[int, float]]:
        """Rank candidates by similarity to query vector.
        
        Args:
            query: Query vector
            candidates: Matrix of candidate vectors
            descending: If True, rank from highest to lowest similarity
            
        Returns:
            List of (index, similarity_score) tuples sorted by similarity
        """
        similarities = self.compute_batch_similarity(query, candidates)
        indexed_similarities = list(enumerate(similarities))
        indexed_similarities.sort(key=lambda x: x[1], reverse=descending)
        return indexed_similarities