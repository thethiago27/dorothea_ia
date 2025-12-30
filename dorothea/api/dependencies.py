"""
FastAPI dependency injection for Dorothea AI.
"""
import os
from functools import lru_cache
from typing import Optional
import logging

from ..config.settings import ConfigManager, AppConfig
from ..utils.logging import StructuredLogger
from ..models.trainer import ModelExporter
from ..persistence.track_mapper import TrackMapper
from ..recommendation.engine import RecommendationEngine
from ..recommendation.similarity import SimilarityCalculator


logger = logging.getLogger(__name__)


class AppState:
    """Singleton state for the Dorothea AI application."""
    
    _instance: Optional["AppState"] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self.config_manager = ConfigManager()
        self.config: Optional[AppConfig] = None
        self.logger: Optional[StructuredLogger] = None
        self.recommendation_engine: Optional[RecommendationEngine] = None
        self.model_loaded: bool = False
        self._initialized = True
    
    def initialize(self, config_path: str = "dorothea/config/default_config.yaml") -> None:
        """Initialize the application state."""
        if self.config is not None:
            return
            
        try:
            self.config = self.config_manager.load(config_path)
            self.logger = StructuredLogger(
                "dorothea.api",
                level=self.config.logging.level
            )
            self.logger.info("Dorothea AI API initialized")
            
            # Try to load recommendation engine
            self._load_recommendation_engine()
            
        except Exception as e:
            logger.error(f"Failed to initialize Dorothea AI: {e}")
            raise
    
    def _load_recommendation_engine(self) -> None:
        """Load the recommendation engine if model exists."""
        try:
            model_path = os.path.join(
                self.config.data.output_path, 
                "model_latest.h5"
            )
            
            if not os.path.exists(model_path):
                self.logger.warning(
                    "Model not found. Run 'python main.py train' first.",
                    path=model_path
                )
                self.model_loaded = False
                return
            
            # Load model
            exporter = ModelExporter(self.logger)
            model = exporter.load_model(model_path)
            
            # Load track mapper
            track_mapper_path = os.path.join(
                self.config.data.output_path, 
                "track_mapping.json"
            )
            track_mapper = TrackMapper(track_mapper_path)
            
            # Create recommendation engine
            similarity_calculator = SimilarityCalculator()
            self.recommendation_engine = RecommendationEngine(
                track_mapper, 
                similarity_calculator
            )
            
            self.model_loaded = True
            self.logger.info("Recommendation engine loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load recommendation engine: {e}")
            self.model_loaded = False


@lru_cache()
def get_app_state() -> AppState:
    """Get or create the application state singleton."""
    state = AppState()
    state.initialize()
    return state


def get_recommendation_engine() -> RecommendationEngine:
    """Dependency for getting the recommendation engine."""
    state = get_app_state()
    if state.recommendation_engine is None:
        raise RuntimeError(
            "Recommendation engine not loaded. "
            "Please train the model first with: python main.py train"
        )
    return state.recommendation_engine


def get_config() -> AppConfig:
    """Dependency for getting the app configuration."""
    state = get_app_state()
    return state.config
