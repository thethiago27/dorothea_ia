"""
Configuration management for Dorothea AI.

Provides centralized configuration loading, validation, and environment variable overrides.
"""

import os
import yaml
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from pathlib import Path


@dataclass
class ModelConfig:
    """Configuration for machine learning model parameters."""
    hidden_layers: List[int] = field(default_factory=lambda: [128, 64, 32])
    dropout_rate: float = 0.3
    learning_rate: float = 0.001
    epochs: int = 150
    batch_size: int = 32
    early_stopping_patience: int = 15


@dataclass
class DataConfig:
    dataset_path: str = "data-train/taylor_swift_spotify.csv"
    output_path: str = "data/processed"
    album_blocklist: List[str] = field(default_factory=lambda: [
        "reputation Stadium Tour Surprise Song Playlist",
        "Speak Now World Tour Live",
        "Live From Clear Channel Stripped 2008"
    ])
    mood_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "very_low": 0.2,
        "low": 0.4,
        "neutral": 0.6,
        "high": 0.8
    })
    feature_columns: List[str] = field(default_factory=lambda: [
        "valence", "energy", "danceability", "acousticness", "tempo"
    ])


@dataclass
class RecommendationConfig:
    """Configuration for recommendation engine parameters."""
    default_limit: int = 10
    similarity_metric: str = "cosine"
    min_confidence: float = 0.5


@dataclass
class LoggingConfig:
    """Configuration for logging parameters."""
    level: str = "INFO"
    format: str = "json"


@dataclass
class VersioningConfig:
    """Configuration for model versioning."""
    model_version: str = "2.0.0"


@dataclass
class AppConfig:
    """Main application configuration containing all sub-configurations."""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    recommendation: RecommendationConfig = field(default_factory=RecommendationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    versioning: VersioningConfig = field(default_factory=VersioningConfig)


class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""
    pass


class ConfigManager:
    """Manages application configuration loading, validation, and environment overrides."""
    
    def __init__(self):
        self._config: Optional[AppConfig] = None
    
    def load(self, config_path: str) -> AppConfig:
        """
        Load configuration from YAML file with environment variable overrides.
        
        Args:
            config_path: Path to the YAML configuration file
            
        Returns:
            AppConfig: Loaded and validated configuration
            
        Raises:
            ConfigValidationError: If configuration is invalid
            FileNotFoundError: If config file doesn't exist
        """
        config_file = Path(config_path)
        
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        # Load YAML configuration
        with open(config_file, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f) or {}
        
        # Apply environment variable overrides
        config_data = self._apply_env_overrides(config_data)
        
        # Create configuration objects
        config = self._create_config_from_dict(config_data)
        
        # Validate configuration
        self.validate(config)
        
        self._config = config
        return config
    
    def _create_config_from_dict(self, config_data: Dict[str, Any]) -> AppConfig:
        """Create AppConfig from dictionary data."""
        # Extract sub-configurations with defaults
        model_data = config_data.get('model', {})
        data_data = config_data.get('data', {})
        recommendation_data = config_data.get('recommendation', {})
        logging_data = config_data.get('logging', {})
        versioning_data = config_data.get('versioning', {})
        
        # Create sub-configurations
        model_config = ModelConfig(
            hidden_layers=model_data.get('hidden_layers', ModelConfig().hidden_layers),
            dropout_rate=model_data.get('dropout_rate', ModelConfig().dropout_rate),
            learning_rate=model_data.get('learning_rate', ModelConfig().learning_rate),
            epochs=model_data.get('epochs', ModelConfig().epochs),
            batch_size=model_data.get('batch_size', ModelConfig().batch_size),
            early_stopping_patience=model_data.get('early_stopping_patience', ModelConfig().early_stopping_patience)
        )
        
        data_config = DataConfig(
            dataset_path=data_data.get('dataset_path', DataConfig().dataset_path),
            output_path=data_data.get('output_path', DataConfig().output_path),
            album_blocklist=data_data.get('album_blocklist', DataConfig().album_blocklist),
            mood_thresholds=data_data.get('mood_thresholds', DataConfig().mood_thresholds),
            feature_columns=data_data.get('feature_columns', DataConfig().feature_columns)
        )
        
        recommendation_config = RecommendationConfig(
            default_limit=recommendation_data.get('default_limit', RecommendationConfig().default_limit),
            similarity_metric=recommendation_data.get('similarity_metric', RecommendationConfig().similarity_metric),
            min_confidence=recommendation_data.get('min_confidence', RecommendationConfig().min_confidence)
        )
        
        logging_config = LoggingConfig(
            level=logging_data.get('level', LoggingConfig().level),
            format=logging_data.get('format', LoggingConfig().format)
        )
        
        versioning_config = VersioningConfig(
            model_version=versioning_data.get('model_version', VersioningConfig().model_version)
        )
        
        return AppConfig(
            model=model_config,
            data=data_config,
            recommendation=recommendation_config,
            logging=logging_config,
            versioning=versioning_config
        )
    
    def _apply_env_overrides(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment variable overrides to configuration data."""
        # Define environment variable mappings
        env_mappings = {
            'DOROTHEA_DATASET_PATH': ['data', 'dataset_path'],
            'DOROTHEA_OUTPUT_PATH': ['data', 'output_path'],
            'DOROTHEA_LOG_LEVEL': ['logging', 'level'],
            'DOROTHEA_MODEL_VERSION': ['versioning', 'model_version'],
            'DOROTHEA_EPOCHS': ['model', 'epochs'],
            'DOROTHEA_BATCH_SIZE': ['model', 'batch_size'],
            'DOROTHEA_LEARNING_RATE': ['model', 'learning_rate'],
            'DOROTHEA_DROPOUT_RATE': ['model', 'dropout_rate'],
        }
        
        for env_var, config_path in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                # Navigate to the nested dictionary location
                current = config_data
                for key in config_path[:-1]:
                    if key not in current:
                        current[key] = {}
                    current = current[key]
                
                # Convert value to appropriate type
                final_key = config_path[-1]
                if final_key in ['epochs', 'batch_size']:
                    current[final_key] = int(env_value)
                elif final_key in ['learning_rate', 'dropout_rate']:
                    current[final_key] = float(env_value)
                else:
                    current[final_key] = env_value
        
        return config_data
    
    def validate(self, config: AppConfig) -> bool:
        """
        Validate configuration values.
        
        Args:
            config: Configuration to validate
            
        Returns:
            bool: True if valid
            
        Raises:
            ConfigValidationError: If validation fails
        """
        errors = []
        
        # Validate model configuration
        if config.model.epochs <= 0:
            errors.append("Model epochs must be positive")
        
        if config.model.batch_size <= 0:
            errors.append("Model batch_size must be positive")
        
        if not (0.0 <= config.model.dropout_rate <= 1.0):
            errors.append("Model dropout_rate must be between 0.0 and 1.0")
        
        if config.model.learning_rate <= 0:
            errors.append("Model learning_rate must be positive")
        
        if not config.model.hidden_layers:
            errors.append("Model hidden_layers cannot be empty")
        
        if any(layer <= 0 for layer in config.model.hidden_layers):
            errors.append("All hidden layer sizes must be positive")
        
        # Validate data configuration
        if not config.data.dataset_path:
            errors.append("Data dataset_path cannot be empty")
        
        if not config.data.output_path:
            errors.append("Data output_path cannot be empty")
        
        if not config.data.feature_columns:
            errors.append("Data feature_columns cannot be empty")
        
        # Validate mood thresholds are in ascending order
        thresholds = list(config.data.mood_thresholds.values())
        if thresholds != sorted(thresholds):
            errors.append("Mood thresholds must be in ascending order")
        
        # Validate recommendation configuration
        if config.recommendation.default_limit <= 0:
            errors.append("Recommendation default_limit must be positive")
        
        if not (0.0 <= config.recommendation.min_confidence <= 1.0):
            errors.append("Recommendation min_confidence must be between 0.0 and 1.0")
        
        if config.recommendation.similarity_metric not in ['cosine', 'euclidean']:
            errors.append("Recommendation similarity_metric must be 'cosine' or 'euclidean'")
        
        # Validate logging configuration
        valid_log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if config.logging.level not in valid_log_levels:
            errors.append(f"Logging level must be one of: {valid_log_levels}")
        
        if config.logging.format not in ['json', 'text']:
            errors.append("Logging format must be 'json' or 'text'")
        
        # Validate versioning
        if not config.versioning.model_version:
            errors.append("Model version cannot be empty")
        
        if errors:
            raise ConfigValidationError(f"Configuration validation failed: {'; '.join(errors)}")
        
        return True
    
    def get_with_env_override(self, key: str) -> Any:
        """
        Get configuration value with potential environment variable override.
        
        Args:
            key: Configuration key in dot notation (e.g., 'model.epochs')
            
        Returns:
            Configuration value
        """
        if self._config is None:
            raise RuntimeError("Configuration not loaded. Call load() first.")
        
        # Check for environment variable override
        env_key = f"DOROTHEA_{key.upper().replace('.', '_')}"
        env_value = os.getenv(env_key)
        
        if env_value is not None:
            return env_value
        
        # Navigate configuration object
        keys = key.split('.')
        current = self._config
        
        for k in keys:
            if hasattr(current, k):
                current = getattr(current, k)
            else:
                raise KeyError(f"Configuration key not found: {key}")
        
        return current