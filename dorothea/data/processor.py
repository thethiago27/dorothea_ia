"""
Data processing module for the Dorothea AI system.
"""
import os
import pickle
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from .schemas import ProcessedDataset, ValidationResult, SpotifyTrackData, MoodVector
from .validator import DataValidator


class DataProcessor:
    """Handles data loading, preprocessing, and feature engineering."""
    
    def __init__(self, config: Dict[str, Any], validator: Optional[DataValidator] = None):
        """Initialize the data processor with configuration.
        
        Args:
            config: Configuration dictionary
            validator: Data validator instance (optional)
        """
        self.config = config
        self.validator = validator or DataValidator()
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.feature_columns = config.get('feature_columns', [
            'valence', 'energy', 'danceability', 'acousticness', 'tempo'
        ])
        self.album_blocklist = set(config.get('album_blocklist', []))
        self.mood_thresholds = config.get('mood_thresholds', {
            'low': 0.3,
            'neutral': 0.5,
            'high': 0.7
        })
        
    def load_dataset(self, path: Optional[str] = None) -> pd.DataFrame:
        """Load dataset from CSV file.
        
        Args:
            path: Path to CSV file (uses config path if None)
            
        Returns:
            Loaded DataFrame
            
        Raises:
            ValueError: If no dataset path provided or CSV loading fails
            FileNotFoundError: If dataset file doesn't exist
        """
        dataset_path = path or self.config.get('dataset_path')
        if not dataset_path:
            raise ValueError("No dataset path provided in config or parameter")
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
        try:
            df = pd.read_csv(dataset_path)
        except Exception as e:
            raise ValueError(f"Failed to load CSV file: {e}")
        validation_result = self.validator.validate_all(df)
        if validation_result.has_errors():
            error_msg = "Dataset validation failed:\n" + "\n".join(validation_result.errors)
            raise ValueError(error_msg)
        if validation_result.has_warnings():
            print("Dataset validation warnings:")
            for warning in validation_result.warnings:
                print(f"  - {warning}")
        return df
    def filter_albums(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter out albums from the blocklist.
        
        Args:
            df: Input DataFrame containing album information
            
        Returns:
            Filtered DataFrame with blocked albums removed
        """
        if not self.album_blocklist:
            return df.copy()
        initial_count = len(df)
        filtered_df = df[~df['album'].isin(self.album_blocklist)].copy()
        filtered_count = len(filtered_df)
        removed_count = initial_count - filtered_count
        if removed_count > 0:
            print(f"Filtered out {removed_count} tracks from {len(self.album_blocklist)} blocked albums")
        return filtered_df
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset using appropriate strategies.
        
        Args:
            df: Input DataFrame with potential missing values
            
        Returns:
            DataFrame with missing values handled
        """
        df_processed = df.copy()
        feature_cols_in_df = [col for col in self.feature_columns if col in df_processed.columns]
        if feature_cols_in_df:
            df_processed[feature_cols_in_df] = self.imputer.fit_transform(df_processed[feature_cols_in_df])
        categorical_cols = ['name', 'album', 'release_date', 'uri']
        for col in categorical_cols:
            if col in df_processed.columns:
                df_processed[col] = df_processed[col].fillna('Unknown')
        numeric_cols = ['track_number', 'popularity', 'duration_ms', 'loudness', 
                       'instrumentalness', 'liveness', 'speechiness']
        for col in numeric_cols:
            if col in df_processed.columns:
                df_processed[col] = df_processed[col].fillna(df_processed[col].median())
        return df_processed
    def _normalize_tempo(self, tempo_values: np.ndarray) -> np.ndarray:
        """Normalize tempo values to [0, 1] range.
        
        Args:
            tempo_values: Array of tempo values to normalize
            
        Returns:
            Normalized tempo values in [0, 1] range
        """
        clipped_tempo = np.clip(tempo_values, 40, 200)
        normalized_tempo = (clipped_tempo - 40) / (200 - 40)
        return normalized_tempo
    def normalize_features(self, df: pd.DataFrame) -> ProcessedDataset:
        """Normalize features and create processed dataset.
        
        Args:
            df: Input DataFrame with raw features
            
        Returns:
            ProcessedDataset with normalized features and mood labels
            
        Raises:
            ValueError: If required feature columns are not found in dataset
        """
        df_clean = self._handle_missing_values(df)
        feature_cols_in_df = [col for col in self.feature_columns if col in df_clean.columns]
        if not feature_cols_in_df:
            raise ValueError(f"None of the required feature columns {self.feature_columns} found in dataset")
        feature_data = df_clean[feature_cols_in_df].copy()
        if 'tempo' in feature_data.columns:
            feature_data['tempo'] = self._normalize_tempo(feature_data['tempo'].values)
        normalized_features = self.scaler.fit_transform(feature_data)
        if 'valence' in df_clean.columns:
            valence_values = df_clean['valence'].values
            mood_labels = self._create_mood_labels(valence_values)
        else:
            if 'energy' in df_clean.columns:
                mood_labels = self._create_mood_labels(df_clean['energy'].values)
            else:
                mood_labels = np.array(['neutral'] * len(df_clean))
        scaler_params = {
            'mean': self.scaler.mean_.tolist(),
            'scale': self.scaler.scale_.tolist(),
            'feature_columns': feature_cols_in_df
        }
        
        track_ids = df_clean['id'].tolist() if 'id' in df_clean.columns else list(range(len(df_clean)))
        
        metadata = {
            'original_shape': df_clean.shape,
            'feature_columns': feature_cols_in_df,
            'mood_thresholds': self.mood_thresholds
        }
        return ProcessedDataset(
            features=normalized_features,
            labels=mood_labels,
            track_ids=track_ids,
            scaler_params=scaler_params,
            metadata=metadata
        )
    def _create_mood_labels(self, values: np.ndarray) -> np.ndarray:
        """Create mood labels based on value thresholds.
        
        Args:
            values: Array of values to classify (typically valence)
            
        Returns:
            Array of mood labels ('low', 'neutral', 'high')
        """
        labels = np.full(len(values), 'neutral', dtype=object)
        labels[values <= self.mood_thresholds['low']] = 'low'
        labels[(values > self.mood_thresholds['low']) & 
               (values <= self.mood_thresholds['neutral'])] = 'neutral'
        labels[values > self.mood_thresholds['high']] = 'high'
        return labels
    def save_scaler(self, path: str) -> None:
        """Save the fitted scaler and related parameters to disk.
        
        Args:
            path: File path to save the scaler data
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        scaler_data = {
            'scaler': self.scaler,
            'imputer': self.imputer,
            'feature_columns': self.feature_columns,
            'mood_thresholds': self.mood_thresholds
        }
        with open(path, 'wb') as f:
            pickle.dump(scaler_data, f)
    def load_scaler(self, path: str) -> StandardScaler:
        """Load a previously saved scaler and related parameters from disk.
        
        Args:
            path: File path to load the scaler data from
            
        Returns:
            The loaded StandardScaler instance
            
        Raises:
            FileNotFoundError: If scaler file doesn't exist
            ValueError: If scaler loading fails
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Scaler file not found: {path}")
        try:
            with open(path, 'rb') as f:
                scaler_data = pickle.load(f)
            self.scaler = scaler_data['scaler']
            self.imputer = scaler_data['imputer']
            self.feature_columns = scaler_data['feature_columns']
            self.mood_thresholds = scaler_data['mood_thresholds']
            return self.scaler
        except Exception as e:
            raise ValueError(f"Failed to load scaler: {e}")
            
    def process_full_pipeline(self, dataset_path: Optional[str] = None) -> ProcessedDataset:
        """Execute the complete data processing pipeline.
        
        Args:
            dataset_path: Path to dataset (uses config if None)
            
        Returns:
            ProcessedDataset with normalized features and labels
        """
        df = self.load_dataset(dataset_path)
        df_filtered = self.filter_albums(df)
        processed_dataset = self.normalize_features(df_filtered)
        return processed_dataset