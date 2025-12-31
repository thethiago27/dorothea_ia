"""
Data validation module for the Dorothea AI system.
"""
from typing import List, Set, Dict, Any, Optional
import pandas as pd
import numpy as np
from .schemas import ValidationResult

class DataValidator:
    """Validates dataset schema, missing values, and feature ranges."""
    
    REQUIRED_COLUMNS = {
        'id', 'name', 'album', 'uri', 'valence', 'energy', 
        'danceability', 'acousticness', 'tempo'
    }
    
    COLUMN_TYPES = {
        'id': 'object',
        'name': 'object', 
        'album': 'object',
        'uri': 'object',
        'valence': 'float64',
        'energy': 'float64',
        'danceability': 'float64',
        'acousticness': 'float64',
        'instrumentalness': 'float64',
        'liveness': 'float64',
        'speechiness': 'float64',
        'tempo': 'float64',
        'loudness': 'float64',
        'popularity': 'int64',
        'duration_ms': 'int64',
        'track_number': 'int64'
    }
    FEATURE_RANGES = {
        'acousticness': (0.0, 1.0),
        'danceability': (0.0, 1.0),
        'energy': (0.0, 1.0),
        'instrumentalness': (0.0, 1.0),
        'liveness': (0.0, 1.0),
        'speechiness': (0.0, 1.0),
        'valence': (0.0, 1.0),
        'tempo': (0.0, 250.0),
        'loudness': (-60.0, 5.0),
        'popularity': (0, 100),
        'duration_ms': (1000, 3600000),
        'track_number': (1, 100)
    }
    def validate_schema(self, df: pd.DataFrame) -> ValidationResult:
        """Validate dataset schema including required columns and data types.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            ValidationResult with schema validation results
        """
        result = ValidationResult(is_valid=True, errors=[], warnings=[])
        missing_columns = self.REQUIRED_COLUMNS - set(df.columns)
        if missing_columns:
            for col in missing_columns:
                result.add_error(f"Missing required column: {col}")
        extra_columns = set(df.columns) - self.REQUIRED_COLUMNS
        if extra_columns:
            for col in extra_columns:
                result.add_warning(f"Unexpected column found: {col}")
        for col in self.REQUIRED_COLUMNS.intersection(set(df.columns)):
            expected_type = self.COLUMN_TYPES[col]
            actual_type = str(df[col].dtype)
            if expected_type == 'float64' and actual_type in ['float32', 'int64', 'int32']:
                continue
            elif expected_type == 'int64' and actual_type in ['int32', 'float64']:
                if actual_type == 'float64' and not df[col].apply(lambda x: pd.isna(x) or x.is_integer()).all():
                    result.add_error(f"Column {col} contains non-integer values but should be integer")
                continue
            elif actual_type != expected_type:
                result.add_error(f"Column {col} has type {actual_type}, expected {expected_type}")
        result.metadata = {
            'total_columns': len(df.columns),
            'total_rows': len(df),
            'missing_columns_count': len(missing_columns),
            'extra_columns_count': len(extra_columns)
        }
        return result

    def check_missing_values(self, df: pd.DataFrame) -> ValidationResult:
        """Check for missing values in the dataset.
        
        Args:
            df: DataFrame to check for missing values
            
        Returns:
            ValidationResult with missing value analysis
        """
        result = ValidationResult(is_valid=True, errors=[], warnings=[])
        missing_counts = df.isnull().sum()
        total_rows = len(df)
        for col, missing_count in missing_counts.items():
            if missing_count > 0:
                missing_percentage = (missing_count / total_rows) * 100
                critical_columns = {'id', 'name', 'album', 'uri'}
                if col in critical_columns:
                    result.add_error(f"Critical column {col} has {missing_count} missing values ({missing_percentage:.1f}%)")
                elif missing_percentage > 10:
                    result.add_error(f"Column {col} has {missing_count} missing values ({missing_percentage:.1f}%) - exceeds 10% threshold")
                else:
                    result.add_warning(f"Column {col} has {missing_count} missing values ({missing_percentage:.1f}%)")
        result.metadata = {
            'total_missing_values': int(missing_counts.sum()),
            'columns_with_missing': len(missing_counts[missing_counts > 0]),
            'missing_value_percentage': float((missing_counts.sum() / (total_rows * len(df.columns))) * 100)
        }
        return result

    def validate_ranges(self, df: pd.DataFrame) -> ValidationResult:
        """Validate that feature values are within expected ranges.
        
        Args:
            df: DataFrame to validate ranges for
            
        Returns:
            ValidationResult with range validation results
        """
        result = ValidationResult(is_valid=True, errors=[], warnings=[])
        for col, (min_val, max_val) in self.FEATURE_RANGES.items():
            if col not in df.columns:
                continue
            valid_data = df[col].dropna()
            if len(valid_data) == 0:
                continue
            out_of_range = valid_data[(valid_data < min_val) | (valid_data > max_val)]
            if len(out_of_range) > 0:
                out_of_range_percentage = (len(out_of_range) / len(valid_data)) * 100
                if out_of_range_percentage > 5:
                    result.add_error(
                        f"Column {col} has {len(out_of_range)} values ({out_of_range_percentage:.1f}%) "
                        f"outside valid range [{min_val}, {max_val}]"
                    )
                else:
                    result.add_warning(
                        f"Column {col} has {len(out_of_range)} values ({out_of_range_percentage:.1f}%) "
                        f"outside valid range [{min_val}, {max_val}]"
                    )
        result.metadata = {
            'columns_validated': len([col for col in self.FEATURE_RANGES.keys() if col in df.columns]),
            'total_values_checked': sum(len(df[col].dropna()) for col in self.FEATURE_RANGES.keys() if col in df.columns)
        }
        return result
        
    def validate_all(self, df: pd.DataFrame) -> ValidationResult:
        """Run all validation checks on the dataset.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Combined ValidationResult from all checks
        """
        schema_result = self.validate_schema(df)
        missing_result = self.check_missing_values(df)
        range_result = self.validate_ranges(df)
        combined_result = ValidationResult(
            is_valid=schema_result.is_valid and missing_result.is_valid and range_result.is_valid,
            errors=schema_result.errors + missing_result.errors + range_result.errors,
            warnings=schema_result.warnings + missing_result.warnings + range_result.warnings,
            metadata={
                'schema_validation': schema_result.metadata,
                'missing_values': missing_result.metadata,
                'range_validation': range_result.metadata
            }
        )
        return combined_result