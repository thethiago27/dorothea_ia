"""
Track mapping module for Dorothea AI.

This module provides functionality for storing and retrieving track metadata
with internal ID mapping for efficient lookups and persistence.
"""
import json
import os
import tempfile
from dataclasses import dataclass, asdict
from typing import Dict, Optional, List, Any
import logging
from ..data.schemas import MoodVector, ValidationResult


@dataclass
class TrackInfo:
    """
    Track information with internal ID mapping.
    
    Attributes:
        internal_id: Internal numeric ID for efficient indexing
        spotify_id: Spotify track ID
        uri: Spotify URI for the track
        name: Track name
        album: Album name
        mood_vector: Optional mood vector for the track
    """
    internal_id: int
    spotify_id: str
    uri: str
    name: str
    album: str
    mood_vector: Optional[MoodVector] = None

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert TrackInfo to dictionary for JSON serialization.
        
        Returns:
            Dictionary representation of the track info
        """
        result = asdict(self)
        if self.mood_vector is not None:
            result['mood_vector'] = {
                'valence': self.mood_vector.valence,
                'energy': self.mood_vector.energy,
                'danceability': self.mood_vector.danceability,
                'acousticness': self.mood_vector.acousticness,
                'tempo_normalized': self.mood_vector.tempo_normalized
            }
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrackInfo':
        """
        Create TrackInfo from dictionary data.
        
        Args:
            data: Dictionary containing track information
            
        Returns:
            TrackInfo instance
            
        Raises:
            KeyError: If required fields are missing
            ValueError: If data types are invalid
        """
        mood_vector = None
        if 'mood_vector' in data and data['mood_vector'] is not None:
            mv_data = data['mood_vector']
            mood_vector = MoodVector(
                valence=mv_data['valence'],
                energy=mv_data['energy'],
                danceability=mv_data['danceability'],
                acousticness=mv_data['acousticness'],
                tempo_normalized=mv_data['tempo_normalized']
            )
        return cls(
            internal_id=data['internal_id'],
            spotify_id=data['spotify_id'],
            uri=data['uri'],
            name=data['name'],
            album=data['album'],
            mood_vector=mood_vector
        )


class TrackMapper:
    """
    Manages track mapping persistence and retrieval.
    
    Provides functionality for creating, saving, and loading track mappings
    with internal ID indexing for efficient lookups.
    """

    def __init__(self, storage_path: str):
        """
        Initialize TrackMapper with storage path.
        
        Args:
            storage_path: Path to the JSON file for storing track mappings
        """
        self.storage_path = storage_path
        self.logger = logging.getLogger(__name__)
        self._mapping_cache: Optional[Dict[int, TrackInfo]] = None
        self._spotify_id_index: Optional[Dict[str, int]] = None

    def create_mapping(self, tracks: List[TrackInfo]) -> None:
        """
        Create and save a new track mapping from a list of tracks.
        
        Args:
            tracks: List of TrackInfo objects to create mapping from
            
        Raises:
            ValueError: If duplicate internal IDs are found
            IOError: If saving fails
        """
        seen_ids = set()
        seen_spotify_ids = set()
        
        for track in tracks:
            if track.internal_id in seen_ids:
                error_msg = f"ID collision detected: internal_id {track.internal_id} appears multiple times"
                self.logger.error(error_msg)
                raise ValueError(error_msg)
            if track.spotify_id in seen_spotify_ids:
                self.logger.warning(f"Duplicate Spotify ID detected: {track.spotify_id}")
            
            seen_ids.add(track.internal_id)
            seen_spotify_ids.add(track.spotify_id)

        mapping = {track.internal_id: track for track in tracks}
        self.save_atomic(mapping)
        
        # Clear cache to force reload
        self._mapping_cache = None
        self._spotify_id_index = None
        
        self.logger.info(f"Created mapping with {len(tracks)} tracks")

    def load_mapping(self) -> Dict[int, TrackInfo]:
        """
        Load track mapping from storage.
        
        Returns:
            Dictionary mapping internal IDs to TrackInfo objects
            
        Raises:
            FileNotFoundError: If mapping file doesn't exist
            json.JSONDecodeError: If file contains invalid JSON
            ValueError: If file format is invalid
        """
        if self._mapping_cache is not None:
            return self._mapping_cache

        if not os.path.exists(self.storage_path):
            raise FileNotFoundError(f"Mapping file not found: {self.storage_path}")

        try:
            with open(self.storage_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if not isinstance(data, dict):
                raise ValueError("Mapping file must contain a JSON object")

            mapping = {}
            for key, value in data.items():
                try:
                    internal_id = int(key)
                    track_info = TrackInfo.from_dict(value)
                    
                    if track_info.internal_id != internal_id:
                        self.logger.warning(
                            f"Internal ID mismatch: key={internal_id}, "
                            f"track.internal_id={track_info.internal_id}"
                        )
                    
                    mapping[internal_id] = track_info
                except (ValueError, KeyError, TypeError) as e:
                    self.logger.error(f"Invalid track data for ID {key}: {e}")
                    continue

            self._mapping_cache = mapping
            self._build_spotify_id_index()
            
            self.logger.info(f"Loaded mapping with {len(mapping)} tracks")
            return mapping

        except json.JSONDecodeError as e:
            error_msg = f"JSON corruption in mapping file {self.storage_path}: {e}"
            self.logger.error(error_msg)
            raise json.JSONDecodeError(error_msg, e.doc, e.pos)
        except Exception as e:
            error_msg = f"Failed to load mapping from {self.storage_path}: {e}"
            self.logger.error(error_msg)
            raise

    def get_track(self, internal_id: int) -> Optional[TrackInfo]:
        """
        Get track information by internal ID.
        
        Args:
            internal_id: Internal ID of the track
            
        Returns:
            TrackInfo object if found, None otherwise
        """
        try:
            mapping = self.load_mapping()
            return mapping.get(internal_id)
        except Exception as e:
            self.logger.error(f"Failed to get track {internal_id}: {e}")
            return None

    def get_by_spotify_id(self, spotify_id: str) -> Optional[TrackInfo]:
        """
        Get track information by Spotify ID.
        
        Args:
            spotify_id: Spotify ID of the track
            
        Returns:
            TrackInfo object if found, None otherwise
        """
        try:
            self.load_mapping()  # Ensure mapping is loaded
            
            if self._spotify_id_index is None:
                self._build_spotify_id_index()
            
            internal_id = self._spotify_id_index.get(spotify_id)
            if internal_id is not None:
                return self._mapping_cache.get(internal_id)
            
            return None
        except Exception as e:
            self.logger.error(f"Failed to get track by Spotify ID {spotify_id}: {e}")
            return None

    def validate_mapping(self) -> ValidationResult:
        """
        Validate the current track mapping for consistency and completeness.
        
        Returns:
            ValidationResult with validation status and any issues found
        """
        result = ValidationResult(is_valid=True, errors=[], warnings=[])
        
        try:
            mapping = self.load_mapping()
            
            # Check for missing required fields
            for internal_id, track in mapping.items():
                if not track.spotify_id:
                    result.add_error(f"Track {internal_id} missing spotify_id")
                if not track.uri:
                    result.add_error(f"Track {internal_id} missing uri")
                if not track.name:
                    result.add_error(f"Track {internal_id} missing name")
                if not track.album:
                    result.add_error(f"Track {internal_id} missing album")

            # Check for duplicate Spotify IDs
            spotify_ids = {}
            for internal_id, track in mapping.items():
                if track.spotify_id in spotify_ids:
                    result.add_warning(
                        f"Duplicate Spotify ID {track.spotify_id} found in tracks "
                        f"{spotify_ids[track.spotify_id]} and {internal_id}"
                    )
                else:
                    spotify_ids[track.spotify_id] = internal_id

            # Add metadata
            result.metadata = {
                'total_tracks': len(mapping),
                'tracks_with_mood_vector': sum(1 for t in mapping.values() if t.mood_vector is not None),
                'unique_albums': len(set(t.album for t in mapping.values())),
                'unique_spotify_ids': len(spotify_ids)
            }

        except Exception as e:
            result.add_error(f"Failed to validate mapping: {e}")

        return result

    def save_atomic(self, mapping: Dict[int, TrackInfo]) -> None:
        """
        Atomically save track mapping to storage.
        
        Args:
            mapping: Dictionary mapping internal IDs to TrackInfo objects
            
        Raises:
            IOError: If saving fails
        """
        # Convert mapping to serializable format
        data = {}
        for internal_id, track in mapping.items():
            data[str(internal_id)] = track.to_dict()

        # Ensure directory exists
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)

        temp_path = None
        try:
            # Write to temporary file first
            temp_dir = os.path.dirname(self.storage_path)
            with tempfile.NamedTemporaryFile(
                mode='w', 
                dir=temp_dir, 
                delete=False, 
                suffix='.tmp',
                encoding='utf-8'
            ) as temp_file:
                temp_path = temp_file.name
                json.dump(data, temp_file, indent=2, ensure_ascii=False)

            # Atomic rename (handle Windows compatibility)
            if os.name == 'nt':  # Windows
                if os.path.exists(self.storage_path):
                    os.remove(self.storage_path)
            
            os.rename(temp_path, self.storage_path)
            temp_path = None  # Successfully renamed
            
            self.logger.info(f"Atomically saved mapping with {len(mapping)} tracks to {self.storage_path}")

        except Exception as e:
            # Clean up temporary file if it exists
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass
            
            error_msg = f"Failed to save mapping atomically: {e}"
            self.logger.error(error_msg)
            raise IOError(error_msg)

    def _build_spotify_id_index(self) -> None:
        """
        Build internal index for Spotify ID lookups.
        
        This method is called internally to create a reverse index
        for efficient Spotify ID to internal ID mapping.
        """
        if self._mapping_cache is None:
            return

        self._spotify_id_index = {}
        for internal_id, track in self._mapping_cache.items():
            if track.spotify_id:
                self._spotify_id_index[track.spotify_id] = internal_id