"""
Pydantic schemas for Dorothea AI REST API.
"""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, field_validator


class MoodInput(BaseModel):
    """Mood vector input for recommendations."""
    valence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Musical positivity/happiness (0.0-1.0)"
    )
    energy: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Intensity and power (0.0-1.0)"
    )
    danceability: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Rhythm suitability for dancing (0.0-1.0)"
    )
    acousticness: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Acoustic vs electronic sound (0.0-1.0)"
    )
    tempo: float = Field(
        default=120.0,
        ge=0.0,
        le=300.0,
        description="Song tempo in BPM (will be normalized)"
    )

    @property
    def tempo_normalized(self) -> float:
        """Normalize tempo to 0-1 range."""
        return min(1.0, self.tempo / 250.0)


class RecommendationFilters(BaseModel):
    """Filters for recommendation queries."""
    album: Optional[str] = Field(
        default=None,
        description="Filter by specific album name"
    )
    era: Optional[str] = Field(
        default=None,
        description="Filter by Taylor Swift era (e.g., 'folklore', 'red', '1989')"
    )
    exclude_albums: Optional[List[str]] = Field(
        default=None,
        description="List of albums to exclude"
    )


class RecommendRequest(BaseModel):
    """Request body for recommendation endpoint."""
    mood_input: MoodInput = Field(
        default_factory=MoodInput,
        description="Mood vector describing desired emotional state"
    )
    filters: Optional[RecommendationFilters] = Field(
        default=None,
        description="Optional filters for recommendations"
    )
    limit: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Number of recommendations to return (1-50)"
    )


class MoodVectorResponse(BaseModel):
    """Mood vector in response format."""
    valence: float
    energy: float
    danceability: float
    acousticness: float
    tempo_normalized: float


class SongResponse(BaseModel):
    """Individual song recommendation response."""
    track_id: str = Field(description="Internal track ID")
    track_name: str = Field(description="Song name")
    album: str = Field(description="Album name")
    uri: str = Field(description="Spotify URI")
    similarity_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Cosine similarity to query mood"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score for this recommendation"
    )
    mood_vector: Optional[MoodVectorResponse] = Field(
        default=None,
        description="Track's mood vector"
    )


class RecommendResponse(BaseModel):
    """Response body for recommendation endpoint."""
    recommendations: List[SongResponse] = Field(
        description="List of recommended songs"
    )
    query_mood: MoodVectorResponse = Field(
        description="The mood vector used for the query"
    )
    processing_time_ms: float = Field(
        ge=0.0,
        description="Time taken to process the request in milliseconds"
    )
    total_candidates: int = Field(
        ge=0,
        description="Total number of candidate songs considered"
    )


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(description="API status")
    version: str = Field(description="API version")
    model_loaded: bool = Field(description="Whether the ML model is loaded")


class ErrorResponse(BaseModel):
    """Error response format."""
    error: str = Field(description="Error message")
    detail: Optional[str] = Field(default=None, description="Detailed error information")


class AlbumResponse(BaseModel):
    """Available albums response."""
    albums: List[str] = Field(description="List of available albums")
    total: int = Field(description="Total number of albums")


class EraResponse(BaseModel):
    """Available eras response."""
    eras: Dict[str, List[str]] = Field(
        description="Mapping of era names to album lists"
    )
