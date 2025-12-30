"""
FastAPI routes for Dorothea AI REST API.
"""
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status

from .schemas import (
    MoodInput,
    RecommendRequest,
    RecommendResponse,
    SongResponse,
    MoodVectorResponse,
    HealthResponse,
    ErrorResponse,
    AlbumResponse,
    EraResponse,
    RecommendationFilters
)
from .dependencies import get_app_state, get_recommendation_engine, get_config
from ..recommendation.engine import RecommendationEngine
from ..recommendation.schemas import RecommendationRequest as InternalRecommendRequest
from ..data.schemas import MoodVector


router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Health check endpoint"
)
async def health_check():
    """
    Check the health status of the API.
    
    Returns information about the API status and whether the ML model is loaded.
    """
    state = get_app_state()
    config = get_config()
    
    return HealthResponse(
        status="healthy",
        version=config.versioning.model_version,
        model_loaded=state.model_loaded
    )


@router.post(
    "/recommend",
    response_model=RecommendResponse,
    responses={
        503: {"model": ErrorResponse, "description": "Model not loaded"}
    },
    tags=["Recommendations"],
    summary="Generate song recommendations based on mood"
)
async def get_recommendations(
    request: RecommendRequest,
    engine: RecommendationEngine = Depends(get_recommendation_engine)
):
    """
    Generate Taylor Swift song recommendations based on your mood.
    
    Provide a mood vector with values for valence, energy, danceability, 
    acousticness, and tempo. The API will return songs that best match
    your emotional state.
    
    **Mood Parameters:**
    - **valence**: Musical positivity/happiness (0.0 = sad, 1.0 = happy)
    - **energy**: Intensity and power (0.0 = calm, 1.0 = energetic)
    - **danceability**: Rhythm suitability (0.0 = not danceable, 1.0 = very danceable)
    - **acousticness**: Sound type (0.0 = electronic, 1.0 = acoustic)
    - **tempo**: Song tempo in BPM (typically 60-200)
    
    **Example moods:**
    - Melancholic: valence=0.2, energy=0.3, acousticness=0.8
    - Energetic happy: valence=0.8, energy=0.8, danceability=0.7
    - Calm and peaceful: valence=0.6, energy=0.2, acousticness=0.7
    """
    try:
        # Convert API request to internal format
        mood_vector = MoodVector(
            valence=request.mood_input.valence,
            energy=request.mood_input.energy,
            danceability=request.mood_input.danceability,
            acousticness=request.mood_input.acousticness,
            tempo_normalized=request.mood_input.tempo_normalized
        )
        
        # Build filters dict
        filters = None
        if request.filters:
            filters = {}
            if request.filters.album:
                filters["album"] = request.filters.album
            if request.filters.era:
                filters["era"] = request.filters.era
            if request.filters.exclude_albums:
                filters["exclude_albums"] = request.filters.exclude_albums
        
        # Create internal request
        internal_request = InternalRecommendRequest(
            mood_input=mood_vector,
            filters=filters,
            limit=request.limit
        )
        
        # Get recommendations
        response = engine.recommend(internal_request)
        
        # Convert to API response
        recommendations = []
        for rec in response.recommendations:
            mood_resp = None
            if rec.mood_vector:
                mood_resp = MoodVectorResponse(
                    valence=rec.mood_vector.valence,
                    energy=rec.mood_vector.energy,
                    danceability=rec.mood_vector.danceability,
                    acousticness=rec.mood_vector.acousticness,
                    tempo_normalized=rec.mood_vector.tempo_normalized
                )
            
            recommendations.append(SongResponse(
                track_id=rec.track_id,
                track_name=rec.track_name,
                album=rec.album,
                uri=rec.uri,
                similarity_score=rec.similarity_score,
                confidence=rec.confidence,
                mood_vector=mood_resp
            ))
        
        query_mood = MoodVectorResponse(
            valence=response.query_mood.valence,
            energy=response.query_mood.energy,
            danceability=response.query_mood.danceability,
            acousticness=response.query_mood.acousticness,
            tempo_normalized=response.query_mood.tempo_normalized
        )
        
        return RecommendResponse(
            recommendations=recommendations,
            query_mood=query_mood,
            processing_time_ms=response.processing_time_ms,
            total_candidates=response.total_candidates
        )
        
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate recommendations: {str(e)}"
        )


@router.get(
    "/albums",
    response_model=AlbumResponse,
    responses={
        503: {"model": ErrorResponse, "description": "Model not loaded"}
    },
    tags=["Metadata"],
    summary="List available albums"
)
async def list_albums(
    engine: RecommendationEngine = Depends(get_recommendation_engine)
):
    """
    Get a list of all available Taylor Swift albums in the dataset.
    
    Use these album names as filters in the `/recommend` endpoint.
    """
    try:
        track_mapping = engine.track_mapper.load_mapping()
        albums = set()
        
        for track_info in track_mapping.values():
            if track_info.album:
                albums.add(track_info.album)
        
        album_list = sorted(list(albums))
        
        return AlbumResponse(
            albums=album_list,
            total=len(album_list)
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load albums: {str(e)}"
        )


@router.get(
    "/eras",
    response_model=EraResponse,
    tags=["Metadata"],
    summary="List Taylor Swift eras with their albums"
)
async def list_eras():
    """
    Get a mapping of Taylor Swift eras to their associated albums.
    
    Use era names as filters in the `/recommend` endpoint.
    """
    era_mappings = {
        "debut": ["Taylor Swift"],
        "fearless": ["Fearless", "Fearless (Taylor's Version)"],
        "speak_now": ["Speak Now", "Speak Now (Taylor's Version)"],
        "red": ["Red", "Red (Taylor's Version)"],
        "1989": ["1989", "1989 (Taylor's Version)"],
        "reputation": ["reputation"],
        "lover": ["Lover"],
        "folklore": ["folklore"],
        "evermore": ["evermore"],
        "midnights": ["Midnights"],
        "tortured_poets": ["The Tortured Poets Department"]
    }
    
    return EraResponse(eras=era_mappings)
