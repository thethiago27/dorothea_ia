"""
Dorothea AI REST API Server

Run the server with:
    python api.py
    
Or with uvicorn directly:
    uvicorn api:app --reload --host 0.0.0.0 --port 8000
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from dorothea.api.routes import router
from dorothea.api.dependencies import get_app_state


# Create FastAPI app
app = FastAPI(
    title="Dorothea AI",
    description="""
🎵 **Taylor Swift Song Recommendation API** 🎵

Dorothea AI provides personalized Taylor Swift song recommendations based on your mood.
Using advanced machine learning and audio feature analysis, it matches songs to your emotional state.

## Features

- **Mood-based recommendations**: Provide your mood parameters and get matching songs
- **Album/Era filtering**: Filter recommendations by specific albums or Taylor Swift eras
- **Similarity scoring**: Each recommendation includes a similarity and confidence score

## Quick Start

1. Check API health: `GET /health`
2. Get recommendations: `POST /recommend` with your mood parameters
3. Explore available albums: `GET /albums`
4. See era mappings: `GET /eras`

## Mood Parameters

| Parameter | Description | Range |
|-----------|-------------|-------|
| valence | Musical positivity/happiness | 0.0 - 1.0 |
| energy | Intensity and power | 0.0 - 1.0 |
| danceability | Rhythm suitability | 0.0 - 1.0 |
| acousticness | Acoustic vs electronic | 0.0 - 1.0 |
| tempo | Song tempo in BPM | 0 - 300 |
    """,
    version="2.0.0",
    contact={
        "name": "Dorothea AI",
        "url": "https://github.com/thethiago27/dorothea_ia"
    },
    license_info={
        "name": "MIT",
    }
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix="/api/v1")

# Also mount at root for convenience
app.include_router(router)


@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup."""
    print("🎵 Starting Dorothea AI API...")
    try:
        state = get_app_state()
        if state.model_loaded:
            print("✅ Model loaded successfully")
        else:
            print("⚠️  Model not loaded. Train first with: python main.py train")
        print("🚀 Dorothea AI API is ready!")
        print("📚 API docs available at: http://localhost:8000/docs")
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
