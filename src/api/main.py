"""
FastAPI Application for Media Recommendation Service

Production-ready API for serving personalized recommendations.
"""

import os
import time
import logging
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException, Depends, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.responses import Response

# Optional imports - graceful degradation
try:
    import structlog
    logger = structlog.get_logger()
except ImportError:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

try:
    from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    Counter = None
    Histogram = None

try:
    from opentelemetry import trace
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    trace = None
    FastAPIInstrumentor = None

# Local imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import media database
try:
    from src.data.media_database import (
        MEDIA_DATABASE, get_all_media, get_media_by_id, 
        get_media_by_genre, get_recommendations_for_user,
        get_similar_items as get_similar_media,
        get_trending, get_top_rated
    )
    from src.data.user_profiles import (
        get_user_profile, get_personalized_recommendations,
        get_user_profile_summary, get_recommendation_explanation,
        USER_PROFILES
    )
    MEDIA_DB_AVAILABLE = True
except ImportError:
    MEDIA_DB_AVAILABLE = False
    MEDIA_DATABASE = []
    USER_PROFILES = {}

# Prometheus metrics (only if available)
if PROMETHEUS_AVAILABLE:
    REQUEST_COUNT = Counter(
        "recommendation_requests_total",
        "Total recommendation requests",
        ["endpoint", "status"]
    )
    REQUEST_LATENCY = Histogram(
        "recommendation_request_latency_seconds",
        "Request latency in seconds",
        ["endpoint"]
    )
    RECOMMENDATION_QUALITY = Histogram(
        "recommendation_diversity_score",
        "Diversity score of recommendations"
    )
else:
    REQUEST_COUNT = None
    REQUEST_LATENCY = None
    RECOMMENDATION_QUALITY = None


# Pydantic models
class UserInteraction(BaseModel):
    """User interaction event."""
    user_id: int
    item_id: int
    interaction_type: str = "view"  # view, click, purchase, rating
    rating: Optional[float] = None
    timestamp: Optional[int] = None


class UserProfile(BaseModel):
    """User profile for personalization."""
    user_id: int
    preferences: List[str] = Field(default_factory=list)
    demographics: Dict[str, Any] = Field(default_factory=dict)
    history: List[int] = Field(default_factory=list)


class RecommendationRequest(BaseModel):
    """Request for recommendations."""
    n_recommendations: int = Field(default=10, ge=1, le=100)
    exclude_items: List[int] = Field(default_factory=list)
    content_filter: Optional[str] = None
    diversity_factor: float = Field(default=0.3, ge=0, le=1)


class RecommendationItem(BaseModel):
    """Single recommendation item."""
    item_id: int
    score: float
    explanation: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RecommendationResponse(BaseModel):
    """Recommendation response."""
    user_id: int
    recommendations: List[RecommendationItem]
    model_version: str
    latency_ms: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    models_loaded: bool
    uptime_seconds: float


class ItemInfo(BaseModel):
    """Item information."""
    item_id: int
    title: str
    description: Optional[str] = None
    genre: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    popularity_score: float = 0.0


# Global state
class AppState:
    """Application state container."""
    
    def __init__(self):
        self.recommender = None
        self.models_loaded = False
        self.start_time = time.time()
        self.model_version = "1.0.0"
        self.item_catalog: Dict[int, Dict] = {}
    
    async def initialize(self):
        """Initialize models and state."""
        logger.info("Initializing recommendation models...")
        
        try:
            # In production, load actual models here
            # self.recommender = HybridRecommender(config)
            # self.recommender.load("checkpoints/")
            
            # For demo, we'll simulate model loading
            self.models_loaded = True
            self._load_sample_catalog()
            
            logger.info("Models initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize models: {e}")
            raise
    
    def _load_sample_catalog(self):
        """Load sample item catalog."""
        # In production, load from database
        self.item_catalog = {
            i: {
                "title": f"Item {i}",
                "description": f"Description for item {i}",
                "genre": ["Action", "Comedy", "Drama", "Sci-Fi"][i % 4],
                "tags": ["popular", "trending"] if i < 100 else ["niche"]
            }
            for i in range(1, 10001)
        }
    
    def get_uptime(self) -> float:
        return time.time() - self.start_time


app_state = AppState()


# Lifespan management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan."""
    # Startup
    await app_state.initialize()
    yield
    # Shutdown
    logger.info("Shutting down...")


# Create FastAPI app
app = FastAPI(
    title="Media Recommendation API",
    description="Personalized content recommendation service using hybrid ML models",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OpenTelemetry instrumentation (optional)
if OTEL_AVAILABLE and FastAPIInstrumentor is not None:
    try:
        FastAPIInstrumentor.instrument_app(app)
    except Exception as e:
        logging.warning(f"Could not instrument FastAPI with OpenTelemetry: {e}")

# Static files for demo UI
from pathlib import Path
static_dir = Path(__file__).parent.parent.parent / "static"
if static_dir.exists():
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import FileResponse
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    
    @app.get("/", include_in_schema=False)
    async def serve_demo():
        """Serve the demo UI."""
        return FileResponse(str(static_dir / "index.html"))


# Dependencies
async def get_state() -> AppState:
    """Get application state."""
    if not app_state.models_loaded:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Service is starting up."
        )
    return app_state


# Routes
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if app_state.models_loaded else "starting",
        version=app_state.model_version,
        models_loaded=app_state.models_loaded,
        uptime_seconds=app_state.get_uptime()
    )


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    if PROMETHEUS_AVAILABLE:
        return Response(
            content=generate_latest(),
            media_type=CONTENT_TYPE_LATEST
        )
    else:
        return JSONResponse(
            content={"error": "Prometheus metrics not available. Install prometheus_client."},
            status_code=501
        )


@app.post("/api/v1/recommendations/{user_id}", response_model=RecommendationResponse)
async def get_recommendations(
    user_id: int,
    request: RecommendationRequest,
    state: AppState = Depends(get_state)
):
    """
    Get personalized recommendations for a user.
    
    Uses hybrid model combining collaborative filtering, content-based,
    sequential, and contrastive learning approaches.
    """
    start_time = time.time()
    
    try:
        # Use intelligent user profile system
        if MEDIA_DB_AVAILABLE and MEDIA_DATABASE:
            # Get ML-based personalized recommendations
            recs_data = get_personalized_recommendations(
                user_id=user_id,
                n=request.n_recommendations,
                exclude_ids=request.exclude_items
            )
            
            recommendations = []
            for i, item in enumerate(recs_data):
                recommendations.append(RecommendationItem(
                    item_id=item["id"],
                    score=item.get("score", 0.95 - i * 0.03),
                    explanation="; ".join(item.get("reasons", ["Recommended for you"])),
                    metadata={
                        "title": item["title"],
                        "genre": ", ".join(item["genres"]),
                        "genres": item["genres"],
                        "year": item["year"],
                        "rating": item["rating"],
                        "poster": item["poster"],
                        "backdrop": item.get("backdrop", ""),
                        "description": item["description"],
                        "duration": item["duration"],
                        "director": item["director"],
                        "cast": item["cast"],
                        "media_type": item["media_type"],
                        "match_percentage": item.get("match_percentage", 85),
                        "reasons": item.get("reasons", [])
                    }
                ))
        else:
            # Fallback to demo data
            import random
            all_items = list(state.item_catalog.keys())
            available_items = [i for i in all_items if i not in request.exclude_items]
            
            selected_items = random.sample(
                available_items, 
                min(request.n_recommendations, len(available_items))
            )
            
            recommendations = []
            for i, item_id in enumerate(selected_items):
                item_data = state.item_catalog.get(item_id, {})
                score = 1.0 - (i * 0.05)
                
                recommendations.append(RecommendationItem(
                    item_id=item_id,
                    score=score,
                    explanation=f"Recommended based on your interest in {item_data.get('genre', 'content')}",
                    metadata={
                        "title": item_data.get("title"),
                        "genre": item_data.get("genre")
                    }
                ))
        
        latency_ms = (time.time() - start_time) * 1000
        
        # Record metrics (if prometheus available)
        if PROMETHEUS_AVAILABLE and REQUEST_COUNT:
            REQUEST_COUNT.labels(endpoint="recommendations", status="success").inc()
            REQUEST_LATENCY.labels(endpoint="recommendations").observe(latency_ms / 1000)
        
        return RecommendationResponse(
            user_id=user_id,
            recommendations=recommendations,
            model_version=state.model_version,
            latency_ms=latency_ms
        )
    
    except Exception as e:
        if PROMETHEUS_AVAILABLE and REQUEST_COUNT:
            REQUEST_COUNT.labels(endpoint="recommendations", status="error").inc()
        logger.error(f"Recommendation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/users", status_code=201)
async def create_user(
    profile: UserProfile,
    state: AppState = Depends(get_state)
):
    """
    Create or update user profile for cold start handling.
    """
    try:
        # In production, save to database and update model
        logger.info(f"Created/updated user profile: {profile.user_id}")
        
        return {
            "status": "success",
            "user_id": profile.user_id,
            "message": "User profile created/updated successfully"
        }
    
    except Exception as e:
        logger.error(f"User creation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/interactions")
async def log_interaction(
    interaction: UserInteraction,
    background_tasks: BackgroundTasks,
    state: AppState = Depends(get_state)
):
    """
    Log user interaction for model updates.
    
    Interactions are processed asynchronously to update recommendations.
    """
    try:
        # Add to background processing queue
        background_tasks.add_task(
            process_interaction,
            interaction
        )
        
        if PROMETHEUS_AVAILABLE and REQUEST_COUNT:
            REQUEST_COUNT.labels(endpoint="interactions", status="success").inc()
        
        return {
            "status": "accepted",
            "message": "Interaction logged for processing"
        }
    
    except Exception as e:
        if PROMETHEUS_AVAILABLE and REQUEST_COUNT:
            REQUEST_COUNT.labels(endpoint="interactions", status="error").inc()
        logger.error(f"Interaction logging error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/items/{item_id}", response_model=ItemInfo)
async def get_item(
    item_id: int,
    state: AppState = Depends(get_state)
):
    """Get item information."""
    # Try media database first
    if MEDIA_DB_AVAILABLE:
        item_data = get_media_by_id(item_id)
        if item_data:
            return ItemInfo(
                item_id=item_id,
                title=item_data.get("title", ""),
                description=item_data.get("description"),
                genre=", ".join(item_data.get("genres", [])),
                tags=item_data.get("genres", []),
                popularity_score=item_data.get("popularity", 0.0)
            )
    
    # Fallback to state catalog
    item_data = state.item_catalog.get(item_id)
    
    if not item_data:
        raise HTTPException(status_code=404, detail="Item not found")
    
    return ItemInfo(
        item_id=item_id,
        title=item_data.get("title", ""),
        description=item_data.get("description"),
        genre=item_data.get("genre"),
        tags=item_data.get("tags", []),
        popularity_score=item_data.get("popularity_score", 0.0)
    )


@app.get("/api/v1/items/{item_id}/similar")
async def get_similar_items(
    item_id: int,
    n: int = Query(default=10, ge=1, le=50),
    state: AppState = Depends(get_state)
):
    """Get similar items based on content."""
    # Try media database first
    if MEDIA_DB_AVAILABLE:
        similar = get_similar_media(item_id, n)
        if similar:
            return {
                "source_item_id": item_id,
                "similar_items": [
                    {
                        "item_id": item["id"],
                        "score": item.get("similarity", 0.9),
                        "title": item["title"],
                        "year": item["year"],
                        "genres": item["genres"],
                        "rating": item["rating"],
                        "poster": item["poster"],
                        "description": item["description"]
                    }
                    for item in similar
                ]
            }
    
    # Fallback
    if item_id not in state.item_catalog:
        raise HTTPException(status_code=404, detail="Item not found")
    
    source_genre = state.item_catalog[item_id].get("genre")
    similar_items = [
        {"item_id": i, "score": 0.9 - (idx * 0.05)}
        for idx, (i, data) in enumerate(state.item_catalog.items())
        if data.get("genre") == source_genre and i != item_id
    ][:n]
    
    return {
        "source_item_id": item_id,
        "similar_items": similar_items
    }


@app.get("/api/v1/popular")
async def get_popular_items(
    n: int = Query(default=10, ge=1, le=100),
    genre: Optional[str] = None,
    state: AppState = Depends(get_state)
):
    """Get popular items, optionally filtered by genre."""
    # Try media database first
    if MEDIA_DB_AVAILABLE:
        if genre:
            items = get_media_by_genre(genre)[:n]
        else:
            items = get_trending(n)
        
        return {
            "items": [
                {
                    "item_id": item["id"],
                    "title": item["title"],
                    "year": item["year"],
                    "genres": item["genres"],
                    "rating": item["rating"],
                    "poster": item["poster"],
                    "backdrop": item.get("backdrop", ""),
                    "description": item["description"],
                    "popularity_score": item["popularity"],
                    "media_type": item["media_type"]
                }
                for item in items
            ]
        }
    
    # Fallback
    items = list(state.item_catalog.items())
    
    if genre:
        items = [(i, d) for i, d in items if d.get("genre") == genre]
    
    popular = [
        {"item_id": i, "score": 1.0 - (idx * 0.01), **d}
        for idx, (i, d) in enumerate(items[:n])
    ]
    
    return {"items": popular}


@app.get("/api/v1/trending")
async def get_trending_items(
    n: int = Query(default=10, ge=1, le=50)
):
    """Get trending items."""
    if MEDIA_DB_AVAILABLE:
        items = get_trending(n)
        return {
            "items": [
                {
                    "item_id": item["id"],
                    "title": item["title"],
                    "year": item["year"],
                    "genres": item["genres"],
                    "rating": item["rating"],
                    "poster": item["poster"],
                    "backdrop": item.get("backdrop", ""),
                    "description": item["description"],
                    "popularity_score": item["popularity"],
                    "media_type": item["media_type"]
                }
                for item in items
            ]
        }
    return {"items": []}


@app.get("/api/v1/top-rated")
async def get_top_rated_items(
    n: int = Query(default=10, ge=1, le=50)
):
    """Get top rated items."""
    if MEDIA_DB_AVAILABLE:
        items = get_top_rated(n)
        return {
            "items": [
                {
                    "item_id": item["id"],
                    "title": item["title"],
                    "year": item["year"],
                    "genres": item["genres"],
                    "rating": item["rating"],
                    "poster": item["poster"],
                    "backdrop": item.get("backdrop", ""),
                    "description": item["description"],
                    "popularity_score": item["popularity"],
                    "media_type": item["media_type"]
                }
                for item in items
            ]
        }
    return {"items": []}


@app.get("/api/v1/genres")
async def get_genres():
    """Get all available genres."""
    if MEDIA_DB_AVAILABLE:
        genres = set()
        for item in MEDIA_DATABASE:
            genres.update(item.get("genres", []))
        return {"genres": sorted(list(genres))}
    return {"genres": []}


@app.get("/api/v1/all-media")
async def get_all_media_items(
    limit: int = Query(default=100, ge=1, le=200),
    offset: int = Query(default=0, ge=0)
):
    """Get all media items with pagination."""
    if MEDIA_DB_AVAILABLE:
        items = MEDIA_DATABASE[offset:offset + limit]
        return {
            "total": len(MEDIA_DATABASE),
            "limit": limit,
            "offset": offset,
            "items": [
                {
                    "item_id": item["id"],
                    "title": item["title"],
                    "year": item["year"],
                    "genres": item["genres"],
                    "rating": item["rating"],
                    "poster": item["poster"],
                    "backdrop": item.get("backdrop", ""),
                    "description": item["description"],
                    "duration": item["duration"],
                    "director": item["director"],
                    "cast": item["cast"],
                    "popularity_score": item["popularity"],
                    "media_type": item["media_type"]
                }
                for item in items
            ]
        }
    return {"total": 0, "items": []}


@app.get("/api/v1/users/{user_id}/profile")
async def get_user_profile_endpoint(user_id: int):
    """Get user profile with preferences and viewing history."""
    if MEDIA_DB_AVAILABLE:
        try:
            profile = get_user_profile_summary(user_id)
            return profile
        except Exception as e:
            logger.error(f"Error getting user profile: {e}")
    
    # Fallback profile
    return {
        "user_id": user_id,
        "name": f"User #{user_id}",
        "type": "general",
        "type_label": "General Viewer",
        "preferred_genres": ["Drama", "Action"],
        "disliked_genres": [],
        "min_rating": 7.0,
        "watched_count": 0,
        "liked_count": 0
    }


@app.get("/api/v1/users/profiles")
async def list_user_profiles():
    """List all available demo user profiles."""
    if MEDIA_DB_AVAILABLE and USER_PROFILES:
        profiles = []
        for uid, profile in USER_PROFILES.items():
            profiles.append({
                "user_id": uid,
                "name": profile.name,
                "type": profile.user_type.value,
                "type_label": profile.user_type.value.replace("_", " ").title(),
                "preferred_genres": profile.preferred_genres,
                "description": f"Prefers {', '.join(profile.preferred_genres[:2])} content"
            })
        return {"profiles": profiles}
    return {"profiles": []}


@app.get("/api/v1/explain/{user_id}/{item_id}")
async def explain_recommendation(user_id: int, item_id: int):
    """Get detailed explanation for why an item was recommended to a user."""
    if MEDIA_DB_AVAILABLE:
        try:
            explanation = get_recommendation_explanation(user_id, item_id)
            return explanation
        except Exception as e:
            logger.error(f"Error getting explanation: {e}")
    
    return {
        "error": "Explanation not available",
        "user_id": user_id,
        "item_id": item_id
    }


# Background task
async def process_interaction(interaction: UserInteraction):
    """Process interaction asynchronously."""
    try:
        # In production:
        # 1. Store in database
        # 2. Update real-time features
        # 3. Trigger model retraining if needed
        
        logger.info(
            f"Processed interaction: user={interaction.user_id}, "
            f"item={interaction.item_id}, type={interaction.interaction_type}"
        )
    except Exception as e:
        logger.error(f"Failed to process interaction: {e}")


# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        workers=4
    )
