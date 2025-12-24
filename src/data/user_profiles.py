"""
Intelligent User Profile System for Personalized Recommendations

This module simulates a production ML recommendation system with:
- User profiles with preferences, viewing history, and demographics
- Content-based filtering using genre/feature matching
- Collaborative filtering simulation
- Explainable AI - shows why each item was recommended
"""

import random
import math
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

# Import media database
try:
    from src.data.media_database import MEDIA_DATABASE, get_media_by_id
except ImportError:
    try:
        from .media_database import MEDIA_DATABASE, get_media_by_id
    except ImportError:
        # For standalone testing
        import os
        import sys
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from media_database import MEDIA_DATABASE, get_media_by_id


class UserType(Enum):
    """User persona types for demonstration."""
    ACTION_LOVER = "action_lover"
    SCI_FI_GEEK = "sci_fi_geek"
    DRAMA_FAN = "drama_fan"
    HORROR_ENTHUSIAST = "horror_enthusiast"
    COMEDY_LOVER = "comedy_lover"
    ROMANCE_SEEKER = "romance_seeker"
    ANIME_OTAKU = "anime_otaku"
    CLASSIC_CINEPHILE = "classic_cinephile"
    TV_BINGER = "tv_binger"
    FAMILY_VIEWER = "family_viewer"


@dataclass
class UserProfile:
    """User profile with preferences and history."""
    user_id: int
    name: str
    user_type: UserType
    preferred_genres: List[str]
    disliked_genres: List[str]
    preferred_years: Tuple[int, int]  # (min_year, max_year)
    min_rating: float
    watched_ids: List[int]
    liked_ids: List[int]
    demographics: Dict[str, Any]
    
    # ML feature weights (simulating learned preferences)
    genre_weights: Dict[str, float] = field(default_factory=dict)
    director_preferences: List[str] = field(default_factory=list)
    actor_preferences: List[str] = field(default_factory=list)


# Predefined user profiles for demonstration
USER_PROFILES: Dict[int, UserProfile] = {
    # User 1: Action Movie Lover
    1: UserProfile(
        user_id=1,
        name="Alex Chen",
        user_type=UserType.ACTION_LOVER,
        preferred_genres=["Action", "Thriller", "Crime"],
        disliked_genres=["Romance", "Musical"],
        preferred_years=(2000, 2024),
        min_rating=7.0,
        watched_ids=[1, 2, 3, 4, 5, 46, 47],
        liked_ids=[1, 2, 4, 47],
        demographics={"age": 28, "gender": "M", "country": "US"},
        genre_weights={"Action": 0.95, "Thriller": 0.85, "Crime": 0.80, "Sci-Fi": 0.70},
        director_preferences=["Christopher Nolan", "Chad Stahelski", "George Miller"],
        actor_preferences=["Keanu Reeves", "Tom Cruise", "Christian Bale"]
    ),
    
    # User 2: Sci-Fi & Tech Enthusiast
    2: UserProfile(
        user_id=2,
        name="Sarah Kim",
        user_type=UserType.SCI_FI_GEEK,
        preferred_genres=["Sci-Fi", "Mystery", "Thriller"],
        disliked_genres=["Horror", "War"],
        preferred_years=(1990, 2024),
        min_rating=7.5,
        watched_ids=[2, 6, 7, 8, 9, 10],
        liked_ids=[2, 6, 7, 8, 9],
        demographics={"age": 32, "gender": "F", "country": "KR"},
        genre_weights={"Sci-Fi": 0.98, "Mystery": 0.85, "Thriller": 0.75, "Drama": 0.60},
        director_preferences=["Denis Villeneuve", "Christopher Nolan", "Ridley Scott"],
        actor_preferences=["Ryan Gosling", "Timothée Chalamet", "Amy Adams"]
    ),
    
    # User 3: Drama & Award Films Lover
    3: UserProfile(
        user_id=3,
        name="Michael Brown",
        user_type=UserType.DRAMA_FAN,
        preferred_genres=["Drama", "Biography", "History"],
        disliked_genres=["Horror", "Animation"],
        preferred_years=(1970, 2024),
        min_rating=8.0,
        watched_ids=[11, 12, 13, 14, 15, 51, 52],
        liked_ids=[11, 13, 14, 15],
        demographics={"age": 45, "gender": "M", "country": "UK"},
        genre_weights={"Drama": 0.95, "Biography": 0.85, "History": 0.80, "Crime": 0.70},
        director_preferences=["Martin Scorsese", "Steven Spielberg", "Francis Ford Coppola"],
        actor_preferences=["Leonardo DiCaprio", "Tom Hanks", "Morgan Freeman"]
    ),
    
    # User 4: Horror Fan
    4: UserProfile(
        user_id=4,
        name="Emma Wilson",
        user_type=UserType.HORROR_ENTHUSIAST,
        preferred_genres=["Horror", "Thriller", "Mystery"],
        disliked_genres=["Romance", "Family", "Animation"],
        preferred_years=(2010, 2024),
        min_rating=6.5,
        watched_ids=[21, 22, 23, 24, 25],
        liked_ids=[21, 22, 23],
        demographics={"age": 25, "gender": "F", "country": "US"},
        genre_weights={"Horror": 0.98, "Thriller": 0.85, "Mystery": 0.80, "Sci-Fi": 0.50},
        director_preferences=["Jordan Peele", "Ari Aster", "James Wan"],
        actor_preferences=["Toni Collette", "Florence Pugh", "Daniel Kaluuya"]
    ),
    
    # User 5: Comedy Enthusiast
    5: UserProfile(
        user_id=5,
        name="David Garcia",
        user_type=UserType.COMEDY_LOVER,
        preferred_genres=["Comedy", "Adventure", "Family"],
        disliked_genres=["Horror", "War"],
        preferred_years=(2000, 2024),
        min_rating=7.0,
        watched_ids=[26, 27, 28, 29, 30, 44],
        liked_ids=[26, 28, 29, 44],
        demographics={"age": 35, "gender": "M", "country": "MX"},
        genre_weights={"Comedy": 0.95, "Adventure": 0.80, "Family": 0.75, "Animation": 0.70},
        director_preferences=["Wes Anderson", "Taika Waititi", "Edgar Wright"],
        actor_preferences=["Ryan Reynolds", "Steve Carell", "Melissa McCarthy"]
    ),
    
    # User 6: Romance & Drama Lover
    6: UserProfile(
        user_id=6,
        name="Sophie Martin",
        user_type=UserType.ROMANCE_SEEKER,
        preferred_genres=["Romance", "Drama", "Music"],
        disliked_genres=["Horror", "Action", "War"],
        preferred_years=(1995, 2024),
        min_rating=7.0,
        watched_ids=[36, 37, 38, 39, 40],
        liked_ids=[36, 37, 39],
        demographics={"age": 29, "gender": "F", "country": "FR"},
        genre_weights={"Romance": 0.95, "Drama": 0.90, "Music": 0.80, "Comedy": 0.60},
        director_preferences=["Damien Chazelle", "Richard Linklater", "Nora Ephron"],
        actor_preferences=["Ryan Gosling", "Emma Stone", "Rachel McAdams"]
    ),
    
    # User 7: Animation & Anime Fan
    7: UserProfile(
        user_id=7,
        name="Yuki Tanaka",
        user_type=UserType.ANIME_OTAKU,
        preferred_genres=["Animation", "Fantasy", "Adventure"],
        disliked_genres=["Horror", "War"],
        preferred_years=(1990, 2024),
        min_rating=7.5,
        watched_ids=[31, 32, 33, 34, 35],
        liked_ids=[31, 33, 34],
        demographics={"age": 24, "gender": "F", "country": "JP"},
        genre_weights={"Animation": 0.98, "Fantasy": 0.90, "Adventure": 0.85, "Family": 0.70},
        director_preferences=["Hayao Miyazaki", "Makoto Shinkai", "Pete Docter"],
        actor_preferences=[]
    ),
    
    # User 8: Classic Film Buff
    8: UserProfile(
        user_id=8,
        name="Robert Taylor",
        user_type=UserType.CLASSIC_CINEPHILE,
        preferred_genres=["Drama", "Crime", "Mystery"],
        disliked_genres=["Animation", "Family"],
        preferred_years=(1970, 2010),
        min_rating=8.0,
        watched_ids=[13, 55, 54, 56, 57],
        liked_ids=[13, 55, 54, 56],
        demographics={"age": 58, "gender": "M", "country": "US"},
        genre_weights={"Drama": 0.90, "Crime": 0.88, "Mystery": 0.85, "Thriller": 0.80},
        director_preferences=["Francis Ford Coppola", "Quentin Tarantino", "Martin Scorsese"],
        actor_preferences=["Al Pacino", "Robert De Niro", "Marlon Brando"]
    ),
    
    # User 9: TV Series Binger
    9: UserProfile(
        user_id=9,
        name="Jessica Lee",
        user_type=UserType.TV_BINGER,
        preferred_genres=["Drama", "Fantasy", "Comedy"],
        disliked_genres=["Documentary"],
        preferred_years=(2005, 2024),
        min_rating=8.0,
        watched_ids=[41, 42, 43, 44, 45],
        liked_ids=[41, 42, 43, 44],
        demographics={"age": 27, "gender": "F", "country": "US"},
        genre_weights={"Drama": 0.90, "Fantasy": 0.85, "Comedy": 0.80, "Crime": 0.75},
        director_preferences=["Vince Gilligan", "The Duffer Brothers"],
        actor_preferences=["Bryan Cranston", "Peter Dinklage", "Steve Carell"]
    ),
    
    # User 10: Family-Friendly Viewer
    10: UserProfile(
        user_id=10,
        name="Jennifer Adams",
        user_type=UserType.FAMILY_VIEWER,
        preferred_genres=["Animation", "Family", "Adventure", "Comedy"],
        disliked_genres=["Horror", "Crime", "Thriller"],
        preferred_years=(2000, 2024),
        min_rating=7.0,
        watched_ids=[32, 34, 35, 31],
        liked_ids=[32, 34, 35],
        demographics={"age": 38, "gender": "F", "country": "US", "has_children": True},
        genre_weights={"Animation": 0.95, "Family": 0.95, "Adventure": 0.85, "Comedy": 0.80},
        director_preferences=["Pete Docter", "Lee Unkrich", "Brad Bird"],
        actor_preferences=["Tom Hanks", "Ellen DeGeneres"]
    ),
}


def get_user_profile(user_id: int) -> UserProfile:
    """Get user profile by ID, or generate one for unknown users."""
    if user_id in USER_PROFILES:
        return USER_PROFILES[user_id]
    
    # Generate a random profile for unknown users
    user_types = list(UserType)
    random.seed(user_id)  # Consistent for same user_id
    user_type = random.choice(user_types)
    
    # Genre mappings for each user type
    genre_mappings = {
        UserType.ACTION_LOVER: (["Action", "Thriller", "Crime"], ["Romance", "Musical"]),
        UserType.SCI_FI_GEEK: (["Sci-Fi", "Mystery", "Thriller"], ["Horror", "Romance"]),
        UserType.DRAMA_FAN: (["Drama", "Biography", "History"], ["Horror", "Animation"]),
        UserType.HORROR_ENTHUSIAST: (["Horror", "Thriller", "Mystery"], ["Romance", "Family"]),
        UserType.COMEDY_LOVER: (["Comedy", "Adventure", "Family"], ["Horror", "War"]),
        UserType.ROMANCE_SEEKER: (["Romance", "Drama", "Music"], ["Horror", "Action"]),
        UserType.ANIME_OTAKU: (["Animation", "Fantasy", "Adventure"], ["Horror", "War"]),
        UserType.CLASSIC_CINEPHILE: (["Drama", "Crime", "Mystery"], ["Animation", "Family"]),
        UserType.TV_BINGER: (["Drama", "Fantasy", "Comedy"], ["Documentary"]),
        UserType.FAMILY_VIEWER: (["Animation", "Family", "Adventure"], ["Horror", "Crime"]),
    }
    
    preferred, disliked = genre_mappings.get(user_type, (["Drama"], ["Horror"]))
    
    names = ["User", "Guest", "Viewer", "Member", "Subscriber"]
    
    return UserProfile(
        user_id=user_id,
        name=f"{random.choice(names)} #{user_id}",
        user_type=user_type,
        preferred_genres=preferred,
        disliked_genres=disliked,
        preferred_years=(1990, 2024),
        min_rating=7.0,
        watched_ids=[],
        liked_ids=[],
        demographics={"age": random.randint(18, 60)},
        genre_weights={g: 0.9 - i * 0.1 for i, g in enumerate(preferred)}
    )


def calculate_recommendation_score(
    item: Dict[str, Any],
    user: UserProfile
) -> Tuple[float, List[str]]:
    """
    Calculate recommendation score for an item based on user profile.
    
    Returns (score, list of reasons)
    """
    score = 0.0
    reasons = []
    
    item_genres = set(item.get("genres", []))
    preferred_genres = set(user.preferred_genres)
    disliked_genres = set(user.disliked_genres)
    
    # 1. Genre matching (40% weight)
    genre_overlap = item_genres & preferred_genres
    if genre_overlap:
        genre_score = len(genre_overlap) / len(preferred_genres) * 0.4
        score += genre_score
        reasons.append(f"Matches your favorite genres: {', '.join(genre_overlap)}")
    
    # Penalty for disliked genres
    disliked_overlap = item_genres & disliked_genres
    if disliked_overlap:
        score -= 0.2
    
    # 2. Rating quality (20% weight)
    item_rating = item.get("rating", 7.0)
    if item_rating >= user.min_rating:
        rating_score = (item_rating - 5) / 5 * 0.2
        score += rating_score
        if item_rating >= 8.5:
            reasons.append(f"Highly rated ({item_rating}/10)")
    
    # 3. Year preference (15% weight)
    item_year = item.get("year", 2020)
    min_year, max_year = user.preferred_years
    if min_year <= item_year <= max_year:
        year_score = 0.15
        score += year_score
    else:
        score -= 0.1
    
    # 4. Director matching (10% weight)
    item_director = item.get("director", "")
    if item_director in user.director_preferences:
        score += 0.1
        reasons.append(f"Directed by {item_director} (one of your favorites)")
    
    # 5. Actor matching (10% weight)
    item_cast = set(item.get("cast", []))
    matching_actors = item_cast & set(user.actor_preferences)
    if matching_actors:
        score += 0.1
        reasons.append(f"Stars {', '.join(list(matching_actors)[:2])}")
    
    # 6. Collaborative filtering simulation (5% weight)
    # Items liked by similar users
    if item.get("popularity", 0) > 85:
        score += 0.05
        reasons.append("Popular among similar users")
    
    # 7. Not already watched bonus
    if item["id"] not in user.watched_ids:
        score += 0.05
    else:
        score -= 0.3  # Big penalty for already watched
        reasons = ["Already watched"]
    
    # Normalize score to 0-1
    score = max(0, min(1, score))
    
    # Add default reason if none
    if not reasons:
        reasons.append("Recommended based on your viewing history")
    
    return score, reasons


def get_personalized_recommendations(
    user_id: int,
    n: int = 10,
    exclude_ids: Optional[List[int]] = None
) -> List[Dict[str, Any]]:
    """
    Get personalized recommendations for a user using ML-based scoring.
    """
    user = get_user_profile(user_id)
    exclude_ids = exclude_ids or []
    
    # Score all items
    scored_items = []
    for item in MEDIA_DATABASE:
        if item["id"] in exclude_ids:
            continue
        
        score, reasons = calculate_recommendation_score(item, user)
        
        # Only include items with positive scores
        if score > 0.2:
            item_copy = item.copy()
            item_copy["score"] = score
            item_copy["reasons"] = reasons
            item_copy["match_percentage"] = int(score * 100)
            scored_items.append(item_copy)
    
    # Sort by score descending
    scored_items.sort(key=lambda x: x["score"], reverse=True)
    
    return scored_items[:n]


def get_user_profile_summary(user_id: int) -> Dict[str, Any]:
    """Get a summary of user profile for display."""
    user = get_user_profile(user_id)
    
    return {
        "user_id": user.user_id,
        "name": user.name,
        "type": user.user_type.value,
        "type_label": user.user_type.value.replace("_", " ").title(),
        "preferred_genres": user.preferred_genres,
        "disliked_genres": user.disliked_genres,
        "min_rating": user.min_rating,
        "watched_count": len(user.watched_ids),
        "liked_count": len(user.liked_ids),
        "favorite_directors": user.director_preferences[:3],
        "favorite_actors": user.actor_preferences[:3],
        "demographics": user.demographics
    }


def get_recommendation_explanation(
    user_id: int,
    item_id: int
) -> Dict[str, Any]:
    """Get detailed explanation for why an item was recommended."""
    user = get_user_profile(user_id)
    item = get_media_by_id(item_id)
    
    if not item:
        return {"error": "Item not found"}
    
    score, reasons = calculate_recommendation_score(item, user)
    
    # Calculate feature contributions
    item_genres = set(item.get("genres", []))
    preferred_genres = set(user.preferred_genres)
    
    explanation = {
        "item": {
            "id": item["id"],
            "title": item["title"],
            "genres": item["genres"],
            "rating": item["rating"],
            "year": item["year"]
        },
        "user": {
            "id": user.user_id,
            "name": user.name,
            "type": user.user_type.value
        },
        "overall_score": score,
        "match_percentage": int(score * 100),
        "reasons": reasons,
        "feature_breakdown": {
            "genre_match": {
                "weight": "40%",
                "matched": list(item_genres & preferred_genres),
                "user_preferences": user.preferred_genres
            },
            "rating_quality": {
                "weight": "20%",
                "item_rating": item["rating"],
                "user_minimum": user.min_rating,
                "passed": item["rating"] >= user.min_rating
            },
            "director_match": {
                "weight": "10%",
                "item_director": item.get("director"),
                "user_favorites": user.director_preferences,
                "matched": item.get("director") in user.director_preferences
            },
            "collaborative_signal": {
                "weight": "5%",
                "popularity": item.get("popularity", 0),
                "similar_users_liked": item.get("popularity", 0) > 85
            }
        },
        "ml_model_info": {
            "algorithm": "Hybrid Recommendation",
            "components": [
                "Content-Based Filtering (genre, director, actor matching)",
                "Collaborative Filtering (similar user preferences)",
                "Quality Filtering (rating threshold)",
                "Temporal Filtering (release year preference)"
            ]
        }
    }
    
    return explanation


# Export
__all__ = [
    "UserProfile",
    "UserType", 
    "USER_PROFILES",
    "get_user_profile",
    "get_personalized_recommendations",
    "get_user_profile_summary",
    "get_recommendation_explanation",
    "calculate_recommendation_score"
]
