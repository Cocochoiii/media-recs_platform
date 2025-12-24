"""Media Recommender System - Data Module"""

# Media database (no dependencies)
from .media_database import (
    MEDIA_DATABASE,
    get_all_media,
    get_media_by_id,
    get_media_by_genre,
    get_recommendations_for_user,
    get_similar_items,
    get_trending,
    get_top_rated
)

# User profiles (no dependencies)
from .user_profiles import (
    UserProfile,
    UserType,
    USER_PROFILES,
    get_user_profile,
    get_personalized_recommendations,
    get_user_profile_summary,
    get_recommendation_explanation
)

# Optional imports that require torch/numpy
try:
    from .dataset import *
    from .preprocessor import *
    from .feature_engineering import *
except ImportError:
    pass