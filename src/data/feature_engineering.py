"""
Feature Engineering Module

Comprehensive feature extraction and transformation for users and items.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class FeatureConfig:
    """Configuration for feature engineering."""
    # Time-based features
    time_bins: int = 24  # Hours of day
    day_bins: int = 7    # Days of week
    
    # Interaction features
    interaction_window_days: int = 30
    min_interactions: int = 5
    
    # Content features
    max_tags: int = 10
    max_genres: int = 5
    
    # Embedding dimensions
    user_feature_dim: int = 64
    item_feature_dim: int = 128


class UserFeatureExtractor:
    """
    Extract features from user behavior and demographics.
    """
    
    def __init__(self, config: FeatureConfig):
        self.config = config
    
    def extract_interaction_features(
        self,
        user_id: int,
        interactions: pd.DataFrame
    ) -> Dict[str, float]:
        """Extract features from user's interaction history."""
        user_interactions = interactions[interactions["user_id"] == user_id]
        
        if len(user_interactions) == 0:
            return self._get_default_features()
        
        features = {}
        
        # Basic statistics
        features["total_interactions"] = len(user_interactions)
        features["unique_items"] = user_interactions["item_id"].nunique()
        
        # Rating statistics (if available)
        if "rating" in user_interactions.columns:
            features["avg_rating"] = user_interactions["rating"].mean()
            features["rating_std"] = user_interactions["rating"].std()
            features["rating_range"] = (
                user_interactions["rating"].max() - 
                user_interactions["rating"].min()
            )
        
        # Temporal features
        if "timestamp" in user_interactions.columns:
            timestamps = pd.to_datetime(user_interactions["timestamp"], unit="s")
            
            # Activity patterns
            features["avg_hour"] = timestamps.dt.hour.mean() / 24
            features["weekend_ratio"] = (timestamps.dt.dayofweek >= 5).mean()
            
            # Recency
            latest = timestamps.max()
            features["days_since_last"] = (
                datetime.now() - latest
            ).days if pd.notna(latest) else 365
            
            # Session patterns
            time_diffs = timestamps.diff().dt.total_seconds() / 3600
            features["avg_session_gap"] = time_diffs.mean() if len(time_diffs) > 1 else 0
        
        # Interaction type distribution (if available)
        if "interaction_type" in user_interactions.columns:
            type_counts = user_interactions["interaction_type"].value_counts(normalize=True)
            for itype in ["view", "click", "purchase", "rating"]:
                features[f"{itype}_ratio"] = type_counts.get(itype, 0)
        
        return features
    
    def extract_demographic_features(
        self,
        demographics: Dict[str, Any]
    ) -> Dict[str, float]:
        """Extract features from user demographics."""
        features = {}
        
        # Age group encoding
        age_groups = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
        age_group = demographics.get("age_group", "unknown")
        for i, ag in enumerate(age_groups):
            features[f"age_{ag}"] = 1.0 if age_group == ag else 0.0
        
        # Gender encoding
        gender = demographics.get("gender", "unknown")
        features["gender_male"] = 1.0 if gender == "male" else 0.0
        features["gender_female"] = 1.0 if gender == "female" else 0.0
        
        # Location (simplified)
        features["has_location"] = 1.0 if demographics.get("location") else 0.0
        
        # Device type
        device = demographics.get("device_type", "unknown")
        for dtype in ["mobile", "desktop", "tablet"]:
            features[f"device_{dtype}"] = 1.0 if device == dtype else 0.0
        
        # Subscription tier
        tier = demographics.get("subscription_tier", "free")
        for t in ["free", "basic", "premium"]:
            features[f"tier_{t}"] = 1.0 if tier == t else 0.0
        
        return features
    
    def _get_default_features(self) -> Dict[str, float]:
        """Return default features for new users."""
        return {
            "total_interactions": 0,
            "unique_items": 0,
            "avg_rating": 3.0,
            "rating_std": 0.0,
            "days_since_last": 365,
        }
    
    def to_vector(self, features: Dict[str, float]) -> np.ndarray:
        """Convert feature dict to numpy vector."""
        # Ensure consistent ordering
        keys = sorted(features.keys())
        return np.array([features[k] for k in keys], dtype=np.float32)


class ItemFeatureExtractor:
    """
    Extract features from item content and metadata.
    """
    
    def __init__(self, config: FeatureConfig):
        self.config = config
        self.genre_encoder: Dict[str, int] = {}
        self.tag_encoder: Dict[str, int] = {}
    
    def fit(self, items: List[Dict[str, Any]]):
        """Learn encodings from item catalog."""
        # Collect all genres and tags
        all_genres = set()
        all_tags = set()
        
        for item in items:
            if "genre" in item:
                genres = item["genre"] if isinstance(item["genre"], list) else [item["genre"]]
                all_genres.update(genres)
            if "tags" in item:
                all_tags.update(item["tags"])
        
        # Create encodings
        self.genre_encoder = {g: i for i, g in enumerate(sorted(all_genres))}
        self.tag_encoder = {t: i for i, t in enumerate(sorted(all_tags))}
        
        logger.info(f"Fitted {len(self.genre_encoder)} genres, {len(self.tag_encoder)} tags")
    
    def extract_content_features(
        self,
        item: Dict[str, Any]
    ) -> Dict[str, float]:
        """Extract content-based features from item."""
        features = {}
        
        # Text length features
        title = item.get("title", "")
        description = item.get("description", "")
        
        features["title_length"] = len(title.split())
        features["description_length"] = len(description.split())
        features["has_description"] = 1.0 if description else 0.0
        
        # Duration (for videos)
        duration = item.get("duration", 0)
        features["duration_minutes"] = duration / 60 if duration else 0
        features["is_short"] = 1.0 if duration < 600 else 0.0  # < 10 min
        features["is_long"] = 1.0 if duration > 3600 else 0.0  # > 1 hour
        
        # Release date
        release_date = item.get("release_date")
        if release_date:
            try:
                release = pd.to_datetime(release_date)
                features["age_days"] = (datetime.now() - release).days
                features["is_new"] = 1.0 if features["age_days"] < 30 else 0.0
            except:
                features["age_days"] = 365
                features["is_new"] = 0.0
        
        return features
    
    def extract_categorical_features(
        self,
        item: Dict[str, Any]
    ) -> np.ndarray:
        """Extract one-hot encoded categorical features."""
        # Genre encoding
        genre_vector = np.zeros(len(self.genre_encoder))
        genres = item.get("genre", [])
        if isinstance(genres, str):
            genres = [genres]
        for genre in genres:
            if genre in self.genre_encoder:
                genre_vector[self.genre_encoder[genre]] = 1.0
        
        # Tag encoding (multi-hot)
        tag_vector = np.zeros(min(len(self.tag_encoder), self.config.max_tags))
        tags = item.get("tags", [])[:self.config.max_tags]
        for tag in tags:
            if tag in self.tag_encoder and self.tag_encoder[tag] < len(tag_vector):
                tag_vector[self.tag_encoder[tag]] = 1.0
        
        return np.concatenate([genre_vector, tag_vector])
    
    def extract_popularity_features(
        self,
        item_id: int,
        interactions: pd.DataFrame
    ) -> Dict[str, float]:
        """Extract popularity-based features."""
        item_interactions = interactions[interactions["item_id"] == item_id]
        
        features = {}
        features["interaction_count"] = len(item_interactions)
        features["unique_users"] = item_interactions["user_id"].nunique()
        
        if "rating" in item_interactions.columns and len(item_interactions) > 0:
            features["avg_rating"] = item_interactions["rating"].mean()
            features["rating_count"] = item_interactions["rating"].notna().sum()
        else:
            features["avg_rating"] = 0.0
            features["rating_count"] = 0
        
        # Trend (recent vs old interactions)
        if "timestamp" in item_interactions.columns and len(item_interactions) > 0:
            timestamps = pd.to_datetime(item_interactions["timestamp"], unit="s")
            recent_cutoff = datetime.now() - timedelta(days=7)
            recent_count = (timestamps > recent_cutoff).sum()
            features["recent_ratio"] = recent_count / len(item_interactions)
        else:
            features["recent_ratio"] = 0.0
        
        return features


class FeatureStore:
    """
    Feature store for caching and serving features.
    """
    
    def __init__(self):
        self.user_features: Dict[int, np.ndarray] = {}
        self.item_features: Dict[int, np.ndarray] = {}
        self.feature_metadata: Dict[str, Any] = {}
    
    def update_user_features(self, user_id: int, features: np.ndarray):
        """Update cached user features."""
        self.user_features[user_id] = features
    
    def update_item_features(self, item_id: int, features: np.ndarray):
        """Update cached item features."""
        self.item_features[item_id] = features
    
    def get_user_features(self, user_id: int) -> Optional[np.ndarray]:
        """Get cached user features."""
        return self.user_features.get(user_id)
    
    def get_item_features(self, item_id: int) -> Optional[np.ndarray]:
        """Get cached item features."""
        return self.item_features.get(item_id)
    
    def batch_get_item_features(
        self,
        item_ids: List[int]
    ) -> np.ndarray:
        """Get features for multiple items."""
        features = []
        for item_id in item_ids:
            feat = self.item_features.get(item_id)
            if feat is not None:
                features.append(feat)
            else:
                # Return zeros for missing items
                features.append(np.zeros(self.feature_metadata.get("item_dim", 128)))
        return np.array(features)
    
    def save(self, path: str):
        """Save feature store to disk."""
        np.savez(
            path,
            user_features=dict(self.user_features),
            item_features=dict(self.item_features),
            metadata=self.feature_metadata
        )
    
    def load(self, path: str):
        """Load feature store from disk."""
        data = np.load(path, allow_pickle=True)
        self.user_features = dict(data["user_features"].item())
        self.item_features = dict(data["item_features"].item())
        self.feature_metadata = dict(data["metadata"].item())


class FeatureEngineeringPipeline:
    """
    Complete feature engineering pipeline.
    """
    
    def __init__(self, config: FeatureConfig):
        self.config = config
        self.user_extractor = UserFeatureExtractor(config)
        self.item_extractor = ItemFeatureExtractor(config)
        self.feature_store = FeatureStore()
    
    def fit(
        self,
        items: List[Dict[str, Any]],
        interactions: pd.DataFrame
    ):
        """Fit the pipeline on training data."""
        logger.info("Fitting feature engineering pipeline...")
        
        # Fit item extractor
        self.item_extractor.fit(items)
        
        # Compute and cache item features
        for item in items:
            item_id = item["id"]
            content_features = self.item_extractor.extract_content_features(item)
            categorical_features = self.item_extractor.extract_categorical_features(item)
            popularity_features = self.item_extractor.extract_popularity_features(
                item_id, interactions
            )
            
            # Combine all features
            all_features = {**content_features, **popularity_features}
            numeric_vector = self.user_extractor.to_vector(all_features)
            combined = np.concatenate([numeric_vector, categorical_features])
            
            self.feature_store.update_item_features(item_id, combined)
        
        # Store metadata
        self.feature_store.feature_metadata["item_dim"] = len(combined)
        
        logger.info(f"Processed {len(items)} items with {len(combined)} features each")
    
    def transform_user(
        self,
        user_id: int,
        interactions: pd.DataFrame,
        demographics: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """Transform user data to feature vector."""
        interaction_features = self.user_extractor.extract_interaction_features(
            user_id, interactions
        )
        
        if demographics:
            demographic_features = self.user_extractor.extract_demographic_features(
                demographics
            )
            all_features = {**interaction_features, **demographic_features}
        else:
            all_features = interaction_features
        
        vector = self.user_extractor.to_vector(all_features)
        self.feature_store.update_user_features(user_id, vector)
        
        return vector
    
    def get_user_item_features(
        self,
        user_id: int,
        item_ids: List[int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get features for user-item pairs."""
        user_features = self.feature_store.get_user_features(user_id)
        item_features = self.feature_store.batch_get_item_features(item_ids)
        
        return user_features, item_features


if __name__ == "__main__":
    # Example usage
    config = FeatureConfig()
    pipeline = FeatureEngineeringPipeline(config)
    
    # Sample data
    items = [
        {"id": 1, "title": "Movie A", "genre": "Action", "tags": ["exciting"], "duration": 7200},
        {"id": 2, "title": "Movie B", "genre": "Comedy", "tags": ["funny"], "duration": 5400},
    ]
    
    interactions = pd.DataFrame({
        "user_id": [1, 1, 2],
        "item_id": [1, 2, 1],
        "rating": [5, 4, 3],
        "timestamp": [1000000, 1000100, 1000200]
    })
    
    pipeline.fit(items, interactions)
    
    user_features = pipeline.transform_user(1, interactions, {"age_group": "25-34"})
    print(f"User features shape: {user_features.shape}")
