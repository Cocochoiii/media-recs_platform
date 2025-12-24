"""
Data Preprocessing Module

Handles data cleaning, normalization, and transformation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import re
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
import json

logger = logging.getLogger(__name__)


@dataclass
class PreprocessConfig:
    """Configuration for data preprocessing."""
    # Missing value handling
    fill_missing_numeric: str = "median"  # mean, median, zero
    fill_missing_categorical: str = "unknown"
    
    # Outlier handling
    remove_outliers: bool = True
    outlier_std_threshold: float = 3.0
    
    # Text preprocessing
    lowercase: bool = True
    remove_punctuation: bool = True
    remove_numbers: bool = False
    
    # Scaling
    scaling_method: str = "standard"  # standard, minmax, none
    
    # Encoding
    max_categories: int = 100  # For high-cardinality features


class TextPreprocessor:
    """Text cleaning and normalization."""
    
    def __init__(self, config: PreprocessConfig):
        self.config = config
    
    def clean(self, text: str) -> str:
        """Clean and normalize text."""
        if not isinstance(text, str):
            return ""
        
        # Lowercase
        if self.config.lowercase:
            text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove punctuation
        if self.config.remove_punctuation:
            text = re.sub(r'[^\w\s]', ' ', text)
        
        # Remove numbers
        if self.config.remove_numbers:
            text = re.sub(r'\d+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def batch_clean(self, texts: List[str]) -> List[str]:
        """Clean multiple texts."""
        return [self.clean(t) for t in texts]


class NumericPreprocessor:
    """Numeric data preprocessing."""
    
    def __init__(self, config: PreprocessConfig):
        self.config = config
        self.scalers: Dict[str, Union[StandardScaler, MinMaxScaler]] = {}
        self.statistics: Dict[str, Dict[str, float]] = {}
    
    def fit(self, df: pd.DataFrame, columns: List[str]):
        """Fit scalers on training data."""
        for col in columns:
            if col not in df.columns:
                continue
            
            values = df[col].dropna()
            
            # Store statistics
            self.statistics[col] = {
                "mean": values.mean(),
                "median": values.median(),
                "std": values.std(),
                "min": values.min(),
                "max": values.max()
            }
            
            # Fit scaler
            if self.config.scaling_method == "standard":
                self.scalers[col] = StandardScaler()
            elif self.config.scaling_method == "minmax":
                self.scalers[col] = MinMaxScaler()
            
            if col in self.scalers:
                self.scalers[col].fit(values.values.reshape(-1, 1))
    
    def transform(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Transform numeric columns."""
        df = df.copy()
        
        for col in columns:
            if col not in df.columns:
                continue
            
            # Fill missing values
            if self.config.fill_missing_numeric == "mean":
                fill_value = self.statistics.get(col, {}).get("mean", 0)
            elif self.config.fill_missing_numeric == "median":
                fill_value = self.statistics.get(col, {}).get("median", 0)
            else:
                fill_value = 0
            
            df[col] = df[col].fillna(fill_value)
            
            # Remove outliers
            if self.config.remove_outliers and col in self.statistics:
                mean = self.statistics[col]["mean"]
                std = self.statistics[col]["std"]
                threshold = self.config.outlier_std_threshold
                
                lower = mean - threshold * std
                upper = mean + threshold * std
                df[col] = df[col].clip(lower, upper)
            
            # Scale
            if col in self.scalers:
                df[col] = self.scalers[col].transform(
                    df[col].values.reshape(-1, 1)
                ).flatten()
        
        return df
    
    def fit_transform(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Fit and transform in one step."""
        self.fit(df, columns)
        return self.transform(df, columns)


class CategoricalPreprocessor:
    """Categorical data preprocessing."""
    
    def __init__(self, config: PreprocessConfig):
        self.config = config
        self.encoders: Dict[str, LabelEncoder] = {}
        self.category_mappings: Dict[str, Dict[str, int]] = {}
    
    def fit(self, df: pd.DataFrame, columns: List[str]):
        """Fit encoders on training data."""
        for col in columns:
            if col not in df.columns:
                continue
            
            values = df[col].fillna(self.config.fill_missing_categorical).astype(str)
            
            # Handle high cardinality
            value_counts = values.value_counts()
            if len(value_counts) > self.config.max_categories:
                # Keep top categories, group rest as "other"
                top_categories = value_counts.head(self.config.max_categories - 1).index.tolist()
                values = values.apply(
                    lambda x: x if x in top_categories else "other"
                )
            
            # Create encoder
            encoder = LabelEncoder()
            encoder.fit(values)
            self.encoders[col] = encoder
            
            # Store mapping for reference
            self.category_mappings[col] = {
                cat: idx for idx, cat in enumerate(encoder.classes_)
            }
    
    def transform(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Transform categorical columns."""
        df = df.copy()
        
        for col in columns:
            if col not in df.columns or col not in self.encoders:
                continue
            
            values = df[col].fillna(self.config.fill_missing_categorical).astype(str)
            
            # Handle unseen categories
            known_categories = set(self.encoders[col].classes_)
            values = values.apply(
                lambda x: x if x in known_categories else "unknown"
            )
            
            # Add "unknown" to encoder if not present
            if "unknown" not in known_categories:
                # Use the most common category as fallback
                values = values.apply(
                    lambda x: x if x in known_categories else self.encoders[col].classes_[0]
                )
            
            df[col] = self.encoders[col].transform(values)
        
        return df
    
    def fit_transform(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Fit and transform in one step."""
        self.fit(df, columns)
        return self.transform(df, columns)
    
    def get_one_hot(
        self, 
        df: pd.DataFrame, 
        columns: List[str]
    ) -> pd.DataFrame:
        """Get one-hot encoded representation."""
        result_dfs = [df.drop(columns=columns, errors='ignore')]
        
        for col in columns:
            if col not in df.columns:
                continue
            
            dummies = pd.get_dummies(df[col], prefix=col)
            result_dfs.append(dummies)
        
        return pd.concat(result_dfs, axis=1)


class InteractionPreprocessor:
    """Preprocess user-item interaction data."""
    
    def __init__(self, config: PreprocessConfig):
        self.config = config
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
        self.fitted = False
    
    def fit(self, interactions: pd.DataFrame):
        """Fit on interaction data."""
        self.user_encoder.fit(interactions["user_id"])
        self.item_encoder.fit(interactions["item_id"])
        self.fitted = True
        
        logger.info(
            f"Fitted on {len(self.user_encoder.classes_)} users, "
            f"{len(self.item_encoder.classes_)} items"
        )
    
    def transform(self, interactions: pd.DataFrame) -> pd.DataFrame:
        """Transform interaction data."""
        if not self.fitted:
            raise RuntimeError("Must call fit() first")
        
        df = interactions.copy()
        
        # Encode IDs
        df["user_idx"] = self._safe_transform(
            df["user_id"], self.user_encoder
        )
        df["item_idx"] = self._safe_transform(
            df["item_id"], self.item_encoder
        )
        
        # Normalize ratings if present
        if "rating" in df.columns:
            # Scale to 0-1
            df["rating_norm"] = (df["rating"] - df["rating"].min()) / (
                df["rating"].max() - df["rating"].min() + 1e-8
            )
        
        # Process timestamps if present
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", errors="coerce")
            df["hour"] = df["timestamp"].dt.hour
            df["day_of_week"] = df["timestamp"].dt.dayofweek
            df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
        
        return df
    
    def _safe_transform(
        self, 
        values: pd.Series, 
        encoder: LabelEncoder
    ) -> np.ndarray:
        """Transform with handling for unseen values."""
        known = set(encoder.classes_)
        
        # Replace unknown values with first known value
        safe_values = values.apply(
            lambda x: x if x in known else encoder.classes_[0]
        )
        
        return encoder.transform(safe_values)
    
    def inverse_transform_user(self, idx: int) -> int:
        """Convert user index back to ID."""
        return self.user_encoder.inverse_transform([idx])[0]
    
    def inverse_transform_item(self, idx: int) -> int:
        """Convert item index back to ID."""
        return self.item_encoder.inverse_transform([idx])[0]
    
    @property
    def num_users(self) -> int:
        return len(self.user_encoder.classes_)
    
    @property
    def num_items(self) -> int:
        return len(self.item_encoder.classes_)


class DataPreprocessingPipeline:
    """
    Complete data preprocessing pipeline.
    """
    
    def __init__(self, config: Optional[PreprocessConfig] = None):
        self.config = config or PreprocessConfig()
        self.text_processor = TextPreprocessor(self.config)
        self.numeric_processor = NumericPreprocessor(self.config)
        self.categorical_processor = CategoricalPreprocessor(self.config)
        self.interaction_processor = InteractionPreprocessor(self.config)
    
    def preprocess_interactions(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None,
        test_df: Optional[pd.DataFrame] = None
    ) -> Tuple[pd.DataFrame, ...]:
        """Preprocess interaction data."""
        # Fit on training data
        self.interaction_processor.fit(train_df)
        
        # Transform all splits
        train_processed = self.interaction_processor.transform(train_df)
        
        results = [train_processed]
        
        if val_df is not None:
            results.append(self.interaction_processor.transform(val_df))
        
        if test_df is not None:
            results.append(self.interaction_processor.transform(test_df))
        
        return tuple(results)
    
    def preprocess_items(
        self,
        items: List[Dict],
        text_columns: List[str] = ["title", "description"],
        numeric_columns: List[str] = ["duration", "popularity_score"],
        categorical_columns: List[str] = ["genre"]
    ) -> pd.DataFrame:
        """Preprocess item catalog."""
        df = pd.DataFrame(items)
        
        # Text preprocessing
        for col in text_columns:
            if col in df.columns:
                df[f"{col}_clean"] = self.text_processor.batch_clean(
                    df[col].fillna("").tolist()
                )
        
        # Numeric preprocessing
        if numeric_columns:
            df = self.numeric_processor.fit_transform(df, numeric_columns)
        
        # Categorical preprocessing
        if categorical_columns:
            df = self.categorical_processor.fit_transform(df, categorical_columns)
        
        return df
    
    def save(self, path: str):
        """Save preprocessing state."""
        state = {
            "config": self.config.__dict__,
            "numeric_statistics": self.numeric_processor.statistics,
            "category_mappings": self.categorical_processor.category_mappings,
        }
        
        with open(path, "w") as f:
            json.dump(state, f, indent=2, default=str)
    
    def load(self, path: str):
        """Load preprocessing state."""
        with open(path, "r") as f:
            state = json.load(f)
        
        self.config = PreprocessConfig(**state["config"])
        self.numeric_processor.statistics = state["numeric_statistics"]
        self.categorical_processor.category_mappings = state["category_mappings"]


if __name__ == "__main__":
    # Example usage
    config = PreprocessConfig()
    pipeline = DataPreprocessingPipeline(config)
    
    # Sample interaction data
    interactions = pd.DataFrame({
        "user_id": [1, 1, 2, 2, 3],
        "item_id": [10, 20, 10, 30, 20],
        "rating": [5, 4, 3, 5, 4],
        "timestamp": [1000000, 1000100, 1000200, 1000300, 1000400]
    })
    
    processed = pipeline.preprocess_interactions(interactions)
    print(f"Processed shape: {processed[0].shape}")
    print(f"Num users: {pipeline.interaction_processor.num_users}")
    print(f"Num items: {pipeline.interaction_processor.num_items}")
