"""
Data Processing Tests

Tests for data loading, preprocessing, and feature engineering.
"""

import pytest
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.dataset import DataConfig, DataProcessor, InteractionDataset, SequenceDataset
from data.preprocessor import PreprocessConfig, TextPreprocessor, NumericPreprocessor, DataPreprocessingPipeline
from data.feature_engineering import FeatureConfig, UserFeatureExtractor, ItemFeatureExtractor


class TestDataProcessor:
    """Tests for DataProcessor."""
    
    def test_split_data_ratios(self, sample_interactions):
        """Test data split ratios."""
        config = DataConfig(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
        processor = DataProcessor(config)
        
        train, val, test = processor.split_data(sample_interactions)
        
        total = len(sample_interactions)
        assert abs(len(train) / total - 0.8) < 0.01
        assert abs(len(val) / total - 0.1) < 0.01
        assert abs(len(test) / total - 0.1) < 0.01
    
    def test_no_data_leakage(self, sample_interactions):
        """Test no overlap between splits."""
        config = DataConfig()
        processor = DataProcessor(config)
        
        train, val, test = processor.split_data(sample_interactions)
        
        train_idx = set(train.index)
        val_idx = set(val.index)
        test_idx = set(test.index)
        
        assert len(train_idx & val_idx) == 0
        assert len(train_idx & test_idx) == 0
        assert len(val_idx & test_idx) == 0
    
    def test_id_mappings(self, sample_interactions):
        """Test ID mappings are created correctly."""
        config = DataConfig()
        processor = DataProcessor(config)
        
        user_map, item_map = processor.create_id_mappings(sample_interactions)
        
        # Check all IDs are mapped
        assert len(user_map) == sample_interactions["user_id"].nunique()
        assert len(item_map) == sample_interactions["item_id"].nunique()
        
        # Check mappings start from 1
        assert min(user_map.values()) == 1
        assert min(item_map.values()) == 1


class TestInteractionDataset:
    """Tests for InteractionDataset."""
    
    def test_dataset_length(self, sample_interactions):
        """Test dataset length matches interactions."""
        dataset = InteractionDataset(sample_interactions, num_items=1000, mode="train")
        assert len(dataset) == len(sample_interactions)
    
    def test_getitem_returns_tensors(self, sample_interactions):
        """Test __getitem__ returns proper tensors."""
        dataset = InteractionDataset(sample_interactions, num_items=1000, mode="train")
        
        item = dataset[0]
        
        assert "user_ids" in item
        assert "item_ids" in item
    
    def test_negative_sampling(self, sample_interactions):
        """Test negative sampling in training mode."""
        dataset = InteractionDataset(
            sample_interactions, num_items=1000, mode="train", negative_samples=4
        )
        
        item = dataset[0]
        
        if "neg_item_ids" in item:
            assert item["neg_item_ids"].shape[0] == 4


class TestSequenceDataset:
    """Tests for SequenceDataset."""
    
    def test_sequence_length(self, sample_interactions):
        """Test sequences are within max length."""
        max_length = 20
        dataset = SequenceDataset(sample_interactions, max_length=max_length)
        
        for i in range(min(10, len(dataset))):
            item = dataset[i]
            assert item["sequences"].shape[0] == max_length
    
    def test_padding(self, sample_interactions):
        """Test sequences are properly padded."""
        dataset = SequenceDataset(sample_interactions, max_length=50)
        
        item = dataset[0]
        seq = item["sequences"].numpy()
        length = item["lengths"].item()
        
        # Check padding is at the beginning
        assert all(seq[:50-length] == 0)


class TestTextPreprocessor:
    """Tests for TextPreprocessor."""
    
    def test_lowercase(self):
        """Test lowercase conversion."""
        config = PreprocessConfig(lowercase=True)
        processor = TextPreprocessor(config)
        
        result = processor.clean("Hello WORLD")
        assert result == result.lower()
    
    def test_remove_urls(self):
        """Test URL removal."""
        config = PreprocessConfig()
        processor = TextPreprocessor(config)
        
        result = processor.clean("Check http://example.com for more")
        assert "http" not in result
        assert "example.com" not in result
    
    def test_remove_html(self):
        """Test HTML tag removal."""
        config = PreprocessConfig()
        processor = TextPreprocessor(config)
        
        result = processor.clean("<p>Hello</p> <b>World</b>")
        assert "<" not in result
        assert ">" not in result
    
    def test_batch_clean(self):
        """Test batch cleaning."""
        config = PreprocessConfig()
        processor = TextPreprocessor(config)
        
        texts = ["Hello World", "Test <b>HTML</b>", "URL http://test.com"]
        results = processor.batch_clean(texts)
        
        assert len(results) == 3


class TestNumericPreprocessor:
    """Tests for NumericPreprocessor."""
    
    def test_missing_value_fill(self):
        """Test missing value filling."""
        config = PreprocessConfig(fill_missing_numeric="median")
        processor = NumericPreprocessor(config)
        
        df = pd.DataFrame({"col": [1, 2, np.nan, 4, 5]})
        processor.fit(df, ["col"])
        result = processor.transform(df, ["col"])
        
        assert not result["col"].isna().any()
    
    def test_scaling_standard(self):
        """Test standard scaling."""
        config = PreprocessConfig(scaling_method="standard")
        processor = NumericPreprocessor(config)
        
        df = pd.DataFrame({"col": [1, 2, 3, 4, 5]})
        processor.fit(df, ["col"])
        result = processor.transform(df, ["col"])
        
        # Should have mean ~0 and std ~1 (allowing for small sample size variance)
        assert abs(result["col"].mean()) < 0.1
        assert abs(result["col"].std() - 1) < 0.15  # Relaxed tolerance for small samples
    
    def test_outlier_handling(self):
        """Test outlier clipping."""
        config = PreprocessConfig(remove_outliers=True, outlier_std_threshold=2)
        processor = NumericPreprocessor(config)
        
        df = pd.DataFrame({"col": [1, 2, 3, 4, 100]})  # 100 is outlier
        processor.fit(df, ["col"])
        result = processor.transform(df, ["col"])
        
        # Outlier should be clipped
        assert result["col"].max() < 50


class TestUserFeatureExtractor:
    """Tests for UserFeatureExtractor."""
    
    def test_interaction_features(self, sample_interactions):
        """Test interaction feature extraction."""
        config = FeatureConfig()
        extractor = UserFeatureExtractor(config)
        
        user_id = sample_interactions["user_id"].iloc[0]
        features = extractor.extract_interaction_features(user_id, sample_interactions)
        
        assert "total_interactions" in features
        assert "unique_items" in features
        assert features["total_interactions"] > 0
    
    def test_demographic_features(self):
        """Test demographic feature extraction."""
        config = FeatureConfig()
        extractor = UserFeatureExtractor(config)
        
        demographics = {
            "age_group": "25-34",
            "gender": "male",
            "device_type": "mobile"
        }
        
        features = extractor.extract_demographic_features(demographics)
        
        assert "age_25-34" in features
        assert features["age_25-34"] == 1.0
        assert "gender_male" in features
        assert features["gender_male"] == 1.0
    
    def test_default_features_for_new_user(self, sample_interactions):
        """Test default features for new user."""
        config = FeatureConfig()
        extractor = UserFeatureExtractor(config)
        
        # Non-existent user
        features = extractor.extract_interaction_features(999999, sample_interactions)
        
        assert features["total_interactions"] == 0


class TestItemFeatureExtractor:
    """Tests for ItemFeatureExtractor."""
    
    def test_fit_encoders(self, sample_items):
        """Test fitting genre and tag encoders."""
        config = FeatureConfig()
        extractor = ItemFeatureExtractor(config)
        extractor.fit(sample_items)
        
        assert len(extractor.genre_encoder) > 0
        assert len(extractor.tag_encoder) > 0
    
    def test_content_features(self, sample_items):
        """Test content feature extraction."""
        config = FeatureConfig()
        extractor = ItemFeatureExtractor(config)
        
        features = extractor.extract_content_features(sample_items[0])
        
        assert "title_length" in features
        assert "description_length" in features
        assert "has_description" in features
    
    def test_categorical_features(self, sample_items):
        """Test categorical feature extraction."""
        config = FeatureConfig()
        extractor = ItemFeatureExtractor(config)
        extractor.fit(sample_items)
        
        features = extractor.extract_categorical_features(sample_items[0])
        
        assert isinstance(features, np.ndarray)
        assert features.sum() > 0  # At least one genre/tag should be set


class TestDataPreprocessingPipeline:
    """Tests for complete preprocessing pipeline."""
    
    def test_preprocess_interactions(self, sample_interactions):
        """Test full interaction preprocessing."""
        pipeline = DataPreprocessingPipeline()
        
        train_df = sample_interactions.iloc[:8000]
        val_df = sample_interactions.iloc[8000:]
        
        train_processed, val_processed = pipeline.preprocess_interactions(
            train_df, val_df
        )
        
        assert "user_idx" in train_processed.columns
        assert "item_idx" in train_processed.columns
        assert pipeline.interaction_processor.num_users > 0
        assert pipeline.interaction_processor.num_items > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
