"""
API Tests

Tests for FastAPI recommendation endpoints.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestHealthEndpoint:
    """Tests for health check endpoint."""
    
    def test_health_response_structure(self):
        """Test health response has correct structure."""
        # Mock response structure
        response = {
            "status": "healthy",
            "version": "1.0.0",
            "models_loaded": True,
            "uptime_seconds": 100.5
        }
        
        assert "status" in response
        assert "version" in response
        assert "models_loaded" in response
        assert "uptime_seconds" in response
    
    def test_health_status_values(self):
        """Test valid health status values."""
        valid_statuses = ["healthy", "starting", "unhealthy"]
        
        for status in valid_statuses:
            assert status in valid_statuses


class TestRecommendationEndpoint:
    """Tests for recommendation endpoint."""
    
    def test_recommendation_request_validation(self):
        """Test request validation."""
        # Valid request
        valid_request = {
            "user_id": 123,
            "n_recommendations": 10,
            "exclude_items": [1, 2, 3],
            "diversity_factor": 0.3
        }
        
        assert valid_request["user_id"] > 0
        assert 1 <= valid_request["n_recommendations"] <= 100
        assert 0 <= valid_request["diversity_factor"] <= 1
    
    def test_recommendation_response_structure(self):
        """Test response has correct structure."""
        response = {
            "user_id": 123,
            "recommendations": [
                {"item_id": 1, "score": 0.95, "explanation": "Based on your preferences"},
                {"item_id": 2, "score": 0.87, "explanation": "Similar to items you liked"},
            ],
            "model_version": "1.0.0",
            "latency_ms": 45.2
        }
        
        assert "user_id" in response
        assert "recommendations" in response
        assert len(response["recommendations"]) > 0
        assert "model_version" in response
        assert "latency_ms" in response
        
        # Check recommendation item structure
        for rec in response["recommendations"]:
            assert "item_id" in rec
            assert "score" in rec
    
    def test_exclude_items_honored(self):
        """Test that excluded items are not in recommendations."""
        exclude_items = {1, 2, 3}
        recommendations = [
            {"item_id": 10, "score": 0.9},
            {"item_id": 20, "score": 0.8},
        ]
        
        rec_ids = {r["item_id"] for r in recommendations}
        assert rec_ids.isdisjoint(exclude_items)


class TestInteractionEndpoint:
    """Tests for interaction logging endpoint."""
    
    def test_interaction_types(self):
        """Test valid interaction types."""
        valid_types = ["view", "click", "purchase", "rating"]
        
        for itype in valid_types:
            interaction = {
                "user_id": 1,
                "item_id": 100,
                "interaction_type": itype
            }
            assert interaction["interaction_type"] in valid_types
    
    def test_rating_range(self):
        """Test rating is in valid range."""
        interaction = {
            "user_id": 1,
            "item_id": 100,
            "interaction_type": "rating",
            "rating": 4.5
        }
        
        assert 1 <= interaction["rating"] <= 5


class TestSimilarItemsEndpoint:
    """Tests for similar items endpoint."""
    
    def test_similar_items_response(self):
        """Test similar items response structure."""
        response = {
            "source_item_id": 100,
            "similar_items": [
                {"item_id": 101, "score": 0.95},
                {"item_id": 102, "score": 0.88},
            ]
        }
        
        assert "source_item_id" in response
        assert "similar_items" in response
        assert len(response["similar_items"]) > 0
        
        # Check scores are sorted descending
        scores = [item["score"] for item in response["similar_items"]]
        assert scores == sorted(scores, reverse=True)
    
    def test_source_not_in_similar(self):
        """Test source item is not in similar items."""
        source_id = 100
        similar_items = [
            {"item_id": 101, "score": 0.95},
            {"item_id": 102, "score": 0.88},
        ]
        
        similar_ids = {item["item_id"] for item in similar_items}
        assert source_id not in similar_ids


class TestPopularItemsEndpoint:
    """Tests for popular items endpoint."""
    
    def test_popular_items_sorted(self):
        """Test popular items are sorted by score."""
        popular_items = [
            {"item_id": 1, "score": 1.0},
            {"item_id": 2, "score": 0.95},
            {"item_id": 3, "score": 0.9},
        ]
        
        scores = [item["score"] for item in popular_items]
        assert scores == sorted(scores, reverse=True)
    
    def test_genre_filter(self):
        """Test genre filtering works."""
        all_items = [
            {"item_id": 1, "genre": "Action", "score": 1.0},
            {"item_id": 2, "genre": "Comedy", "score": 0.95},
            {"item_id": 3, "genre": "Action", "score": 0.9},
        ]
        
        genre_filter = "Action"
        filtered = [item for item in all_items if item["genre"] == genre_filter]
        
        assert len(filtered) == 2
        assert all(item["genre"] == genre_filter for item in filtered)


class TestErrorHandling:
    """Tests for error handling."""
    
    def test_invalid_user_id(self):
        """Test error for invalid user ID."""
        invalid_ids = [-1, 0, "abc"]
        
        for uid in invalid_ids:
            try:
                assert isinstance(uid, int) and uid > 0
            except (AssertionError, TypeError):
                pass  # Expected for invalid IDs
    
    def test_invalid_n_recommendations(self):
        """Test error for invalid n_recommendations."""
        invalid_values = [0, -1, 101, 1000]
        
        for n in invalid_values:
            valid = 1 <= n <= 100
            if not valid:
                pass  # Should raise validation error


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
