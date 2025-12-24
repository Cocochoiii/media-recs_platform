"""Utilities module for Media Recommender."""

from .metrics import (
    RankingMetrics,
    CoverageMetrics,
    DiversityMetrics,
    NoveltyMetrics,
    RecommenderEvaluator,
    MetricsResult,
    compute_ab_test_significance
)

from .cache import (
    CacheConfig,
    RedisCache,
    RecommendationCache,
    InMemoryCache,
    cached
)

from .logging_config import (
    setup_logging,
    get_logger,
    log_execution_time,
    RequestLogger,
    MetricLogger,
    JSONFormatter,
    PrettyFormatter
)

__all__ = [
    # Metrics
    "RankingMetrics",
    "CoverageMetrics", 
    "DiversityMetrics",
    "NoveltyMetrics",
    "RecommenderEvaluator",
    "MetricsResult",
    "compute_ab_test_significance",
    # Cache
    "CacheConfig",
    "RedisCache",
    "RecommendationCache",
    "InMemoryCache",
    "cached",
    # Logging
    "setup_logging",
    "get_logger",
    "log_execution_time",
    "RequestLogger",
    "MetricLogger",
    "JSONFormatter",
    "PrettyFormatter"
]
