"""
Evaluation Metrics for Recommendation Systems

Comprehensive metrics including ranking metrics, coverage, diversity,
and A/B testing utilities.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
import math
from dataclasses import dataclass


@dataclass
class MetricsResult:
    """Container for evaluation metrics."""
    precision: Dict[int, float]
    recall: Dict[int, float]
    ndcg: Dict[int, float]
    hit_rate: Dict[int, float]
    mrr: float
    coverage: float
    diversity: float
    novelty: float
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to flat dictionary."""
        result = {"mrr": self.mrr, "coverage": self.coverage, 
                  "diversity": self.diversity, "novelty": self.novelty}
        for k in self.precision:
            result[f"precision@{k}"] = self.precision[k]
            result[f"recall@{k}"] = self.recall[k]
            result[f"ndcg@{k}"] = self.ndcg[k]
            result[f"hr@{k}"] = self.hit_rate[k]
        return result


class RankingMetrics:
    """Ranking-based evaluation metrics."""
    
    @staticmethod
    def precision_at_k(
        predicted: List[int],
        relevant: Set[int],
        k: int
    ) -> float:
        """Compute Precision@K."""
        if k == 0:
            return 0.0
        predicted_k = predicted[:k]
        hits = len(set(predicted_k) & relevant)
        return hits / k
    
    @staticmethod
    def recall_at_k(
        predicted: List[int],
        relevant: Set[int],
        k: int
    ) -> float:
        """Compute Recall@K."""
        if len(relevant) == 0:
            return 0.0
        predicted_k = predicted[:k]
        hits = len(set(predicted_k) & relevant)
        return hits / len(relevant)
    
    @staticmethod
    def ndcg_at_k(
        predicted: List[int],
        relevant: Set[int],
        k: int
    ) -> float:
        """Compute NDCG@K (Normalized Discounted Cumulative Gain)."""
        dcg = 0.0
        for i, item in enumerate(predicted[:k]):
            if item in relevant:
                dcg += 1.0 / math.log2(i + 2)
        
        # Ideal DCG
        idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant), k)))
        
        if idcg == 0:
            return 0.0
        return dcg / idcg
    
    @staticmethod
    def hit_rate_at_k(
        predicted: List[int],
        relevant: Set[int],
        k: int
    ) -> float:
        """Compute Hit Rate@K (1 if any relevant item in top-k)."""
        return 1.0 if len(set(predicted[:k]) & relevant) > 0 else 0.0
    
    @staticmethod
    def mrr(
        predicted: List[int],
        relevant: Set[int]
    ) -> float:
        """Compute Mean Reciprocal Rank."""
        for i, item in enumerate(predicted):
            if item in relevant:
                return 1.0 / (i + 1)
        return 0.0
    
    @staticmethod
    def map_at_k(
        predicted: List[int],
        relevant: Set[int],
        k: int
    ) -> float:
        """Compute Mean Average Precision@K."""
        if len(relevant) == 0:
            return 0.0
        
        score = 0.0
        hits = 0
        
        for i, item in enumerate(predicted[:k]):
            if item in relevant:
                hits += 1
                score += hits / (i + 1)
        
        return score / min(len(relevant), k)


class CoverageMetrics:
    """Coverage and diversity metrics."""
    
    @staticmethod
    def catalog_coverage(
        all_recommendations: List[List[int]],
        total_items: int
    ) -> float:
        """
        Compute catalog coverage.
        
        Measures what fraction of items are ever recommended.
        """
        recommended_items = set()
        for recs in all_recommendations:
            recommended_items.update(recs)
        return len(recommended_items) / total_items
    
    @staticmethod
    def user_coverage(
        user_recommendations: Dict[int, List[int]],
        total_users: int,
        min_recommendations: int = 1
    ) -> float:
        """
        Compute user coverage.
        
        Measures fraction of users who receive recommendations.
        """
        users_with_recs = sum(
            1 for recs in user_recommendations.values()
            if len(recs) >= min_recommendations
        )
        return users_with_recs / total_users
    
    @staticmethod
    def gini_coefficient(
        item_counts: Dict[int, int]
    ) -> float:
        """
        Compute Gini coefficient for recommendation distribution.
        
        Lower is better (more equal distribution).
        """
        counts = sorted(item_counts.values())
        n = len(counts)
        
        if n == 0 or sum(counts) == 0:
            return 0.0
        
        cumulative = np.cumsum(counts)
        return (n + 1 - 2 * sum(cumulative) / cumulative[-1]) / n


class DiversityMetrics:
    """Diversity metrics for recommendation lists."""
    
    @staticmethod
    def intra_list_diversity(
        recommendations: List[int],
        item_embeddings: Dict[int, np.ndarray]
    ) -> float:
        """
        Compute intra-list diversity.
        
        Average pairwise distance between recommended items.
        """
        if len(recommendations) < 2:
            return 0.0
        
        embeddings = [
            item_embeddings.get(item)
            for item in recommendations
            if item in item_embeddings
        ]
        
        if len(embeddings) < 2:
            return 0.0
        
        embeddings = np.array(embeddings)
        n = len(embeddings)
        
        # Compute pairwise cosine distances
        total_distance = 0.0
        count = 0
        
        for i in range(n):
            for j in range(i + 1, n):
                sim = np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]) + 1e-8
                )
                total_distance += 1 - sim
                count += 1
        
        return total_distance / count if count > 0 else 0.0
    
    @staticmethod
    def category_diversity(
        recommendations: List[int],
        item_categories: Dict[int, str]
    ) -> float:
        """
        Compute category diversity.
        
        Number of unique categories in recommendations.
        """
        categories = set(
            item_categories.get(item)
            for item in recommendations
            if item in item_categories
        )
        return len(categories) / len(recommendations) if recommendations else 0.0


class NoveltyMetrics:
    """Novelty and serendipity metrics."""
    
    @staticmethod
    def popularity_based_novelty(
        recommendations: List[int],
        item_popularity: Dict[int, float]
    ) -> float:
        """
        Compute novelty based on item popularity.
        
        Higher score means recommending less popular (more novel) items.
        """
        if not recommendations:
            return 0.0
        
        novelty_scores = []
        for item in recommendations:
            popularity = item_popularity.get(item, 0)
            if popularity > 0:
                novelty_scores.append(-math.log2(popularity))
            else:
                novelty_scores.append(0.0)
        
        return np.mean(novelty_scores)
    
    @staticmethod
    def serendipity(
        recommendations: List[int],
        relevant: Set[int],
        expected: Set[int]
    ) -> float:
        """
        Compute serendipity.
        
        Relevant items that were not expected (surprising but useful).
        """
        if not recommendations:
            return 0.0
        
        serendipitous = set(recommendations) & relevant - expected
        return len(serendipitous) / len(recommendations)


class RecommenderEvaluator:
    """
    Complete evaluator for recommendation systems.
    
    Computes all metrics and provides summary statistics.
    """
    
    def __init__(
        self,
        k_values: List[int] = [5, 10, 20],
        item_embeddings: Optional[Dict[int, np.ndarray]] = None,
        item_categories: Optional[Dict[int, str]] = None,
        item_popularity: Optional[Dict[int, float]] = None
    ):
        self.k_values = k_values
        self.item_embeddings = item_embeddings or {}
        self.item_categories = item_categories or {}
        self.item_popularity = item_popularity or {}
    
    def evaluate_user(
        self,
        predicted: List[int],
        relevant: Set[int]
    ) -> Dict[str, float]:
        """Evaluate recommendations for a single user."""
        metrics = {}
        
        for k in self.k_values:
            metrics[f"precision@{k}"] = RankingMetrics.precision_at_k(predicted, relevant, k)
            metrics[f"recall@{k}"] = RankingMetrics.recall_at_k(predicted, relevant, k)
            metrics[f"ndcg@{k}"] = RankingMetrics.ndcg_at_k(predicted, relevant, k)
            metrics[f"hr@{k}"] = RankingMetrics.hit_rate_at_k(predicted, relevant, k)
        
        metrics["mrr"] = RankingMetrics.mrr(predicted, relevant)
        
        if self.item_embeddings:
            metrics["diversity"] = DiversityMetrics.intra_list_diversity(
                predicted, self.item_embeddings
            )
        
        if self.item_categories:
            metrics["category_diversity"] = DiversityMetrics.category_diversity(
                predicted, self.item_categories
            )
        
        if self.item_popularity:
            metrics["novelty"] = NoveltyMetrics.popularity_based_novelty(
                predicted, self.item_popularity
            )
        
        return metrics
    
    def evaluate_all(
        self,
        predictions: Dict[int, List[int]],
        ground_truth: Dict[int, Set[int]],
        total_items: int
    ) -> MetricsResult:
        """
        Evaluate all users and compute aggregate metrics.
        
        Args:
            predictions: Dict mapping user_id to predicted item list
            ground_truth: Dict mapping user_id to relevant item set
            total_items: Total number of items in catalog
            
        Returns:
            MetricsResult with all computed metrics
        """
        all_metrics = defaultdict(list)
        all_recommendations = []
        
        for user_id, predicted in predictions.items():
            relevant = ground_truth.get(user_id, set())
            user_metrics = self.evaluate_user(predicted, relevant)
            
            for key, value in user_metrics.items():
                all_metrics[key].append(value)
            
            all_recommendations.append(predicted)
        
        # Aggregate metrics
        precision = {k: np.mean(all_metrics[f"precision@{k}"]) for k in self.k_values}
        recall = {k: np.mean(all_metrics[f"recall@{k}"]) for k in self.k_values}
        ndcg = {k: np.mean(all_metrics[f"ndcg@{k}"]) for k in self.k_values}
        hit_rate = {k: np.mean(all_metrics[f"hr@{k}"]) for k in self.k_values}
        mrr = np.mean(all_metrics["mrr"])
        
        # Coverage
        coverage = CoverageMetrics.catalog_coverage(all_recommendations, total_items)
        
        # Diversity
        diversity = np.mean(all_metrics.get("diversity", [0.0]))
        
        # Novelty
        novelty = np.mean(all_metrics.get("novelty", [0.0]))
        
        return MetricsResult(
            precision=precision,
            recall=recall,
            ndcg=ndcg,
            hit_rate=hit_rate,
            mrr=mrr,
            coverage=coverage,
            diversity=diversity,
            novelty=novelty
        )


def compute_ab_test_significance(
    metric_a: List[float],
    metric_b: List[float],
    alpha: float = 0.05
) -> Tuple[float, float, bool]:
    """
    Compute statistical significance for A/B test.
    
    Returns:
        Tuple of (mean_diff, p_value, is_significant)
    """
    from scipy import stats
    
    mean_diff = np.mean(metric_b) - np.mean(metric_a)
    t_stat, p_value = stats.ttest_ind(metric_a, metric_b)
    is_significant = p_value < alpha
    
    return mean_diff, p_value, is_significant


if __name__ == "__main__":
    # Example usage
    evaluator = RecommenderEvaluator(k_values=[5, 10, 20])
    
    # Sample data
    predictions = {
        1: [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        2: [15, 25, 35, 45, 55, 65, 75, 85, 95, 105],
    }
    
    ground_truth = {
        1: {20, 40, 60, 80, 100},
        2: {15, 35, 55, 75, 95},
    }
    
    results = evaluator.evaluate_all(predictions, ground_truth, total_items=1000)
    
    print("Evaluation Results:")
    for key, value in results.to_dict().items():
        print(f"  {key}: {value:.4f}")
