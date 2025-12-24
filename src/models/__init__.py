"""
Media Recommender Models

This package contains all machine learning models for the recommendation system.
"""

import logging
logger = logging.getLogger(__name__)

# Track what's available
_available_modules = []

# ============================================
# Core Models (should always work)
# ============================================

try:
    from .collaborative_filter import (
        CollaborativeConfig,
        MatrixFactorization,
        NeuralCollaborativeFiltering,
        ImplicitFeedbackNCF,
        CollaborativeFilteringRecommender
    )
    _available_modules.append("collaborative_filter")
except ImportError as e:
    logger.warning(f"Could not import collaborative_filter: {e}")

try:
    from .lstm_sequential import (
        LSTMConfig,
        LSTMSequentialModel,
        GRU4Rec,
        SASRec,
        SequentialRecommender
    )
    _available_modules.append("lstm_sequential")
except ImportError as e:
    logger.warning(f"Could not import lstm_sequential: {e}")

try:
    from .graph_neural_network import (
        GNNConfig,
        LightGCN,
        NGCF,
        GATRecommender,
        GNNRecommender
    )
    _available_modules.append("graph_neural_network")
except ImportError as e:
    logger.warning(f"Could not import graph_neural_network: {e}")

try:
    from .multi_task_learning import (
        MTLConfig,
        SharedBottomMTL,
        MMoE,
        PLE,
        ESMM,
        MultiTaskLoss,
        MultiTaskRecommender
    )
    _available_modules.append("multi_task_learning")
except ImportError as e:
    logger.warning(f"Could not import multi_task_learning: {e}")

try:
    from .deep_feature_interaction import (
        FeatureInteractionConfig,
        DeepFM,
        DCNv2,
        AutoInt,
        FiBiNET,
        FeatureInteractionRecommender
    )
    _available_modules.append("deep_feature_interaction")
except ImportError as e:
    logger.warning(f"Could not import deep_feature_interaction: {e}")

try:
    from .reinforcement_learning import (
        RLConfig,
        DQN,
        DuelingDQN,
        A2C,
        SlateRecommender,
        DQNAgent,
        A2CAgent,
        ReplayBuffer,
        PrioritizedReplayBuffer
    )
    _available_modules.append("reinforcement_learning")
except ImportError as e:
    logger.warning(f"Could not import reinforcement_learning: {e}")

try:
    from .transformer_models import (
        TransformerRecConfig,
        BERT4Rec,
        SASRecPlusPlus,
        BST,
        TransformerRecommender
    )
    _available_modules.append("transformer_models")
except ImportError as e:
    logger.warning(f"Could not import transformer_models: {e}")

try:
    from .ensemble_models import (
        EnsembleConfig,
        WeightedEnsemble,
        StackingMetaLearner,
        LearningToRankStacker,
        CascadeEnsemble,
        MultiArmedBanditSelector,
        CrossStitchEnsemble,
        DiversityAwareEnsemble
    )
    _available_modules.append("ensemble_models")
except ImportError as e:
    logger.warning(f"Could not import ensemble_models: {e}")

try:
    from .realtime_personalization import (
        RealTimeConfig,
        OnlineModel,
        OnlineLearner,
        ContextualBandit,
        SessionManager,
        ABTestFramework,
        RealTimeRecommender
    )
    _available_modules.append("realtime_personalization")
except ImportError as e:
    logger.warning(f"Could not import realtime_personalization: {e}")

try:
    from .knowledge_graph import (
        KGConfig,
        TransE,
        RotatE,
        KGAT,
        RippleNet,
        KGEnhancedRecommender
    )
    _available_modules.append("knowledge_graph")
except ImportError as e:
    logger.warning(f"Could not import knowledge_graph: {e}")

try:
    from .causal_inference import (
        CausalConfig,
        PropensityNet,
        IPSRecommender,
        DoublyRobustEstimator,
        CausalEmbedding,
        CounterfactualRanking,
        TreatmentEffectEstimator,
        CausalRecommender
    )
    _available_modules.append("causal_inference")
except ImportError as e:
    logger.warning(f"Could not import causal_inference: {e}")

try:
    from .explainable_recommendations import (
        ExplainableConfig,
        ExplanationType,
        Explanation,
        AttentionExplainer,
        PathExplainer,
        CounterfactualExplainer,
        TemplateExplainer,
        ExplainableRecommender
    )
    _available_modules.append("explainable_recommendations")
except ImportError as e:
    logger.warning(f"Could not import explainable_recommendations: {e}")

try:
    from .advanced_techniques import (
        AdvancedRecConfig,
        InterestEvolutionModel,
        FairnessConstraint,
        FairnessAwareRecommender,
        DiversityOptimizer,
        NoveltyEnhancer,
        AdvancedRecommenderSystem
    )
    _available_modules.append("advanced_techniques")
except ImportError as e:
    logger.warning(f"Could not import advanced_techniques: {e}")

# ============================================
# Optional Models (may require extra dependencies)
# ============================================

try:
    from .bert_embedder import (
        BertEmbedderConfig,
        BertContentEmbedder,
        ContentBasedRecommender,
        DualEncoderModel
    )
    _available_modules.append("bert_embedder")
except ImportError as e:
    logger.warning(f"Could not import bert_embedder: {e}")

try:
    from .contrastive_learner import (
        ContrastiveConfig,
        ContrastiveEncoder,
        InfoNCELoss,
        HardNegativeMiner,
        ContrastiveLearningRecommender,
        SimCLRLoss
    )
    _available_modules.append("contrastive_learner")
except ImportError as e:
    logger.warning(f"Could not import contrastive_learner: {e}")

try:
    from .hybrid_recommender import (
        HybridConfig,
        HybridRecommender,
        UserProfile,
        ColdStartHandler,
        DiversityReranker,
        RecommenderType
    )
    _available_modules.append("hybrid_recommender")
except ImportError as e:
    logger.warning(f"Could not import hybrid_recommender: {e}")


def list_available_modules():
    """List all successfully imported modules."""
    return _available_modules.copy()
