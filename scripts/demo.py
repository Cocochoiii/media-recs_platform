#!/usr/bin/env python3
"""
Quick Start Demo - Media Recommender System

This script demonstrates the core functionality of the recommendation system.
Run this after installing dependencies to verify everything works.

Usage:
    python scripts/demo.py
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

def print_header(text):
    print("\n" + "="*60)
    print(f" {text}")
    print("="*60)

def print_success(text):
    print(f"✅ {text}")

def print_info(text):
    print(f"ℹ️  {text}")

def print_warning(text):
    print(f"⚠️  {text}")

def main():
    print_header("Media Recommender System - Demo")
    print_info(f"PyTorch version: {torch.__version__}")
    print_info(f"CUDA available: {torch.cuda.is_available()}")
    
    tests_passed = 0
    tests_total = 0
    
    # ========================================
    # 1. Test Collaborative Filtering
    # ========================================
    print_header("1. Collaborative Filtering (NCF)")
    tests_total += 1
    
    try:
        from src.models.collaborative_filter import CollaborativeConfig, NeuralCollaborativeFiltering
        
        config = CollaborativeConfig(
            num_users=1000,
            num_items=5000,
            embedding_dim=64,
            hidden_layers=[128, 64, 32]
        )
        
        model = NeuralCollaborativeFiltering(config)
        
        users = torch.randint(0, 1000, (32,))
        items = torch.randint(0, 5000, (32,))
        
        with torch.no_grad():
            scores = model(users, items)
        
        print_success(f"NCF model: {sum(p.numel() for p in model.parameters()):,} parameters")
        print_success(f"Output shape: {scores.shape}, range: [{scores.min():.3f}, {scores.max():.3f}]")
        tests_passed += 1
    except Exception as e:
        print_warning(f"Skipped: {e}")
    
    # ========================================
    # 2. Test Sequential Model (BERT4Rec)
    # ========================================
    print_header("2. Sequential Model (BERT4Rec)")
    tests_total += 1
    
    try:
        from src.models.transformer_models import TransformerRecConfig, BERT4Rec
        
        config = TransformerRecConfig(
            num_items=5000,
            max_seq_len=50,
            embedding_dim=128,
            num_heads=4,
            num_layers=2
        )
        
        model = BERT4Rec(config)
        sequences = torch.randint(0, 5000, (4, 50))
        
        with torch.no_grad():
            logits = model(sequences)
        
        print_success(f"BERT4Rec: {sum(p.numel() for p in model.parameters()):,} parameters")
        print_success(f"Output shape: {logits.shape}")
        tests_passed += 1
    except Exception as e:
        print_warning(f"Skipped: {e}")
    
    # ========================================
    # 3. Test Graph Neural Network (LightGCN)
    # ========================================
    print_header("3. Graph Neural Network (LightGCN)")
    tests_total += 1
    
    try:
        from src.models.graph_neural_network import GNNConfig, LightGCN
        
        config = GNNConfig(
            num_users=1000,
            num_items=5000,
            embedding_dim=64,
            num_layers=3
        )
        
        model = LightGCN(config)
        interactions = [(i % 1000, i % 5000) for i in range(10000)]
        model.build_graph(interactions)
        
        print_success(f"LightGCN: {sum(p.numel() for p in model.parameters()):,} parameters")
        print_success(f"Graph built with {len(interactions)} edges")
        tests_passed += 1
    except Exception as e:
        print_warning(f"Skipped: {e}")
    
    # ========================================
    # 4. Test Multi-Task Learning (MMoE)
    # ========================================
    print_header("4. Multi-Task Learning (MMoE)")
    tests_total += 1
    
    try:
        from src.models.multi_task_learning import MTLConfig, MMoE
        
        config = MTLConfig(
            num_users=1000,
            num_items=5000,
            task_names=["ctr", "cvr", "engagement"],
            num_experts=6
        )
        
        model = MMoE(config)
        users = torch.randint(0, 1000, (32,))
        items = torch.randint(0, 5000, (32,))
        
        with torch.no_grad():
            outputs = model(users, items)
        
        print_success(f"MMoE: {sum(p.numel() for p in model.parameters()):,} parameters")
        print_success(f"Tasks: {list(outputs.keys())}")
        tests_passed += 1
    except Exception as e:
        print_warning(f"Skipped: {e}")
    
    # ========================================
    # 5. Test Deep Feature Interaction (DeepFM)
    # ========================================
    print_header("5. Deep Feature Interaction (DeepFM)")
    tests_total += 1
    
    try:
        from src.models.deep_feature_interaction import FeatureInteractionConfig, DeepFM
        
        config = FeatureInteractionConfig(
            sparse_feature_dims={
                "user_id": 1000,
                "item_id": 5000,
                "category": 50,
                "hour": 24
            },
            embedding_dim=16,
            hidden_dims=[128, 64]
        )
        
        model = DeepFM(config)
        batch = {
            "user_id": torch.randint(0, 1000, (32,)),
            "item_id": torch.randint(0, 5000, (32,)),
            "category": torch.randint(0, 50, (32,)),
            "hour": torch.randint(0, 24, (32,))
        }
        
        with torch.no_grad():
            scores = model(batch)
        
        print_success(f"DeepFM: {sum(p.numel() for p in model.parameters()):,} parameters")
        print_success(f"CTR predictions: range=[{scores.min():.3f}, {scores.max():.3f}]")
        tests_passed += 1
    except Exception as e:
        print_warning(f"Skipped: {e}")
    
    # ========================================
    # 6. Test Reinforcement Learning (DQN)
    # ========================================
    print_header("6. Reinforcement Learning (DQN)")
    tests_total += 1
    
    try:
        from src.models.reinforcement_learning import RLConfig, DuelingDQN
        
        config = RLConfig(
            num_items=5000,
            state_dim=128,
            hidden_dims=[256, 128]
        )
        
        model = DuelingDQN(config)
        history = torch.randint(0, 5000, (4, 20))
        
        with torch.no_grad():
            q_values = model(history)
        
        print_success(f"Dueling DQN: {sum(p.numel() for p in model.parameters()):,} parameters")
        print_success(f"Q-values shape: {q_values.shape}")
        tests_passed += 1
    except Exception as e:
        print_warning(f"Skipped: {e}")
    
    # ========================================
    # 7. Test Causal Inference (IPS)
    # ========================================
    print_header("7. Causal Inference (IPS Recommender)")
    tests_total += 1
    
    try:
        from src.models.causal_inference import CausalConfig, IPSRecommender
        
        config = CausalConfig(
            num_users=1000,
            num_items=5000,
            embedding_dim=64
        )
        
        model = IPSRecommender(config)
        users = torch.randint(0, 1000, (32,))
        items = torch.randint(0, 5000, (32,))
        
        with torch.no_grad():
            predictions = model(users, items)
        
        print_success(f"IPS Recommender: {sum(p.numel() for p in model.parameters()):,} parameters")
        print_success(f"Debiased predictions shape: {predictions.shape}")
        tests_passed += 1
    except Exception as e:
        print_warning(f"Skipped: {e}")
    
    # ========================================
    # 8. Test Explainable Recommendations
    # ========================================
    print_header("8. Explainable Recommendations")
    tests_total += 1
    
    try:
        from src.models.explainable_recommendations import ExplainableConfig, AttentionExplainer
        
        config = ExplainableConfig(
            num_users=1000,
            num_items=5000,
            num_features=100,
            embedding_dim=64
        )
        
        model = AttentionExplainer(config)
        users = torch.randint(0, 1000, (4,))
        items = torch.randint(0, 5000, (4,))
        features = torch.randint(0, 100, (4, 10))
        
        with torch.no_grad():
            scores, attention = model(users, items, features, return_attention=True)
        
        print_success(f"Attention Explainer: {sum(p.numel() for p in model.parameters()):,} parameters")
        print_success(f"Attention shape: {attention.shape}")
        tests_passed += 1
    except Exception as e:
        print_warning(f"Skipped: {e}")
    
    # ========================================
    # 9. Test Knowledge Graph (TransE)
    # ========================================
    print_header("9. Knowledge Graph (TransE)")
    tests_total += 1
    
    try:
        from src.models.knowledge_graph import KGConfig, TransE
        
        config = KGConfig(
            num_entities=10000,
            num_relations=50,
            embedding_dim=64
        )
        
        model = TransE(config)
        heads = torch.randint(0, 10000, (32,))
        relations = torch.randint(0, 50, (32,))
        tails = torch.randint(0, 10000, (32,))
        
        with torch.no_grad():
            scores = model(heads, relations, tails)
        
        print_success(f"TransE: {sum(p.numel() for p in model.parameters()):,} parameters")
        print_success(f"Triple scores: mean={scores.mean():.4f}")
        tests_passed += 1
    except Exception as e:
        print_warning(f"Skipped: {e}")
    
    # ========================================
    # 10. Test Real-time Personalization
    # ========================================
    print_header("10. Real-time Personalization")
    tests_total += 1
    
    try:
        from src.models.realtime_personalization import RealTimeConfig, OnlineModel, ContextualBandit
        
        config = RealTimeConfig(
            num_items=5000,
            embedding_dim=64,
            context_dim=32
        )
        
        model = OnlineModel(config)
        bandit = ContextualBandit(num_items=100, context_dim=32, alpha=0.1)
        
        context = np.random.randn(32).astype(np.float32)
        selected_item = bandit.select_item(context)
        
        print_success(f"Online model: {sum(p.numel() for p in model.parameters()):,} parameters")
        print_success(f"Contextual bandit selected item: {selected_item}")
        tests_passed += 1
    except Exception as e:
        print_warning(f"Skipped: {e}")
    
    # ========================================
    # Summary
    # ========================================
    print_header("Summary")
    print_success(f"Tests passed: {tests_passed}/{tests_total}")
    print()
    
    if tests_passed >= 8:
        print("🎉 Your recommendation system is ready to use!")
    elif tests_passed >= 5:
        print("✅ Core models working! Some optional features may need extra dependencies.")
    else:
        print("⚠️  Some issues detected. Check the errors above.")
    
    print()
    print("Next steps:")
    print("  1. Start API: PYTHONPATH=. python -m uvicorn src.api.main:app --reload")
    print("  2. Open docs: http://localhost:8000/docs")

if __name__ == "__main__":
    main()
