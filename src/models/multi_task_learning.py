"""
Multi-Task Learning for Recommendations

Implements multi-task models that jointly optimize multiple objectives:
- Click-Through Rate (CTR) prediction
- Conversion Rate (CVR) prediction  
- Watch Time / Engagement prediction
- User Satisfaction prediction

Models:
- Shared-Bottom MTL
- MMoE (Multi-gate Mixture-of-Experts)
- PLE (Progressive Layered Extraction)
- ESMM (Entire Space Multi-Task Model)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np


@dataclass
class MTLConfig:
    """Configuration for Multi-Task Learning."""
    # Architecture
    input_dim: int = 256
    hidden_dims: List[int] = field(default_factory=lambda: [512, 256, 128])
    num_experts: int = 8
    num_tasks: int = 3
    task_names: List[str] = field(default_factory=lambda: ["ctr", "cvr", "engagement"])
    
    # Embeddings
    num_users: int = 50000
    num_items: int = 10000
    user_embedding_dim: int = 64
    item_embedding_dim: int = 64
    
    # Training
    dropout: float = 0.2
    expert_dropout: float = 0.1


class ExpertNetwork(nn.Module):
    """Single expert network."""
    
    def __init__(self, input_dim: int, hidden_dims: List[int], dropout: float = 0.1):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class GatingNetwork(nn.Module):
    """Gating network for expert selection."""
    
    def __init__(self, input_dim: int, num_experts: int, num_tasks: int = 1):
        super().__init__()
        
        self.num_tasks = num_tasks
        
        if num_tasks == 1:
            self.gate = nn.Linear(input_dim, num_experts)
        else:
            # Task-specific gates
            self.gates = nn.ModuleList([
                nn.Linear(input_dim, num_experts) for _ in range(num_tasks)
            ])
    
    def forward(self, x: torch.Tensor, task_id: Optional[int] = None) -> torch.Tensor:
        if self.num_tasks == 1:
            return F.softmax(self.gate(x), dim=-1)
        else:
            if task_id is not None:
                return F.softmax(self.gates[task_id](x), dim=-1)
            else:
                return [F.softmax(gate(x), dim=-1) for gate in self.gates]


class TowerNetwork(nn.Module):
    """Task-specific tower network."""
    
    def __init__(
        self, 
        input_dim: int, 
        hidden_dims: List[int],
        output_dim: int = 1,
        dropout: float = 0.2
    ):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class SharedBottomMTL(nn.Module):
    """
    Shared-Bottom Multi-Task Learning
    
    Simple architecture with shared bottom layers and task-specific towers.
    """
    
    def __init__(self, config: MTLConfig):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.user_embedding = nn.Embedding(config.num_users, config.user_embedding_dim)
        self.item_embedding = nn.Embedding(config.num_items, config.item_embedding_dim)
        
        # Shared bottom
        input_dim = config.user_embedding_dim + config.item_embedding_dim
        self.shared_bottom = ExpertNetwork(
            input_dim, 
            config.hidden_dims[:-1], 
            config.dropout
        )
        
        # Task-specific towers
        tower_input_dim = config.hidden_dims[-2] if len(config.hidden_dims) > 1 else input_dim
        self.towers = nn.ModuleDict({
            name: TowerNetwork(tower_input_dim, [config.hidden_dims[-1]], 1, config.dropout)
            for name in config.task_names
        })
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
    
    def forward(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        features: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for all tasks.
        
        Returns:
            Dictionary mapping task name to predictions
        """
        # Get embeddings
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        
        # Concatenate
        x = torch.cat([user_emb, item_emb], dim=-1)
        if features is not None:
            x = torch.cat([x, features], dim=-1)
        
        # Shared representation
        shared = self.shared_bottom(x)
        
        # Task predictions
        outputs = {}
        for name, tower in self.towers.items():
            outputs[name] = torch.sigmoid(tower(shared)).squeeze(-1)
        
        return outputs


class MMoE(nn.Module):
    """
    Multi-gate Mixture-of-Experts
    
    Paper: https://dl.acm.org/doi/10.1145/3219819.3220007
    
    Uses multiple expert networks with task-specific gating
    to learn task-specific feature representations.
    """
    
    def __init__(self, config: MTLConfig):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.user_embedding = nn.Embedding(config.num_users, config.user_embedding_dim)
        self.item_embedding = nn.Embedding(config.num_items, config.item_embedding_dim)
        
        input_dim = config.user_embedding_dim + config.item_embedding_dim
        
        # Expert networks
        self.experts = nn.ModuleList([
            ExpertNetwork(input_dim, config.hidden_dims, config.expert_dropout)
            for _ in range(config.num_experts)
        ])
        
        expert_output_dim = config.hidden_dims[-1]
        
        # Task-specific gating networks
        self.gates = nn.ModuleDict({
            name: GatingNetwork(input_dim, config.num_experts, num_tasks=1)
            for name in config.task_names
        })
        
        # Task towers
        self.towers = nn.ModuleDict({
            name: TowerNetwork(expert_output_dim, [64], 1, config.dropout)
            for name in config.task_names
        })
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
    
    def forward(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        features: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with expert routing."""
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        
        x = torch.cat([user_emb, item_emb], dim=-1)
        if features is not None:
            x = torch.cat([x, features], dim=-1)
        
        # Get expert outputs
        expert_outputs = torch.stack([
            expert(x) for expert in self.experts
        ], dim=1)  # [batch, num_experts, expert_dim]
        
        outputs = {}
        for name in self.config.task_names:
            # Task-specific gating
            gate_weights = self.gates[name](x)  # [batch, num_experts]
            
            # Weighted combination of experts
            task_input = torch.bmm(
                gate_weights.unsqueeze(1),  # [batch, 1, num_experts]
                expert_outputs  # [batch, num_experts, expert_dim]
            ).squeeze(1)  # [batch, expert_dim]
            
            outputs[name] = torch.sigmoid(self.towers[name](task_input)).squeeze(-1)
        
        return outputs


class PLE(nn.Module):
    """
    Progressive Layered Extraction (PLE)
    
    Paper: https://dl.acm.org/doi/10.1145/3383313.3412236
    
    Separates task-specific and shared experts with progressive extraction.
    """
    
    def __init__(self, config: MTLConfig, num_extraction_layers: int = 2):
        super().__init__()
        self.config = config
        self.num_layers = num_extraction_layers
        
        # Embeddings
        self.user_embedding = nn.Embedding(config.num_users, config.user_embedding_dim)
        self.item_embedding = nn.Embedding(config.num_items, config.item_embedding_dim)
        
        input_dim = config.user_embedding_dim + config.item_embedding_dim
        expert_dim = config.hidden_dims[0]
        num_experts_per_group = config.num_experts // 3  # Shared + task-specific
        
        # Extraction layers
        self.extraction_layers = nn.ModuleList()
        
        for layer in range(num_extraction_layers):
            layer_input_dim = input_dim if layer == 0 else expert_dim
            
            layer_modules = {
                # Shared experts
                "shared_experts": nn.ModuleList([
                    ExpertNetwork(layer_input_dim, [expert_dim], config.expert_dropout)
                    for _ in range(num_experts_per_group)
                ]),
                # Task-specific experts
                "task_experts": nn.ModuleDict({
                    name: nn.ModuleList([
                        ExpertNetwork(layer_input_dim, [expert_dim], config.expert_dropout)
                        for _ in range(num_experts_per_group)
                    ])
                    for name in config.task_names
                }),
                # Gates
                "gates": nn.ModuleDict({
                    name: nn.Linear(layer_input_dim, num_experts_per_group * 2)
                    for name in config.task_names
                }),
                "shared_gate": nn.Linear(layer_input_dim, num_experts_per_group * (1 + len(config.task_names)))
            }
            
            self.extraction_layers.append(nn.ModuleDict(layer_modules))
        
        # Task towers
        self.towers = nn.ModuleDict({
            name: TowerNetwork(expert_dim, [64], 1, config.dropout)
            for name in config.task_names
        })
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
    
    def forward(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        features: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward with progressive extraction."""
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        
        x = torch.cat([user_emb, item_emb], dim=-1)
        if features is not None:
            x = torch.cat([x, features], dim=-1)
        
        # Task-specific representations
        task_inputs = {name: x for name in self.config.task_names}
        shared_input = x
        
        for layer_modules in self.extraction_layers:
            # Shared expert outputs
            shared_expert_outputs = torch.stack([
                expert(shared_input) for expert in layer_modules["shared_experts"]
            ], dim=1)
            
            new_task_inputs = {}
            
            for name in self.config.task_names:
                # Task-specific expert outputs
                task_expert_outputs = torch.stack([
                    expert(task_inputs[name]) 
                    for expert in layer_modules["task_experts"][name]
                ], dim=1)
                
                # Combine shared and task experts
                all_expert_outputs = torch.cat([
                    task_expert_outputs, shared_expert_outputs
                ], dim=1)
                
                # Gating
                gate_weights = F.softmax(
                    layer_modules["gates"][name](task_inputs[name]), dim=-1
                )
                
                # Weighted combination
                new_task_inputs[name] = torch.bmm(
                    gate_weights.unsqueeze(1),
                    all_expert_outputs
                ).squeeze(1)
            
            task_inputs = new_task_inputs
            
            # Update shared input
            all_task_experts = [shared_expert_outputs]
            for name in self.config.task_names:
                task_expert_outputs = torch.stack([
                    expert(task_inputs[name])
                    for expert in layer_modules["task_experts"][name]
                ], dim=1)
                all_task_experts.append(task_expert_outputs)
            
            all_experts = torch.cat(all_task_experts, dim=1)
            shared_gate = F.softmax(layer_modules["shared_gate"](shared_input), dim=-1)
            shared_input = torch.bmm(shared_gate.unsqueeze(1), all_experts).squeeze(1)
        
        # Task predictions
        outputs = {}
        for name in self.config.task_names:
            outputs[name] = torch.sigmoid(self.towers[name](task_inputs[name])).squeeze(-1)
        
        return outputs


class ESMM(nn.Module):
    """
    Entire Space Multi-Task Model
    
    Paper: https://arxiv.org/abs/1804.07931
    
    Models CVR using the entire sample space through:
    pCVR = pCTCVR / pCTR
    
    This addresses sample selection bias in CVR prediction.
    """
    
    def __init__(self, config: MTLConfig):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.user_embedding = nn.Embedding(config.num_users, config.user_embedding_dim)
        self.item_embedding = nn.Embedding(config.num_items, config.item_embedding_dim)
        
        input_dim = config.user_embedding_dim + config.item_embedding_dim
        
        # CTR tower (impression -> click)
        self.ctr_tower = TowerNetwork(
            input_dim, config.hidden_dims, 1, config.dropout
        )
        
        # CVR tower (click -> conversion)
        self.cvr_tower = TowerNetwork(
            input_dim, config.hidden_dims, 1, config.dropout
        )
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
    
    def forward(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        features: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with ESMM formulation.
        
        Returns:
            ctr: P(click|impression)
            cvr: P(conversion|click)  
            ctcvr: P(conversion|impression) = ctr * cvr
        """
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        
        x = torch.cat([user_emb, item_emb], dim=-1)
        if features is not None:
            x = torch.cat([x, features], dim=-1)
        
        # CTR prediction
        ctr = torch.sigmoid(self.ctr_tower(x)).squeeze(-1)
        
        # CVR prediction (in entire space)
        cvr = torch.sigmoid(self.cvr_tower(x)).squeeze(-1)
        
        # CTCVR = CTR * CVR
        ctcvr = ctr * cvr
        
        return {
            "ctr": ctr,
            "cvr": cvr,
            "ctcvr": ctcvr
        }


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss with uncertainty weighting.
    
    Automatically learns task weights based on homoscedastic uncertainty.
    Paper: https://arxiv.org/abs/1705.07115
    """
    
    def __init__(self, num_tasks: int, task_names: List[str]):
        super().__init__()
        self.num_tasks = num_tasks
        self.task_names = task_names
        
        # Learnable log variance for each task
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        loss_fn: nn.Module = nn.BCELoss(reduction='none')
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute weighted multi-task loss.
        
        Returns:
            total_loss: Combined loss
            task_losses: Individual task losses
        """
        total_loss = 0
        task_losses = {}
        
        for i, name in enumerate(self.task_names):
            if name not in predictions or name not in targets:
                continue
            
            pred = predictions[name]
            target = targets[name]
            
            # Task loss
            task_loss = loss_fn(pred, target).mean()
            
            # Uncertainty weighting
            precision = torch.exp(-self.log_vars[i])
            weighted_loss = precision * task_loss + self.log_vars[i]
            
            total_loss += weighted_loss
            task_losses[name] = task_loss.item()
        
        return total_loss, task_losses


class MultiTaskRecommender:
    """
    High-level interface for multi-task recommendation.
    """
    
    def __init__(
        self,
        config: MTLConfig,
        model_type: str = "mmoe",  # shared_bottom, mmoe, ple, esmm
        device: str = "cuda"
    ):
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        if model_type == "shared_bottom":
            self.model = SharedBottomMTL(config)
        elif model_type == "mmoe":
            self.model = MMoE(config)
        elif model_type == "ple":
            self.model = PLE(config)
        elif model_type == "esmm":
            self.model = ESMM(config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.model = self.model.to(self.device)
        self.loss_fn = MultiTaskLoss(config.num_tasks, config.task_names).to(self.device)
    
    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        optimizer: torch.optim.Optimizer
    ) -> Dict[str, float]:
        """Execute one training step."""
        self.model.train()
        optimizer.zero_grad()
        
        user_ids = batch["user_ids"].to(self.device)
        item_ids = batch["item_ids"].to(self.device)
        
        predictions = self.model(user_ids, item_ids)
        
        targets = {
            name: batch[name].to(self.device).float()
            for name in self.config.task_names
            if name in batch
        }
        
        loss, task_losses = self.loss_fn(predictions, targets)
        loss.backward()
        optimizer.step()
        
        return {"total_loss": loss.item(), **task_losses}
    
    @torch.no_grad()
    def predict(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Get predictions for all tasks."""
        self.model.eval()
        return self.model(
            user_ids.to(self.device),
            item_ids.to(self.device)
        )
    
    def save(self, path: str):
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "loss_fn_state_dict": self.loss_fn.state_dict(),
            "config": self.config
        }, path)
    
    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.loss_fn.load_state_dict(checkpoint["loss_fn_state_dict"])


if __name__ == "__main__":
    # Example usage
    config = MTLConfig(
        num_users=10000,
        num_items=5000,
        task_names=["ctr", "cvr", "engagement"],
        num_experts=6
    )
    
    recommender = MultiTaskRecommender(config, model_type="mmoe", device="cpu")
    
    # Sample batch
    batch = {
        "user_ids": torch.randint(0, 10000, (32,)),
        "item_ids": torch.randint(0, 5000, (32,)),
        "ctr": torch.rand(32),
        "cvr": torch.rand(32),
        "engagement": torch.rand(32)
    }
    
    optimizer = torch.optim.Adam(recommender.model.parameters(), lr=0.001)
    losses = recommender.train_step(batch, optimizer)
    print(f"Losses: {losses}")
