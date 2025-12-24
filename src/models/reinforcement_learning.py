"""
Reinforcement Learning for Recommendations

Implements RL-based recommendation algorithms that optimize for long-term user engagement:
- DQN (Deep Q-Network) for discrete action space
- Actor-Critic with slate recommendations
- REINFORCE with baseline
- Soft Actor-Critic (SAC) for continuous embeddings
- Offline RL with Conservative Q-Learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Dict, List, Optional, Tuple, NamedTuple
from dataclasses import dataclass, field
from collections import deque
import numpy as np
import random


@dataclass
class RLConfig:
    """Configuration for RL-based recommender."""
    # Environment
    num_users: int = 50000
    num_items: int = 10000
    state_dim: int = 256
    action_dim: int = 10000  # Same as num_items for item selection
    
    # Network
    hidden_dims: List[int] = field(default_factory=lambda: [512, 256, 128])
    embedding_dim: int = 64
    
    # RL hyperparameters
    gamma: float = 0.99  # Discount factor
    tau: float = 0.005  # Soft update coefficient
    learning_rate: float = 0.0003
    buffer_size: int = 100000
    batch_size: int = 256
    
    # Exploration
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    
    # Training
    update_every: int = 4
    target_update_every: int = 100


class Experience(NamedTuple):
    """Single experience tuple."""
    state: torch.Tensor
    action: int
    reward: float
    next_state: torch.Tensor
    done: bool


class ReplayBuffer:
    """Experience replay buffer for off-policy learning."""
    
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, experience: Experience):
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> List[Experience]:
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def __len__(self) -> int:
        return len(self.buffer)


class PrioritizedReplayBuffer:
    """Prioritized experience replay for important experiences."""
    
    def __init__(self, capacity: int, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
    
    def push(self, experience: Experience, priority: Optional[float] = None):
        max_priority = self.priorities.max() if self.buffer else 1.0
        priority = priority if priority is not None else max_priority
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        
        self.priorities[self.position] = priority ** self.alpha
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int, beta: float = 0.4) -> Tuple[List[Experience], np.ndarray, np.ndarray]:
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        
        priorities = self.priorities[:len(self.buffer)]
        probs = priorities / priorities.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)
        experiences = [self.buffer[i] for i in indices]
        
        # Importance sampling weights
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights = weights / weights.max()
        
        return experiences, indices, weights
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = (priority + 1e-5) ** self.alpha
    
    def __len__(self) -> int:
        return len(self.buffer)


class StateEncoder(nn.Module):
    """Encode user state from history."""
    
    def __init__(self, config: RLConfig):
        super().__init__()
        self.item_embedding = nn.Embedding(config.num_items + 1, config.embedding_dim, padding_idx=0)
        
        self.attention = nn.MultiheadAttention(
            config.embedding_dim, 
            num_heads=4, 
            dropout=0.1,
            batch_first=True
        )
        
        self.fc = nn.Sequential(
            nn.Linear(config.embedding_dim, config.state_dim),
            nn.ReLU(),
            nn.LayerNorm(config.state_dim)
        )
    
    def forward(self, history: torch.Tensor, history_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            history: Item history [batch, seq_len]
            history_mask: Padding mask [batch, seq_len]
            
        Returns:
            State representation [batch, state_dim]
        """
        emb = self.item_embedding(history)  # [batch, seq_len, embed_dim]
        
        # Self-attention
        attn_out, _ = self.attention(emb, emb, emb, key_padding_mask=history_mask)
        
        # Mean pooling
        if history_mask is not None:
            mask = (~history_mask).unsqueeze(-1).float()
            pooled = (attn_out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            pooled = attn_out.mean(dim=1)
        
        return self.fc(pooled)


class DQN(nn.Module):
    """
    Deep Q-Network for item recommendation.
    
    Action space: Select one item from catalog.
    """
    
    def __init__(self, config: RLConfig):
        super().__init__()
        self.config = config
        
        self.state_encoder = StateEncoder(config)
        
        # Q-network
        layers = []
        prev_dim = config.state_dim
        for hidden_dim in config.hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        self.fc = nn.Sequential(*layers)
        self.q_head = nn.Linear(prev_dim, config.action_dim)
    
    def forward(self, history: torch.Tensor, history_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            history: User history [batch, seq_len]
            
        Returns:
            Q-values for all items [batch, num_items]
        """
        state = self.state_encoder(history, history_mask)
        features = self.fc(state)
        return self.q_head(features)
    
    def get_action(
        self, 
        history: torch.Tensor, 
        epsilon: float = 0.0,
        exclude_items: Optional[List[int]] = None
    ) -> int:
        """Select action using epsilon-greedy policy."""
        if random.random() < epsilon:
            # Random exploration
            valid_actions = list(range(self.config.action_dim))
            if exclude_items:
                valid_actions = [a for a in valid_actions if a not in exclude_items]
            return random.choice(valid_actions)
        
        with torch.no_grad():
            q_values = self.forward(history.unsqueeze(0))
            
            if exclude_items:
                q_values[0, exclude_items] = float('-inf')
            
            return q_values.argmax(dim=1).item()


class DuelingDQN(nn.Module):
    """
    Dueling DQN: Separate value and advantage streams.
    
    Q(s,a) = V(s) + A(s,a) - mean(A(s,:))
    """
    
    def __init__(self, config: RLConfig):
        super().__init__()
        self.config = config
        
        self.state_encoder = StateEncoder(config)
        
        # Shared feature layer
        self.shared = nn.Sequential(
            nn.Linear(config.state_dim, config.hidden_dims[0]),
            nn.ReLU()
        )
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(config.hidden_dims[0], config.hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(config.hidden_dims[1], 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(config.hidden_dims[0], config.hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(config.hidden_dims[1], config.action_dim)
        )
    
    def forward(self, history: torch.Tensor, history_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        state = self.state_encoder(history, history_mask)
        shared = self.shared(state)
        
        value = self.value_stream(shared)  # [batch, 1]
        advantage = self.advantage_stream(shared)  # [batch, num_items]
        
        # Combine: Q = V + (A - mean(A))
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values


class ActorNetwork(nn.Module):
    """Actor network for policy gradient methods."""
    
    def __init__(self, config: RLConfig):
        super().__init__()
        self.state_encoder = StateEncoder(config)
        
        self.fc = nn.Sequential(
            nn.Linear(config.state_dim, config.hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(config.hidden_dims[0], config.hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(config.hidden_dims[1], config.action_dim)
        )
    
    def forward(self, history: torch.Tensor, history_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        state = self.state_encoder(history, history_mask)
        logits = self.fc(state)
        return F.softmax(logits, dim=-1)
    
    def get_action(self, history: torch.Tensor, deterministic: bool = False) -> Tuple[int, torch.Tensor]:
        probs = self.forward(history.unsqueeze(0))
        
        if deterministic:
            action = probs.argmax(dim=-1).item()
            log_prob = torch.log(probs[0, action])
        else:
            dist = Categorical(probs)
            action_tensor = dist.sample()
            action = action_tensor.item()
            log_prob = dist.log_prob(action_tensor)
        
        return action, log_prob


class CriticNetwork(nn.Module):
    """Critic network for value estimation."""
    
    def __init__(self, config: RLConfig):
        super().__init__()
        self.state_encoder = StateEncoder(config)
        
        self.fc = nn.Sequential(
            nn.Linear(config.state_dim, config.hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(config.hidden_dims[0], config.hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(config.hidden_dims[1], 1)
        )
    
    def forward(self, history: torch.Tensor, history_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        state = self.state_encoder(history, history_mask)
        return self.fc(state)


class A2C(nn.Module):
    """
    Advantage Actor-Critic for recommendations.
    
    Combines policy gradient (actor) with value function (critic)
    for reduced variance.
    """
    
    def __init__(self, config: RLConfig):
        super().__init__()
        self.config = config
        
        self.actor = ActorNetwork(config)
        self.critic = CriticNetwork(config)
    
    def get_action(self, history: torch.Tensor, deterministic: bool = False) -> Tuple[int, torch.Tensor, torch.Tensor]:
        action, log_prob = self.actor.get_action(history, deterministic)
        value = self.critic(history.unsqueeze(0))
        return action, log_prob, value
    
    def evaluate(self, history: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        probs = self.actor(history)
        dist = Categorical(probs)
        
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        values = self.critic(history)
        
        return log_probs, values, entropy


class SlateRecommender(nn.Module):
    """
    Slate recommendation with RL.
    
    Recommends a slate of k items instead of single item.
    Uses sequential selection with positional embeddings.
    """
    
    def __init__(self, config: RLConfig, slate_size: int = 10):
        super().__init__()
        self.config = config
        self.slate_size = slate_size
        
        self.state_encoder = StateEncoder(config)
        self.item_embedding = nn.Embedding(config.num_items, config.embedding_dim)
        self.position_embedding = nn.Embedding(slate_size, config.embedding_dim)
        
        # Pointer network for sequential selection
        self.query = nn.Linear(config.state_dim + config.embedding_dim, config.hidden_dims[0])
        self.key = nn.Linear(config.embedding_dim, config.hidden_dims[0])
        self.attention = nn.Linear(config.hidden_dims[0], 1)
        
        # Value network
        self.value_head = nn.Sequential(
            nn.Linear(config.state_dim + slate_size * config.embedding_dim, config.hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(config.hidden_dims[0], 1)
        )
    
    def forward(
        self, 
        history: torch.Tensor,
        history_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate slate of recommendations.
        
        Returns:
            slate: Selected item indices [batch, slate_size]
            log_probs: Log probabilities [batch, slate_size]
        """
        batch_size = history.size(0)
        device = history.device
        
        state = self.state_encoder(history, history_mask)  # [batch, state_dim]
        
        # All item embeddings
        all_items = torch.arange(self.config.num_items, device=device)
        item_embs = self.item_embedding(all_items)  # [num_items, embed_dim]
        
        slate = []
        log_probs = []
        selected_mask = torch.zeros(batch_size, self.config.num_items, device=device).bool()
        
        # Context starts with zero
        context = torch.zeros(batch_size, self.config.embedding_dim, device=device)
        
        for pos in range(self.slate_size):
            # Add position embedding to context
            pos_emb = self.position_embedding(torch.tensor([pos], device=device))
            context_with_pos = context + pos_emb
            
            # Query
            query_input = torch.cat([state, context_with_pos], dim=-1)
            query = self.query(query_input)  # [batch, hidden]
            
            # Attention over items
            keys = self.key(item_embs)  # [num_items, hidden]
            
            attn_scores = self.attention(
                torch.tanh(query.unsqueeze(1) + keys.unsqueeze(0))
            ).squeeze(-1)  # [batch, num_items]
            
            # Mask selected items
            attn_scores = attn_scores.masked_fill(selected_mask, float('-inf'))
            
            # Sample
            probs = F.softmax(attn_scores, dim=-1)
            dist = Categorical(probs)
            action = dist.sample()
            
            slate.append(action)
            log_probs.append(dist.log_prob(action))
            
            # Update
            selected_mask.scatter_(1, action.unsqueeze(1), True)
            context = item_embs[action]
        
        slate = torch.stack(slate, dim=1)
        log_probs = torch.stack(log_probs, dim=1)
        
        return slate, log_probs
    
    def get_value(self, history: torch.Tensor, slate: torch.Tensor) -> torch.Tensor:
        """Get value estimate for state-slate pair."""
        state = self.state_encoder(history)
        slate_embs = self.item_embedding(slate).view(slate.size(0), -1)
        
        value_input = torch.cat([state, slate_embs], dim=-1)
        return self.value_head(value_input)


class DQNAgent:
    """
    Complete DQN agent with training loop.
    """
    
    def __init__(self, config: RLConfig, device: str = "cuda"):
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # Networks
        self.policy_net = DuelingDQN(config).to(self.device)
        self.target_net = DuelingDQN(config).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.policy_net.parameters(), 
            lr=config.learning_rate
        )
        
        # Replay buffer
        self.buffer = PrioritizedReplayBuffer(config.buffer_size)
        
        # Exploration
        self.epsilon = config.epsilon_start
        
        self.steps = 0
    
    def select_action(self, history: torch.Tensor, exclude_items: Optional[List[int]] = None) -> int:
        return self.policy_net.get_action(
            history.to(self.device), 
            self.epsilon,
            exclude_items
        )
    
    def store_experience(self, experience: Experience):
        self.buffer.push(experience)
    
    def update(self, beta: float = 0.4) -> Optional[float]:
        """Perform one update step."""
        if len(self.buffer) < self.config.batch_size:
            return None
        
        self.steps += 1
        
        # Sample batch
        experiences, indices, weights = self.buffer.sample(
            self.config.batch_size, beta
        )
        
        # Unpack
        states = torch.stack([e.state for e in experiences]).to(self.device)
        actions = torch.tensor([e.action for e in experiences], device=self.device)
        rewards = torch.tensor([e.reward for e in experiences], device=self.device, dtype=torch.float)
        next_states = torch.stack([e.next_state for e in experiences]).to(self.device)
        dones = torch.tensor([e.done for e in experiences], device=self.device, dtype=torch.float)
        weights = torch.tensor(weights, device=self.device, dtype=torch.float)
        
        # Current Q values
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Target Q values (Double DQN)
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(dim=1)
            next_q = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q = rewards + self.config.gamma * next_q * (1 - dones)
        
        # TD error for prioritized replay
        td_errors = torch.abs(current_q - target_q).detach().cpu().numpy()
        self.buffer.update_priorities(indices, td_errors)
        
        # Weighted loss
        loss = (weights * F.smooth_l1_loss(current_q, target_q, reduction='none')).mean()
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        # Soft update target network
        if self.steps % self.config.target_update_every == 0:
            for target_param, policy_param in zip(
                self.target_net.parameters(), 
                self.policy_net.parameters()
            ):
                target_param.data.copy_(
                    self.config.tau * policy_param.data + 
                    (1 - self.config.tau) * target_param.data
                )
        
        # Decay epsilon
        self.epsilon = max(
            self.config.epsilon_end,
            self.epsilon * self.config.epsilon_decay
        )
        
        return loss.item()
    
    def save(self, path: str):
        torch.save({
            "policy_net": self.policy_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "steps": self.steps
        }, path)
    
    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint["epsilon"]
        self.steps = checkpoint["steps"]


class A2CAgent:
    """Actor-Critic agent for recommendations."""
    
    def __init__(self, config: RLConfig, device: str = "cuda"):
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        self.model = A2C(config).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=config.learning_rate
        )
        
        self.gamma = config.gamma
        self.entropy_coef = 0.01
        self.value_coef = 0.5
    
    def update(
        self, 
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor
    ) -> Dict[str, float]:
        """Update actor and critic."""
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Get current values and next values
        log_probs, values, entropy = self.model.evaluate(states, actions)
        
        with torch.no_grad():
            next_values = self.model.critic(next_states)
        
        # Compute advantages
        td_target = rewards + self.gamma * next_values.squeeze() * (1 - dones)
        advantages = td_target - values.squeeze()
        
        # Losses
        actor_loss = -(log_probs * advantages.detach()).mean()
        critic_loss = F.mse_loss(values.squeeze(), td_target)
        entropy_loss = -entropy.mean()
        
        total_loss = actor_loss + self.value_coef * critic_loss + self.entropy_coef * entropy_loss
        
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        self.optimizer.step()
        
        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "entropy": -entropy_loss.item()
        }


if __name__ == "__main__":
    # Example usage
    config = RLConfig(
        num_users=1000,
        num_items=5000,
        state_dim=128,
        hidden_dims=[256, 128]
    )
    
    agent = DQNAgent(config, device="cpu")
    
    # Simulate some experiences
    for _ in range(1000):
        history = torch.randint(0, 5000, (20,))
        action = agent.select_action(history)
        
        next_history = torch.cat([history[1:], torch.tensor([action])])
        reward = np.random.random()
        done = np.random.random() < 0.1
        
        agent.store_experience(Experience(
            state=history,
            action=action,
            reward=reward,
            next_state=next_history,
            done=done
        ))
    
    # Train
    for _ in range(100):
        loss = agent.update()
        if loss:
            print(f"Loss: {loss:.4f}, Epsilon: {agent.epsilon:.4f}")
