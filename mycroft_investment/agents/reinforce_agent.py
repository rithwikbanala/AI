"""
REINFORCE Agent for Portfolio Optimization

Implements policy gradient reinforcement learning using REINFORCE algorithm.
The agent learns a parameterized policy that directly optimizes portfolio allocation.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple, Dict, Optional
import pickle


class PolicyNetwork(nn.Module):
    """
    Neural network policy for REINFORCE agent.
    
    Maps states to action probabilities for portfolio allocation strategies.
    """
    
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
        """
        Initialize policy network.
        
        Args:
            state_size: Size of state vector
            action_size: Number of possible actions
            hidden_size: Size of hidden layers
        """
        super(PolicyNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, action_size),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through network.
        
        Args:
            state: State tensor
            
        Returns:
            Action probabilities
        """
        return self.network(state)


class REINFORCEAgent:
    """
    REINFORCE (Monte Carlo Policy Gradient) agent for portfolio optimization.
    
    Uses policy gradient method to directly optimize the policy
    without learning value functions.
    """
    
    def __init__(
        self,
        state_size: int,
        action_size: int,
        portfolio_size: int,
        learning_rate: float = 0.001,
        discount_factor: float = 0.99,
        hidden_size: int = 128,
        device: Optional[str] = None
    ):
        """
        Initialize REINFORCE agent.
        
        Args:
            state_size: Size of state vector
            action_size: Number of possible actions
            portfolio_size: Number of stocks in portfolio
            learning_rate: Policy learning rate
            discount_factor: Future reward discount (gamma)
            hidden_size: Hidden layer size for neural network
            device: PyTorch device ('cpu' or 'cuda')
        """
        self.state_size = state_size
        self.action_size = action_size
        self.portfolio_size = portfolio_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Initialize policy network
        self.policy_network = PolicyNetwork(state_size, action_size, hidden_size).to(self.device)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        
        # Episode storage for REINFORCE
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []
        self.episode_log_probs = []
        
        # Training statistics
        self.training_history = {
            'episode_rewards': [],
            'episode_lengths': [],
            'policy_losses': [],
            'entropy_values': []
        }
    
    def _get_portfolio_weights(self, action: int) -> np.ndarray:
        """
        Convert discrete action to portfolio weights.
        
        Args:
            action: Discrete action index
            
        Returns:
            Portfolio weights array (size = portfolio_size)
        """
        weights = np.zeros(self.portfolio_size)
        
        if self.portfolio_size == 1:
            weights[0] = 1.0
        elif action == 0:
            # Equal weight
            weights[:] = 1.0 / self.portfolio_size
        elif action == 1:
            # Concentrated: all in first stock
            weights[0] = 1.0
        elif action == 2:
            # Concentrated: all in second stock
            if self.portfolio_size > 1:
                weights[1] = 1.0
        elif action == 3:
            # Top 2 stocks
            if self.portfolio_size >= 2:
                weights[:2] = 0.5
        elif action == 4:
            # Top 3 stocks
            if self.portfolio_size >= 3:
                weights[:3] = 1.0 / 3.0
        elif action == 5:
            # Top half
            top_half = self.portfolio_size // 2
            weights[:top_half] = 1.0 / top_half
        else:
            # Custom: distribute based on action index
            num_active = min((action % (self.portfolio_size - 1)) + 1, self.portfolio_size)
            weights[:num_active] = 1.0 / num_active
        
        return weights
    
    def select_action(self, state: np.ndarray, training: bool = True) -> Tuple[int, np.ndarray, float]:
        """
        Select action from policy.
        
        Args:
            state: Current state
            training: Whether in training mode
            
        Returns:
            (action, portfolio_weights, log_prob): Selected action, weights, and log probability
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Get action probabilities
        with torch.no_grad():
            action_probs = self.policy_network(state_tensor)
        
        # Sample action
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        
        # Get portfolio weights
        weights = self._get_portfolio_weights(action.item())
        
        return action.item(), weights, log_prob.item()
    
    def record_step(self, state: np.ndarray, action: int, reward: float, log_prob: float):
        """
        Record step for REINFORCE update.
        
        Args:
            state: State
            action: Action taken
            reward: Reward received
            log_prob: Log probability of action
        """
        self.episode_states.append(state)
        self.episode_actions.append(action)
        self.episode_rewards.append(reward)
        self.episode_log_probs.append(log_prob)
    
    def train_step(self):
        """
        Perform REINFORCE update at end of episode.
        
        REINFORCE update:
        gradient = E[G_t * grad(log pi(a_t|s_t))]
        where G_t is the discounted return from time t
        """
        if len(self.episode_rewards) == 0:
            return
        
        # Calculate discounted returns
        returns = []
        G = 0
        for reward in reversed(self.episode_rewards):
            G = reward + self.discount_factor * G
            returns.insert(0, G)
        
        # Normalize returns (reduces variance)
        returns = np.array(returns)
        if len(returns) > 1 and returns.std() > 0:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(self.episode_states)).to(self.device)
        actions = torch.LongTensor(self.episode_actions).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Get action probabilities for all states
        action_probs = self.policy_network(states)
        action_dist = torch.distributions.Categorical(action_probs)
        log_probs = action_dist.log_prob(actions)
        
        # Calculate policy loss (negative because we maximize)
        policy_loss = -(log_probs * returns).mean()
        
        # Add entropy bonus for exploration
        entropy = action_dist.entropy().mean()
        entropy_bonus = 0.01 * entropy
        total_loss = policy_loss - entropy_bonus
        
        # Update policy
        self.optimizer.zero_grad()
        total_loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Record statistics
        total_reward = sum(self.episode_rewards)
        episode_length = len(self.episode_rewards)
        
        self.training_history['episode_rewards'].append(total_reward)
        self.training_history['episode_lengths'].append(episode_length)
        self.training_history['policy_losses'].append(policy_loss.item())
        self.training_history['entropy_values'].append(entropy.item())
        
        # Clear episode buffer
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []
        self.episode_log_probs = []
    
    def end_episode(self):
        """Called at end of episode"""
        # REINFORCE updates happen here
        self.train_step()
    
    def record_episode(self, total_reward: float, episode_length: int):
        """
        Record episode statistics.
        
        Args:
            total_reward: Total reward for episode
            episode_length: Length of episode
        """
        # Already recorded in train_step
        pass
    
    def get_policy(self, state: np.ndarray) -> np.ndarray:
        """
        Get action probabilities for state.
        
        Args:
            state: Current state
            
        Returns:
            Action probabilities
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_probs = self.policy_network(state_tensor)
        return action_probs.cpu().numpy().flatten()
    
    def save(self, filepath: str):
        """Save agent to file"""
        data = {
            'policy_state_dict': self.policy_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_history': self.training_history,
            'state_size': self.state_size,
            'action_size': self.action_size,
            'portfolio_size': self.portfolio_size,
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor
        }
        torch.save(data, filepath)
    
    def load(self, filepath: str):
        """Load agent from file"""
        data = torch.load(filepath, map_location=self.device)
        self.policy_network.load_state_dict(data['policy_state_dict'])
        self.optimizer.load_state_dict(data['optimizer_state_dict'])
        self.training_history = data['training_history']
        self.state_size = data['state_size']
        self.action_size = data['action_size']
        self.portfolio_size = data['portfolio_size']
        self.learning_rate = data['learning_rate']
        self.discount_factor = data['discount_factor']
    
    def get_statistics(self) -> Dict:
        """Get training statistics"""
        if len(self.training_history['episode_rewards']) == 0:
            return {}
        
        recent_rewards = self.training_history['episode_rewards'][-100:]
        return {
            'mean_reward': np.mean(recent_rewards),
            'std_reward': np.std(recent_rewards),
            'mean_episode_length': np.mean(self.training_history['episode_lengths'][-100:]),
            'mean_policy_loss': np.mean(self.training_history['policy_losses'][-100:]),
            'mean_entropy': np.mean(self.training_history['entropy_values'][-100:])
        }

