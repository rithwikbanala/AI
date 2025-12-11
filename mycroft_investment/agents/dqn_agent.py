"""
Deep Q-Network (DQN) Agent for Portfolio Optimization

Implements DQN with experience replay and target network for stable learning.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, List, Dict, Optional
from collections import deque
import random
import pickle


class DQNNetwork(nn.Module):
    """
    Deep Q-Network for portfolio optimization.
    
    Takes market state and outputs Q-values for each action.
    """
    
    def __init__(self, state_size: int, action_size: int, hidden_sizes: List[int] = [128, 128, 64]):
        """
        Initialize DQN network.
        
        Args:
            state_size: Size of state vector
            action_size: Number of possible actions
            hidden_sizes: List of hidden layer sizes
        """
        super(DQNNetwork, self).__init__()
        
        layers = []
        input_size = state_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            input_size = hidden_size
        
        layers.append(nn.Linear(input_size, action_size))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            state: State tensor
            
        Returns:
            Q-values for each action
        """
        return self.network(state)


class DQNAgent:
    """
    DQN Agent for portfolio optimization.
    
    Uses experience replay and target network for stable learning.
    """
    
    def __init__(
        self,
        state_size: int,
        action_size: int,
        portfolio_size: Optional[int] = None,
        learning_rate: float = 0.001,
        discount_factor: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        memory_size: int = 10000,
        batch_size: int = 64,
        target_update_freq: int = 100,
        device: Optional[str] = None
    ):
        """
        Initialize DQN agent.
        
        Args:
            state_size: Size of state vector
            action_size: Number of actions
            learning_rate: Learning rate for optimizer
            discount_factor: Future reward discount (gamma)
            epsilon: Initial exploration rate
            epsilon_decay: Epsilon decay per episode
            epsilon_min: Minimum exploration rate
            memory_size: Size of experience replay buffer
            batch_size: Batch size for training
            target_update_freq: Frequency of target network updates
            device: PyTorch device ('cpu' or 'cuda')
        """
        self.state_size = state_size
        self.action_size = action_size
        self.portfolio_size = portfolio_size or action_size  # Actual number of stocks
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Q-Network and Target Network
        self.q_network = DQNNetwork(state_size, action_size).to(self.device)
        self.target_network = DQNNetwork(state_size, action_size).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Experience replay buffer
        self.memory = deque(maxlen=memory_size)
        
        # Training statistics
        self.training_history = {
            'episode_rewards': [],
            'episode_lengths': [],
            'epsilon_values': [],
            'losses': [],
            'q_values': []
        }
        
        self.step_count = 0
    
    def _get_portfolio_weights(self, action: int) -> np.ndarray:
        """
        Convert discrete action to portfolio weights.
        
        Actions represent different allocation strategies:
        - Equal weight
        - Concentrated (single stock)
        - Diversified (multiple stocks)
        - Market cap weighted
        - Momentum based
        etc.
        
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
    
    def select_action(self, state: np.ndarray, training: bool = True) -> Tuple[int, np.ndarray]:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state
            training: Whether in training mode
            
        Returns:
            (action_index, portfolio_weights): Action and corresponding weights
        """
        if training and np.random.random() < self.epsilon:
            # Explore: random action
            action = np.random.randint(self.action_size)
        else:
            # Exploit: best action according to Q-network
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            action = q_values.argmax().item()
        
        weights = self._get_portfolio_weights(action)
        return action, weights
    
    def remember(self, state: np.ndarray, action: int, reward: float, 
                 next_state: np.ndarray, done: bool):
        """
        Store experience in replay buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self) -> Optional[float]:
        """
        Train on a batch of experiences from replay buffer.
        
        Returns:
            Loss value if training occurred, None otherwise
        """
        if len(self.memory) < self.batch_size:
            return None
        
        # Sample batch
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        # Current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q-values from target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.discount_factor * next_q_values * ~dones)
        
        # Compute loss
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Update target network periodically
        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        return loss.item()
    
    def end_episode(self):
        """Called at end of episode"""
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def record_episode(self, total_reward: float, episode_length: int):
        """
        Record episode statistics.
        
        Args:
            total_reward: Total reward for episode
            episode_length: Length of episode
        """
        self.training_history['episode_rewards'].append(total_reward)
        self.training_history['episode_lengths'].append(episode_length)
        self.training_history['epsilon_values'].append(self.epsilon)
    
    def get_statistics(self) -> Dict:
        """Get training statistics"""
        if len(self.training_history['episode_rewards']) == 0:
            return {}
        
        recent_rewards = self.training_history['episode_rewards'][-100:]
        return {
            'mean_reward': np.mean(recent_rewards),
            'std_reward': np.std(recent_rewards),
            'mean_episode_length': np.mean(self.training_history['episode_lengths'][-100:]),
            'current_epsilon': self.epsilon,
            'memory_size': len(self.memory)
        }
    
    def save(self, filepath: str):
        """Save agent to file"""
        data = {
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_history': self.training_history,
            'state_size': self.state_size,
            'action_size': self.action_size,
            'epsilon': self.epsilon,
            'step_count': self.step_count
        }
        torch.save(data, filepath)
    
    def load(self, filepath: str):
        """Load agent from file"""
        data = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(data['q_network_state_dict'])
        self.target_network.load_state_dict(data['target_network_state_dict'])
        self.optimizer.load_state_dict(data['optimizer_state_dict'])
        self.training_history = data['training_history']
        self.epsilon = data['epsilon']
        self.step_count = data.get('step_count', 0)

