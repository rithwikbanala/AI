"""
Portfolio Agent Orchestrator

Coordinates DQN agent with stock market environment.
Manages portfolio allocation and performance tracking.
"""

import numpy as np
import sys
import os
from typing import Dict, List, Optional

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Use relative imports
from market.stock_market import StockMarket
from .dqn_agent import DQNAgent
from .reinforce_agent import REINFORCEAgent


class PortfolioAgent:
    """
    Main portfolio agent that manages investment strategy.
    
    Uses DQN agent to learn optimal portfolio allocation.
    """
    
    def __init__(
        self,
        market: Optional[StockMarket] = None,
        initial_cash: float = 100000.0,
        agent_type: str = 'dqn',
        **agent_kwargs
    ):
        """
        Initialize portfolio agent.
        
        Args:
            market: Stock market environment (creates new one if None)
            initial_cash: Starting cash amount
            agent_type: 'dqn' or 'reinforce'
            **agent_kwargs: Additional arguments for RL agent
        """
        # Initialize market
        if market is None:
            self.market = StockMarket(initial_cash=initial_cash)
        else:
            self.market = market
        
        state_size = self.market.get_state_size()
        action_size = self.market.get_action_size()
        
        # Create action space (discrete actions representing different allocation strategies)
        # We'll use more actions than just num_stocks for better strategy diversity
        num_allocation_strategies = max(10, action_size * 2)  # At least 10 strategies
        
        # Store agent type
        self.agent_type = agent_type.lower()
        
        # Initialize RL agent
        if self.agent_type == 'dqn':
            self.agent = DQNAgent(
                state_size=state_size,
                action_size=num_allocation_strategies,
                portfolio_size=action_size,  # Pass actual portfolio size
                **agent_kwargs
            )
        elif self.agent_type == 'reinforce':
            self.agent = REINFORCEAgent(
                state_size=state_size,
                action_size=num_allocation_strategies,
                portfolio_size=action_size,
                **agent_kwargs
            )
        else:
            raise ValueError(f"Unknown agent type: {agent_type}. Use 'dqn' or 'reinforce'")
        
        # Store action size for portfolio weights
        self.portfolio_action_size = action_size
        
        # Session tracking
        self.current_state = None
        self.session_history = []
        self.episode_count = 0
    
    def reset(self) -> np.ndarray:
        """
        Reset portfolio to initial state.
        
        Returns:
            Initial state
        """
        self.current_state = self.market.reset()
        self.session_history = []
        return self.current_state
    
    def step(self, training: bool = True) -> Dict:
        """
        Perform one investment step.
        
        Args:
            training: Whether agent is in training mode
            
        Returns:
            Dictionary with step information
        """
        # Select action
        if self.agent_type == 'dqn':
            action_idx, portfolio_weights = self.agent.select_action(self.current_state, training=training)
            log_prob = None
        else:  # reinforce
            action_idx, portfolio_weights, log_prob = self.agent.select_action(self.current_state, training=training)
        
        # Execute trade
        transaction_cost, success = self.market.execute_trade(portfolio_weights)
        
        # Advance market
        next_state, daily_return = self.market.step()
        
        # Calculate reward
        # Reward is based on portfolio return, adjusted for risk
        reward = daily_return * 100  # Scale return
        
        # Penalty for transaction costs
        reward -= transaction_cost / self.market.initial_cash * 10
        
        # Bonus for diversification (reduce risk)
        portfolio_weights_actual = self.market.holdings * self.market.current_prices
        total_value = self.market.get_total_value()
        if total_value > 0:
            actual_weights = portfolio_weights_actual / total_value
            # Entropy bonus (encourages diversification)
            entropy = -np.sum(actual_weights * np.log(actual_weights + 1e-8))
            diversification_bonus = entropy * 0.01
            reward += diversification_bonus
        
        # Record step for training
        if training:
            if self.agent_type == 'dqn':
                done = False  # Will be set at end of episode
                self.agent.remember(
                    self.current_state,
                    action_idx,
                    reward,
                    next_state,
                    done
                )
                
                # Train on batch
                loss = self.agent.replay()
                if loss is not None:
                    self.agent.training_history['losses'].append(loss)
            else:  # reinforce
                self.agent.record_step(
                    self.current_state,
                    action_idx,
                    reward,
                    log_prob
                )
        
        # Store step information
        step_info = {
            'state': self.current_state.copy(),
            'action': action_idx,
            'portfolio_weights': portfolio_weights.copy(),
            'daily_return': daily_return,
            'reward': reward,
            'next_state': next_state.copy(),
            'total_value': self.market.get_total_value(),
            'transaction_cost': transaction_cost
        }
        self.session_history.append(step_info)
        
        # Update state
        self.current_state = next_state
        
        return step_info
    
    def run_episode(self, max_steps: int = 252, training: bool = True) -> Dict:
        """
        Run a complete trading episode (one year).
        
        Args:
            max_steps: Maximum steps per episode (default: 252 trading days)
            training: Whether agent is in training mode
            
        Returns:
            Episode summary dictionary
        """
        state = self.reset()
        total_reward = 0
        steps = 0
        
        for _ in range(max_steps):
            step_info = self.step(training=training)
            total_reward += step_info['reward']
            steps += 1
        
        # End episode (triggers learning updates)
        if training:
            if self.agent_type == 'dqn':
                # Mark final step as done
                if len(self.session_history) > 0:
                    last_step = self.session_history[-1]
                    # Update last experience with done=True
                    if len(self.agent.memory) > 0:
                        # Replace last memory entry
                        self.agent.memory[-1] = (
                            last_step['state'],
                            last_step['action'],
                            last_step['reward'],
                            last_step['next_state'],
                            True  # done
                        )
            
            self.agent.end_episode()
        
        # Record episode
        self.agent.record_episode(total_reward, steps)
        self.episode_count += 1
        
        # Get final metrics
        final_metrics = self.market.get_performance_metrics()
        
        episode_summary = {
            'episode': self.episode_count,
            'total_reward': total_reward,
            'steps': steps,
            'final_value': final_metrics['total_value'],
            'total_return': final_metrics['total_return'],
            'sharpe_ratio': final_metrics['sharpe_ratio'],
            'volatility': final_metrics['volatility'],
            'final_holdings': final_metrics['holdings'].tolist(),
            'final_prices': final_metrics['prices'].tolist()
        }
        
        return episode_summary
    
    def train(self, num_episodes: int = 1000, max_steps_per_episode: int = 252,
              verbose: bool = True, print_interval: int = 100) -> List[Dict]:
        """
        Train the portfolio agent.
        
        Args:
            num_episodes: Number of training episodes
            max_steps_per_episode: Maximum steps per episode
            verbose: Whether to print progress
            print_interval: Print statistics every N episodes
            
        Returns:
            List of episode summaries
        """
        episode_summaries = []
        
        for episode in range(num_episodes):
            summary = self.run_episode(max_steps=max_steps_per_episode, training=True)
            episode_summaries.append(summary)
            
            if verbose and (episode + 1) % print_interval == 0:
                recent_summaries = episode_summaries[-print_interval:]
                avg_reward = np.mean([s['total_reward'] for s in recent_summaries])
                avg_return = np.mean([s['total_return'] for s in recent_summaries])
                avg_sharpe = np.mean([s['sharpe_ratio'] for s in recent_summaries])
                
                print(f"Episode {episode + 1}/{num_episodes}")
                print(f"  Avg Reward: {avg_reward:.2f}")
                print(f"  Avg Return: {avg_return*100:.2f}%")
                print(f"  Avg Sharpe Ratio: {avg_sharpe:.3f}")
                
                # Print agent-specific statistics
                stats = self.agent.get_statistics()
                if stats:
                    print(f"  Agent Stats: {stats}")
                print()
        
        return episode_summaries
    
    def evaluate(self, num_episodes: int = 10, max_steps_per_episode: int = 252) -> Dict:
        """
        Evaluate the trained agent.
        
        Args:
            num_episodes: Number of evaluation episodes
            max_steps_per_episode: Maximum steps per episode
            
        Returns:
            Evaluation metrics
        """
        episode_summaries = []
        
        for _ in range(num_episodes):
            summary = self.run_episode(max_steps=max_steps_per_episode, training=False)
            episode_summaries.append(summary)
        
        # Aggregate metrics
        metrics = {
            'mean_reward': np.mean([s['total_reward'] for s in episode_summaries]),
            'std_reward': np.std([s['total_reward'] for s in episode_summaries]),
            'mean_return': np.mean([s['total_return'] for s in episode_summaries]),
            'std_return': np.std([s['total_return'] for s in episode_summaries]),
            'mean_sharpe': np.mean([s['sharpe_ratio'] for s in episode_summaries]),
            'std_sharpe': np.std([s['sharpe_ratio'] for s in episode_summaries]),
            'mean_volatility': np.mean([s['volatility'] for s in episode_summaries]),
            'episode_summaries': episode_summaries
        }
        
        return metrics
    
    def get_portfolio_allocation(self, state: np.ndarray) -> Dict:
        """
        Get recommended portfolio allocation for a state.
        
        Args:
            state: Current state
            
        Returns:
            Dictionary with allocation recommendations
        """
        if self.agent_type == 'dqn':
            action_idx, portfolio_weights = self.agent.select_action(state, training=False)
        else:  # reinforce
            action_idx, portfolio_weights, _ = self.agent.select_action(state, training=False)
        
        return {
            'action_index': action_idx,
            'recommended_weights': portfolio_weights,
            'recommended_allocation': {
                company.symbol: weight 
                for company, weight in zip(self.market.companies, portfolio_weights)
            }
        }
    
    def save(self, filepath: str):
        """Save portfolio agent"""
        self.agent.save(filepath)
    
    def load(self, filepath: str):
        """Load portfolio agent"""
        self.agent.load(filepath)

