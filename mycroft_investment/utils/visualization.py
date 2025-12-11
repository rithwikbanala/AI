"""
Visualization utilities for portfolio agent training and evaluation.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Dict, Optional
import os


def plot_learning_curves(
    episode_summaries: List[Dict],
    agent_name: str = "Portfolio Agent",
    save_path: Optional[str] = None,
    window_size: int = 50
):
    """
    Plot learning curves for training.
    
    Args:
        episode_summaries: List of episode summary dictionaries
        agent_name: Name of agent for plot title
        save_path: Path to save figure (None to display)
        window_size: Window size for moving average
    """
    episodes = [s['episode'] for s in episode_summaries]
    rewards = [s['total_reward'] for s in episode_summaries]
    returns = [s['total_return'] * 100 for s in episode_summaries]  # Convert to percentage
    sharpe_ratios = [s['sharpe_ratio'] for s in episode_summaries]
    values = [s['final_value'] for s in episode_summaries]
    
    # Calculate moving averages
    def moving_average(data, window):
        return np.convolve(data, np.ones(window)/window, mode='valid')
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'{agent_name} Learning Curves', fontsize=16, fontweight='bold')
    
    # Plot 1: Total Reward
    ax1 = axes[0, 0]
    ax1.plot(episodes, rewards, alpha=0.3, color='blue', label='Raw')
    if len(rewards) >= window_size:
        ma_rewards = moving_average(rewards, window_size)
        ma_episodes = episodes[window_size-1:]
        ax1.plot(ma_episodes, ma_rewards, color='blue', linewidth=2, label=f'MA({window_size})')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.set_title('Episode Rewards')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Portfolio Returns
    ax2 = axes[0, 1]
    ax2.plot(episodes, returns, alpha=0.3, color='green', label='Raw')
    if len(returns) >= window_size:
        ma_returns = moving_average(returns, window_size)
        ax2.plot(ma_episodes, ma_returns, color='green', linewidth=2, label=f'MA({window_size})')
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Total Return (%)')
    ax2.set_title('Portfolio Returns')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Sharpe Ratio
    ax3 = axes[1, 0]
    ax3.plot(episodes, sharpe_ratios, alpha=0.3, color='orange', label='Raw')
    if len(sharpe_ratios) >= window_size:
        ma_sharpe = moving_average(sharpe_ratios, window_size)
        ax3.plot(ma_episodes, ma_sharpe, color='orange', linewidth=2, label=f'MA({window_size})')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Sharpe Ratio')
    ax3.set_title('Risk-Adjusted Returns')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Portfolio Value
    ax4 = axes[1, 1]
    ax4.plot(episodes, values, alpha=0.3, color='purple', label='Raw')
    if len(values) >= window_size:
        ma_values = moving_average(values, window_size)
        ax4.plot(ma_episodes, ma_values, color='purple', linewidth=2, label=f'MA({window_size})')
    initial_value = values[0] if values else 100000
    ax4.axhline(y=initial_value, color='red', linestyle='--', alpha=0.5, label='Initial Value')
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Portfolio Value ($)')
    ax4.set_title('Portfolio Value Over Time')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_portfolio_allocation(
    episode_summaries: List[Dict],
    market_companies: List,
    agent_name: str = "Portfolio Agent",
    save_path: Optional[str] = None,
    num_episodes: int = 5
):
    """
    Plot portfolio allocation over time.
    
    Args:
        episode_summaries: List of episode summaries
        market_companies: List of company stocks
        agent_name: Name of agent
        save_path: Path to save figure
        num_episodes: Number of episodes to show
    """
    # Get recent episodes
    recent_episodes = episode_summaries[-num_episodes:]
    
    fig, axes = plt.subplots(num_episodes, 1, figsize=(12, 3*num_episodes))
    if num_episodes == 1:
        axes = [axes]
    
    fig.suptitle(f'{agent_name} Portfolio Allocation Over Time', fontsize=16, fontweight='bold')
    
    company_symbols = [company.symbol for company in market_companies]
    
    for idx, summary in enumerate(recent_episodes):
        ax = axes[idx]
        holdings = np.array(summary['final_holdings'])
        prices = np.array(summary['final_prices'])
        
        # Calculate weights
        portfolio_value = np.sum(holdings * prices)
        if portfolio_value > 0:
            weights = (holdings * prices) / portfolio_value
        else:
            weights = np.zeros(len(holdings))
        
        # Create bar chart
        bars = ax.bar(company_symbols, weights * 100, alpha=0.7)
        ax.set_ylabel('Allocation (%)')
        ax.set_title(f'Episode {summary["episode"]} - Return: {summary["total_return"]*100:.2f}%')
        ax.set_ylim([0, 100])
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, weight in zip(bars, weights):
            if weight > 0.05:  # Only show labels for >5% allocation
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{weight*100:.1f}%',
                       ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_episode_analysis(
    episode_summaries: List[Dict],
    agent_name: str = "Portfolio Agent",
    save_path: Optional[str] = None
):
    """
    Plot detailed episode analysis.
    
    Args:
        episode_summaries: List of episode summaries
        agent_name: Name of agent
        save_path: Path to save figure
    """
    returns = [s['total_return'] * 100 for s in episode_summaries]
    sharpe_ratios = [s['sharpe_ratio'] for s in episode_summaries]
    volatilities = [s['volatility'] for s in episode_summaries]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'{agent_name} Detailed Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Return distribution
    ax1 = axes[0, 0]
    ax1.hist(returns, bins=30, edgecolor='black', alpha=0.7, color='green')
    ax1.axvline(np.mean(returns), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {np.mean(returns):.2f}%')
    ax1.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax1.set_xlabel('Total Return (%)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Return Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Sharpe ratio distribution
    ax2 = axes[0, 1]
    ax2.hist(sharpe_ratios, bins=30, edgecolor='black', alpha=0.7, color='orange')
    ax2.axvline(np.mean(sharpe_ratios), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(sharpe_ratios):.3f}')
    ax2.set_xlabel('Sharpe Ratio')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Risk-Adjusted Return Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Return vs Volatility
    ax3 = axes[1, 0]
    scatter = ax3.scatter(volatilities, returns, c=sharpe_ratios, 
                         cmap='viridis', alpha=0.6, s=50)
    ax3.set_xlabel('Volatility')
    ax3.set_ylabel('Return (%)')
    ax3.set_title('Return vs Risk (Color = Sharpe Ratio)')
    plt.colorbar(scatter, ax=ax3, label='Sharpe Ratio')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Cumulative returns
    ax4 = axes[1, 1]
    cumulative_returns = np.cumsum(returns)
    ax4.plot(cumulative_returns, linewidth=2, color='blue')
    ax4.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Cumulative Return (%)')
    ax4.set_title('Cumulative Returns Over Time')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()
    
    plt.close()



def plot_agent_comparison(
    dqn_summaries: List[Dict],
    reinforce_summaries: List[Dict],
    save_path: Optional[str] = None,
    window_size: int = 50
):
    """
    Compare DQN and REINFORCE agents.
    
    Args:
        dqn_summaries: DQN episode summaries
        reinforce_summaries: REINFORCE episode summaries
        save_path: Path to save figure
        window_size: Window size for moving average
    """
    def moving_average(data, window):
        return np.convolve(data, np.ones(window)/window, mode='valid')
    
    dqn_episodes = [s['episode'] for s in dqn_summaries]
    dqn_returns = [s['total_return'] * 100 for s in dqn_summaries]
    dqn_sharpe = [s['sharpe_ratio'] for s in dqn_summaries]
    
    rf_episodes = [s['episode'] for s in reinforce_summaries]
    rf_returns = [s['total_return'] * 100 for s in reinforce_summaries]
    rf_sharpe = [s['sharpe_ratio'] for s in reinforce_summaries]
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle('DQN vs REINFORCE Portfolio Agent Comparison', fontsize=16, fontweight='bold')
    
    # Plot 1: Returns
    ax1 = axes[0]
    if len(dqn_returns) >= window_size:
        dqn_ma = moving_average(dqn_returns, window_size)
        dqn_ma_ep = dqn_episodes[window_size-1:]
        ax1.plot(dqn_ma_ep, dqn_ma, label='DQN', linewidth=2, color='blue')
    else:
        ax1.plot(dqn_episodes, dqn_returns, label='DQN', linewidth=2, color='blue')
    
    if len(rf_returns) >= window_size:
        rf_ma = moving_average(rf_returns, window_size)
        rf_ma_ep = rf_episodes[window_size-1:]
        ax1.plot(rf_ma_ep, rf_ma, label='REINFORCE', linewidth=2, color='red')
    else:
        ax1.plot(rf_episodes, rf_returns, label='REINFORCE', linewidth=2, color='red')
    
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Return (%)')
    ax1.set_title('Portfolio Returns')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Sharpe Ratio
    ax2 = axes[1]
    if len(dqn_sharpe) >= window_size:
        dqn_ma = moving_average(dqn_sharpe, window_size)
        ax2.plot(dqn_ma_ep, dqn_ma, label='DQN', linewidth=2, color='blue')
    else:
        ax2.plot(dqn_episodes, dqn_sharpe, label='DQN', linewidth=2, color='blue')
    
    if len(rf_sharpe) >= window_size:
        rf_ma = moving_average(rf_sharpe, window_size)
        ax2.plot(rf_ma_ep, rf_ma, label='REINFORCE', linewidth=2, color='red')
    else:
        ax2.plot(rf_episodes, rf_sharpe, label='REINFORCE', linewidth=2, color='red')
    
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Sharpe Ratio')
    ax2.set_title('Risk-Adjusted Returns')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()
    
    plt.close()
