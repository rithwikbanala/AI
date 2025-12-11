"""
Main execution script for Portfolio Agent Training

Trains DQN agent to optimize portfolio allocation across AI companies.
"""

import argparse
import numpy as np
import sys
import os

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from agents.portfolio_agent import PortfolioAgent
from utils.visualization import plot_learning_curves, plot_portfolio_allocation, plot_episode_analysis, plot_agent_comparison
from utils.metrics import calculate_metrics
from market.stock_market import StockMarket


def train_agent(agent_type: str = 'dqn', num_episodes: int = 1000, **kwargs):
    """
    Train portfolio agent.
    
    Args:
        agent_type: 'dqn' or 'reinforce'
        num_episodes: Number of training episodes
        **kwargs: Additional agent parameters
    """
    agent_name = 'DQN' if agent_type == 'dqn' else 'REINFORCE'
    print(f"Training Portfolio Agent with {agent_name}...")
    print("=" * 60)
    
    # Create market and agent
    market = StockMarket(initial_cash=kwargs.get('initial_cash', 100000.0))
    agent = PortfolioAgent(market=market, agent_type=agent_type, **kwargs)
    
    # Train agent
    episode_summaries = agent.train(
        num_episodes=num_episodes,
        max_steps_per_episode=252,  # One year of trading
        verbose=True,
        print_interval=100
    )
    
    # Calculate metrics
    metrics = calculate_metrics(episode_summaries)
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print("\nFinal Metrics:")
    print(f"  Mean Return: {metrics['return']['mean']*100:.2f}% ± {metrics['return']['std']*100:.2f}%")
    print(f"  Mean Sharpe Ratio: {metrics['sharpe_ratio']['mean']:.3f} ± {metrics['sharpe_ratio']['std']:.3f}")
    print(f"  Mean Volatility: {metrics['volatility']['mean']:.3f} ± {metrics['volatility']['std']:.3f}")
    print(f"  Win Rate: {metrics['win_rate']*100:.1f}%")
    print(f"  Max Drawdown: {metrics['max_drawdown']*100:.2f}%")
    print(f"\nLearning Improvement:")
    print(f"  Return: {metrics['learning_improvement']['return_improvement']*100:.2f}% "
          f"({metrics['learning_improvement']['return_improvement_pct']:.1f}%)")
    print(f"  Sharpe: {metrics['learning_improvement']['sharpe_improvement']:.3f} "
          f"({metrics['learning_improvement']['sharpe_improvement_pct']:.1f}%)")
    
    # Create visualizations
    os.makedirs('results', exist_ok=True)
    agent_name_display = f"{agent_name.upper()} Portfolio Agent"
    file_prefix = agent_type.lower()
    
    plot_learning_curves(
        episode_summaries,
        agent_name=agent_name_display,
        save_path=f'results/{file_prefix}_learning_curves.png'
    )
    plot_episode_analysis(
        episode_summaries,
        agent_name=agent_name_display,
        save_path=f'results/{file_prefix}_analysis.png'
    )
    plot_portfolio_allocation(
        episode_summaries,
        market.companies,
        agent_name=agent_name_display,
        save_path=f'results/{file_prefix}_allocation.png',
        num_episodes=5
    )
    
    # Evaluate trained agent
    print("\nEvaluating trained agent...")
    eval_metrics = agent.evaluate(num_episodes=10)
    print(f"  Evaluation Mean Return: {eval_metrics['mean_return']*100:.2f}% ± {eval_metrics['std_return']*100:.2f}%")
    print(f"  Evaluation Mean Sharpe: {eval_metrics['mean_sharpe']:.3f} ± {eval_metrics['std_sharpe']:.3f}")
    
    # Save agent
    agent_file = f'results/{agent_type.lower()}_agent.pkl'
    agent.save(agent_file)
    print(f"\nAgent saved to {agent_file}")
    
    return agent, episode_summaries, metrics


def compare_agents(num_episodes: int = 1000):
    """Train and compare both DQN and REINFORCE agents"""
    print("Training and Comparing DQN and REINFORCE Agents")
    print("=" * 60)
    
    # Train DQN
    print("\n1. Training DQN Agent...")
    dqn_agent, dqn_summaries, dqn_metrics = train_agent(
        agent_type='dqn',
        num_episodes=num_episodes,
        learning_rate=0.001,
        discount_factor=0.99
    )
    
    # Train REINFORCE
    print("\n2. Training REINFORCE Agent...")
    rf_agent, rf_summaries, rf_metrics = train_agent(
        agent_type='reinforce',
        num_episodes=num_episodes,
        learning_rate=0.001,
        discount_factor=0.99
    )
    
    # Compare
    print("\n" + "=" * 60)
    print("Agent Comparison")
    print("=" * 60)
    
    print("\nReturn Comparison:")
    print(f"  DQN:       {dqn_metrics['return']['mean']*100:.2f}% ± {dqn_metrics['return']['std']*100:.2f}%")
    print(f"  REINFORCE: {rf_metrics['return']['mean']*100:.2f}% ± {rf_metrics['return']['std']*100:.2f}%")
    
    print("\nSharpe Ratio Comparison:")
    print(f"  DQN:       {dqn_metrics['sharpe_ratio']['mean']:.3f} ± {dqn_metrics['sharpe_ratio']['std']:.3f}")
    print(f"  REINFORCE: {rf_metrics['sharpe_ratio']['mean']:.3f} ± {rf_metrics['sharpe_ratio']['std']:.3f}")
    
    print("\nWin Rate Comparison:")
    print(f"  DQN:       {dqn_metrics['win_rate']*100:.1f}%")
    print(f"  REINFORCE: {rf_metrics['win_rate']*100:.1f}%")
    
    # Create comparison visualization
    plot_agent_comparison(
        dqn_summaries,
        rf_summaries,
        save_path='results/agent_comparison.png'
    )
    print("\nComparison visualization saved to results/agent_comparison.png")


def main():
    parser = argparse.ArgumentParser(description='Train Portfolio Agent with DQN or REINFORCE')
    parser.add_argument('--agent', type=str, choices=['dqn', 'reinforce', 'compare'],
                        default='dqn', help='Agent type to train')
    parser.add_argument('--episodes', type=int, default=1000,
                        help='Number of training episodes')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--discount', type=float, default=0.99,
                        help='Discount factor')
    parser.add_argument('--initial-cash', type=float, default=100000.0,
                        help='Initial cash amount')
    parser.add_argument('--epsilon', type=float, default=1.0,
                        help='Initial epsilon (exploration, DQN only)')
    parser.add_argument('--epsilon-decay', type=float, default=0.995,
                        help='Epsilon decay rate (DQN only)')
    
    args = parser.parse_args()
    
    if args.agent == 'compare':
        compare_agents(num_episodes=args.episodes)
    else:
        agent_kwargs = {
            'learning_rate': args.learning_rate,
            'discount_factor': args.discount,
            'initial_cash': args.initial_cash
        }
        
        # Add DQN-specific parameters
        if args.agent == 'dqn':
            agent_kwargs['epsilon'] = args.epsilon
            agent_kwargs['epsilon_decay'] = args.epsilon_decay
        
        train_agent(agent_type=args.agent, num_episodes=args.episodes, **agent_kwargs)


if __name__ == '__main__':
    main()

