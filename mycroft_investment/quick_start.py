"""
Quick Start Script for Portfolio Agent

Demonstrates basic usage of the portfolio investment system.
"""

import sys
import os
import numpy as np

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.portfolio_agent import PortfolioAgent
from utils.visualization import plot_learning_curves, plot_portfolio_allocation
from utils.metrics import calculate_metrics
from market.stock_market import StockMarket


def quick_demo():
    """Run a quick demonstration"""
    print("=" * 60)
    print("Portfolio Agent Quick Start Demo")
    print("=" * 60)
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Create market and agent
    print("\n1. Creating market with AI companies...")
    market = StockMarket(initial_cash=100000.0)
    print(f"   Companies: {[c.symbol for c in market.companies]}")
    print(f"   Initial Cash: ${market.initial_cash:,.2f}")
    
    # Create agent
    print("\n2. Training DQN Portfolio Agent (100 episodes)...")
    agent = PortfolioAgent(market=market, learning_rate=0.001, discount_factor=0.99)
    
    # Train
    episode_summaries = agent.train(
        num_episodes=100,
        max_steps_per_episode=252,  # One year
        verbose=True,
        print_interval=25
    )
    
    # Calculate metrics
    metrics = calculate_metrics(episode_summaries)
    print(f"\nDQN Portfolio Agent Results:")
    print(f"  Final Return: {metrics['return']['mean']*100:.2f}%")
    print(f"  Sharpe Ratio: {metrics['sharpe_ratio']['mean']:.3f}")
    print(f"  Win Rate: {metrics['win_rate']*100:.1f}%")
    
    # Visualize
    plot_learning_curves(
        episode_summaries,
        agent_name="DQN Portfolio Agent (Quick Demo)",
        save_path='results/dqn_quick_demo.png'
    )
    
    # Show final allocation
    print("\n3. Final Portfolio Allocation:")
    final_summary = episode_summaries[-1]
    holdings = np.array(final_summary['final_holdings'])
    prices = np.array(final_summary['final_prices'])
    portfolio_value = np.sum(holdings * prices)
    
    if portfolio_value > 0:
        weights = (holdings * prices) / portfolio_value
        for company, weight in zip(market.companies, weights):
            if weight > 0.01:  # Show allocations > 1%
                print(f"   {company.symbol}: {weight*100:.1f}%")
    
    # Evaluate
    print("\n4. Evaluating trained agent...")
    eval_metrics = agent.evaluate(num_episodes=5)
    print(f"  Evaluation Mean Return: {eval_metrics['mean_return']*100:.2f}% ± {eval_metrics['std_return']*100:.2f}%")
    print(f"  Evaluation Mean Sharpe: {eval_metrics['mean_sharpe']:.3f} ± {eval_metrics['std_sharpe']:.3f}")
    
    print("\n" + "=" * 60)
    print("Quick Demo Complete!")
    print("Check 'results/' directory for visualizations.")
    print("=" * 60)


if __name__ == '__main__':
    quick_demo()

