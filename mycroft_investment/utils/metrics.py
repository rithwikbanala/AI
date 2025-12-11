"""
Evaluation metrics for portfolio agents.
"""

import numpy as np
from typing import List, Dict
from scipy import stats


def calculate_metrics(episode_summaries: List[Dict]) -> Dict:
    """
    Calculate comprehensive metrics from episode summaries.
    
    Args:
        episode_summaries: List of episode summary dictionaries
        
    Returns:
        Dictionary of calculated metrics
    """
    if len(episode_summaries) == 0:
        return {}
    
    rewards = [s['total_reward'] for s in episode_summaries]
    returns = [s['total_return'] for s in episode_summaries]
    sharpe_ratios = [s['sharpe_ratio'] for s in episode_summaries]
    volatilities = [s['volatility'] for s in episode_summaries]
    values = [s['final_value'] for s in episode_summaries]
    
    # Basic statistics
    metrics = {
        'reward': {
            'mean': np.mean(rewards),
            'std': np.std(rewards),
            'min': np.min(rewards),
            'max': np.max(rewards),
            'median': np.median(rewards)
        },
        'return': {
            'mean': np.mean(returns),
            'std': np.std(returns),
            'min': np.min(returns),
            'max': np.max(returns),
            'median': np.median(returns)
        },
        'sharpe_ratio': {
            'mean': np.mean(sharpe_ratios),
            'std': np.std(sharpe_ratios),
            'min': np.min(sharpe_ratios),
            'max': np.max(sharpe_ratios),
            'median': np.median(sharpe_ratios)
        },
        'volatility': {
            'mean': np.mean(volatilities),
            'std': np.std(volatilities),
            'min': np.min(volatilities),
            'max': np.max(volatilities),
            'median': np.median(volatilities)
        },
        'value': {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'median': np.median(values)
        }
    }
    
    # Learning trends (compare first half vs second half)
    mid_point = len(episode_summaries) // 2
    first_half_returns = returns[:mid_point]
    second_half_returns = returns[mid_point:]
    first_half_sharpe = sharpe_ratios[:mid_point]
    second_half_sharpe = sharpe_ratios[mid_point:]
    
    metrics['learning_improvement'] = {
        'return_improvement': np.mean(second_half_returns) - np.mean(first_half_returns),
        'sharpe_improvement': np.mean(second_half_sharpe) - np.mean(first_half_sharpe),
        'return_improvement_pct': ((np.mean(second_half_returns) - np.mean(first_half_returns)) / 
                                   (np.abs(np.mean(first_half_returns)) + 1e-8)) * 100,
        'sharpe_improvement_pct': ((np.mean(second_half_sharpe) - np.mean(first_half_sharpe)) / 
                                   (np.abs(np.mean(first_half_sharpe)) + 1e-8)) * 100
    }
    
    # Convergence analysis (last 10% of episodes)
    convergence_start = int(len(episode_summaries) * 0.9)
    convergence_returns = returns[convergence_start:]
    convergence_sharpe = sharpe_ratios[convergence_start:]
    
    metrics['convergence'] = {
        'mean_return': np.mean(convergence_returns),
        'std_return': np.std(convergence_returns),
        'mean_sharpe': np.mean(convergence_sharpe),
        'std_sharpe': np.std(convergence_sharpe),
        'return_coefficient_of_variation': np.std(convergence_returns) / (np.abs(np.mean(convergence_returns)) + 1e-8),
        'sharpe_coefficient_of_variation': np.std(convergence_sharpe) / (np.abs(np.mean(convergence_sharpe)) + 1e-8)
    }
    
    # Win rate (positive returns)
    win_rate = np.mean([r > 0 for r in returns])
    metrics['win_rate'] = win_rate
    
    # Maximum drawdown (simplified)
    cumulative_returns = np.cumsum(returns)
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdown = cumulative_returns - running_max
    max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0.0
    metrics['max_drawdown'] = max_drawdown
    
    return metrics

