# Mycroft Investment Strategy Learning

## Overview

This project implements a **Deep Q-Network (DQN)** reinforcement learning system for portfolio optimization across AI companies. The system learns optimal investment strategies by adapting portfolio allocation based on market feedback.

## Project Structure

```
mycroft_investment/
├── requirements.txt
├── README.md
├── main.py                 # Main execution script
├── quick_start.py          # Quick demo script
├── market/
│   ├── __init__.py
│   └── stock_market.py     # Stock market simulation
├── agents/
│   ├── __init__.py
│   ├── dqn_agent.py        # DQN implementation
│   └── portfolio_agent.py  # Main portfolio orchestrator
├── utils/
│   ├── __init__.py
│   ├── visualization.py    # Learning curves and plots
│   └── metrics.py          # Evaluation metrics
└── experiments/
    └── __init__.py
```

## Features

1. **Realistic Stock Market Simulation**
   - 8 AI company stocks (NVDA, MSFT, GOOGL, META, TSLA, AMD, INTC, AMZN)
   - Geometric Brownian motion price modeling
   - Market regimes (bull, bear, volatile, stable)
   - Stock correlations and sector effects

2. **DQN Agent**
   - Deep Q-Network with experience replay
   - Target network for stable learning
   - Epsilon-greedy exploration
   - Multiple portfolio allocation strategies

3. **Portfolio Optimization**
   - Dynamic portfolio rebalancing
   - Transaction cost modeling
   - Risk-adjusted returns (Sharpe ratio)
   - Diversification incentives

4. **Comprehensive Evaluation**
   - Return metrics
   - Risk metrics (volatility, Sharpe ratio)
   - Portfolio allocation visualization
   - Learning curves

## Installation

```bash
cd mycroft_investment
pip install -r requirements.txt
```

## Usage

### Quick Start

```bash
python quick_start.py
```

### Full Training

```bash
python main.py --episodes 1000
```

### Custom Parameters

```bash
python main.py --episodes 1000 --learning-rate 0.001 --discount 0.99 --initial-cash 100000
```

## Results

The system demonstrates:
- Learning optimal portfolio allocation strategies
- Adaptation to different market conditions
- Risk-adjusted return optimization
- Convergence to stable investment policies

## Key Metrics

- **Total Return**: Portfolio performance percentage
- **Sharpe Ratio**: Risk-adjusted returns
- **Volatility**: Portfolio risk measure
- **Win Rate**: Percentage of profitable episodes
- **Max Drawdown**: Maximum portfolio decline

## AI Companies Tracked

1. **NVDA** - NVIDIA (Hardware)
2. **MSFT** - Microsoft (Cloud/AI)
3. **GOOGL** - Alphabet (Search/AI)
4. **META** - Meta (Social/AI)
5. **TSLA** - Tesla (EV/AI)
6. **AMD** - AMD (Hardware)
7. **INTC** - Intel (Hardware)
8. **AMZN** - Amazon (Cloud/AI)

## Technical Details

- **State Space**: Market prices, portfolio holdings, cash, market sentiment
- **Action Space**: Discrete portfolio allocation strategies
- **Reward Function**: Portfolio returns adjusted for transaction costs and diversification
- **Algorithm**: DQN with experience replay and target network

