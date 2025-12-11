"""
Stock Market Environment for Portfolio Investment

Simulates a realistic stock market with multiple AI companies.
Models price movements, volatility, and market dynamics.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import random


class MarketRegime(Enum):
    """Market conditions"""
    BULL = "bull"      # Rising market
    BEAR = "bear"      # Falling market
    VOLATILE = "volatile"  # High volatility
    STABLE = "stable"  # Low volatility


@dataclass
class Stock:
    """Represents a single stock"""
    symbol: str
    name: str
    base_price: float
    volatility: float
    trend: float  # Long-term trend (-1 to 1)
    sector: str


class StockMarket:
    """
    Simulates a stock market with multiple AI companies.
    
    Features:
    - Realistic price movements (geometric Brownian motion)
    - Market regimes (bull, bear, volatile, stable)
    - Company-specific characteristics
    - Correlation between stocks
    """
    
    # AI Company stocks to invest in
    AI_COMPANIES = [
        Stock("NVDA", "NVIDIA", 450.0, 0.025, 0.3, "Hardware"),
        Stock("MSFT", "Microsoft", 380.0, 0.020, 0.2, "Cloud/AI"),
        Stock("GOOGL", "Alphabet", 140.0, 0.022, 0.15, "Search/AI"),
        Stock("META", "Meta", 320.0, 0.028, 0.1, "Social/AI"),
        Stock("TSLA", "Tesla", 250.0, 0.035, 0.05, "EV/AI"),
        Stock("AMD", "AMD", 120.0, 0.030, 0.25, "Hardware"),
        Stock("INTC", "Intel", 45.0, 0.018, -0.1, "Hardware"),
        Stock("AMZN", "Amazon", 150.0, 0.021, 0.12, "Cloud/AI"),
    ]
    
    def __init__(
        self,
        initial_cash: float = 100000.0,
        transaction_cost: float = 0.001,  # 0.1% transaction cost
        companies: Optional[List[Stock]] = None,
        market_regime: MarketRegime = MarketRegime.STABLE,
        random_seed: Optional[int] = None
    ):
        """
        Initialize stock market.
        
        Args:
            initial_cash: Starting cash amount
            transaction_cost: Percentage cost per transaction
            companies: List of stocks (default: AI_COMPANIES)
            market_regime: Current market condition
            random_seed: Random seed for reproducibility
        """
        self.companies = companies or self.AI_COMPANIES.copy()
        self.num_stocks = len(self.companies)
        self.initial_cash = initial_cash
        self.transaction_cost = transaction_cost
        self.market_regime = market_regime
        
        if random_seed is not None:
            np.random.seed(random_seed)
            random.seed(random_seed)
        
        # Initialize prices
        self.current_prices = np.array([stock.base_price for stock in self.companies])
        self.price_history = [self.current_prices.copy()]
        
        # Portfolio state
        self.cash = initial_cash
        self.holdings = np.zeros(self.num_stocks)  # Number of shares per stock
        
        # Market state
        self.day = 0
        self.market_sentiment = 0.0  # -1 to 1
        
        # Correlation matrix (stocks move together)
        self.correlation_matrix = self._generate_correlation_matrix()
        
        # Market regime parameters
        self.regime_duration = 0
        self.max_regime_duration = 50  # Days before regime can change
        
    def _generate_correlation_matrix(self) -> np.ndarray:
        """Generate correlation matrix for stock movements"""
        # Base correlation (all stocks somewhat correlated)
        base_corr = 0.3
        corr_matrix = np.eye(self.num_stocks) * (1 - base_corr) + base_corr
        
        # Sector-based correlations (higher within sectors)
        sector_groups = {}
        for i, stock in enumerate(self.companies):
            if stock.sector not in sector_groups:
                sector_groups[stock.sector] = []
            sector_groups[stock.sector].append(i)
        
        for sector, indices in sector_groups.items():
            for i in indices:
                for j in indices:
                    if i != j:
                        corr_matrix[i, j] = 0.6
        
        return corr_matrix
    
    def reset(self) -> np.ndarray:
        """
        Reset market to initial state.
        
        Returns:
            Initial state vector
        """
        self.current_prices = np.array([stock.base_price for stock in self.companies])
        self.price_history = [self.current_prices.copy()]
        self.cash = self.initial_cash
        self.holdings = np.zeros(self.num_stocks)
        self.day = 0
        self.market_sentiment = 0.0
        self.regime_duration = 0
        self.market_regime = MarketRegime.STABLE
        
        return self.get_state()
    
    def get_state(self) -> np.ndarray:
        """
        Get current market state.
        
        Returns:
            State vector: [cash, holdings, prices (normalized), 
                          portfolio_value, day, market_sentiment]
        """
        portfolio_value = self.get_portfolio_value()
        total_value = self.cash + portfolio_value
        
        # Normalize prices (relative to initial prices)
        normalized_prices = self.current_prices / np.array([s.base_price for s in self.companies])
        
        # Normalize holdings (as fraction of portfolio)
        holdings_normalized = self.holdings * self.current_prices / (total_value + 1e-8)
        
        # Normalize cash
        cash_normalized = self.cash / (self.initial_cash + 1e-8)
        
        # Portfolio value normalized
        portfolio_value_normalized = portfolio_value / (self.initial_cash + 1e-8)
        
        # Day normalized (assuming max 252 trading days)
        day_normalized = self.day / 252.0
        
        state = np.concatenate([
            [cash_normalized],
            holdings_normalized,
            normalized_prices,
            [portfolio_value_normalized],
            [day_normalized],
            [self.market_sentiment]
        ])
        
        return state
    
    def get_state_size(self) -> int:
        """Get size of state vector"""
        return 1 + self.num_stocks + self.num_stocks + 1 + 1 + 1  # cash + holdings + prices + portfolio + day + sentiment
    
    def get_portfolio_value(self) -> float:
        """Calculate current portfolio value"""
        return np.sum(self.holdings * self.current_prices)
    
    def get_total_value(self) -> float:
        """Get total portfolio value (cash + holdings)"""
        return self.cash + self.get_portfolio_value()
    
    def step(self) -> Tuple[np.ndarray, float]:
        """
        Advance market by one day.
        
        Returns:
            (new_state, daily_return): New state and portfolio return
        """
        old_value = self.get_total_value()
        
        # Update market regime
        self._update_market_regime()
        
        # Generate price movements
        self._update_prices()
        
        # Update day
        self.day += 1
        
        # Calculate return
        new_value = self.get_total_value()
        daily_return = (new_value - old_value) / (old_value + 1e-8)
        
        return self.get_state(), daily_return
    
    def _update_market_regime(self):
        """Update market regime based on duration and randomness"""
        self.regime_duration += 1
        
        if self.regime_duration >= self.max_regime_duration:
            # Randomly change regime
            if np.random.random() < 0.3:  # 30% chance to change
                self.market_regime = random.choice(list(MarketRegime))
                self.regime_duration = 0
    
    def _update_prices(self):
        """Update stock prices using geometric Brownian motion"""
        # Market-wide sentiment based on regime
        regime_multipliers = {
            MarketRegime.BULL: 1.2,
            MarketRegime.BEAR: 0.8,
            MarketRegime.VOLATILE: 1.5,
            MarketRegime.STABLE: 0.8
        }
        
        volatility_mult = regime_multipliers.get(self.market_regime, 1.0)
        
        # Generate correlated random shocks
        random_shocks = np.random.multivariate_normal(
            mean=np.zeros(self.num_stocks),
            cov=self.correlation_matrix
        )
        
        # Update each stock price
        for i, stock in enumerate(self.companies):
            # Base drift (trend)
            drift = stock.trend * 0.0005
            
            # Volatility (company-specific + market regime)
            volatility = stock.volatility * volatility_mult
            
            # Random shock (correlated with other stocks)
            shock = random_shocks[i] * volatility
            
            # Market sentiment effect
            sentiment_effect = self.market_sentiment * 0.001
            
            # Price change
            price_change = drift + shock + sentiment_effect
            
            # Update price (geometric Brownian motion)
            self.current_prices[i] *= (1 + price_change)
            
            # Ensure price doesn't go negative
            self.current_prices[i] = max(0.01, self.current_prices[i])
        
        # Update market sentiment (mean-reverting)
        target_sentiment = {
            MarketRegime.BULL: 0.5,
            MarketRegime.BEAR: -0.5,
            MarketRegime.VOLATILE: 0.0,
            MarketRegime.STABLE: 0.0
        }.get(self.market_regime, 0.0)
        
        self.market_sentiment = 0.9 * self.market_sentiment + 0.1 * target_sentiment + np.random.normal(0, 0.1)
        self.market_sentiment = np.clip(self.market_sentiment, -1, 1)
        
        # Store price history
        self.price_history.append(self.current_prices.copy())
    
    def execute_trade(self, action: np.ndarray) -> Tuple[float, bool]:
        """
        Execute a trading action.
        
        Args:
            action: Array of target portfolio weights [w1, w2, ..., wn]
                   where sum(weights) should be <= 1.0
        
        Returns:
            (transaction_cost, success): Cost incurred and whether trade succeeded
        """
        # Normalize weights to sum to 1.0
        target_weights = np.clip(action, 0, 1)
        total_weight = np.sum(target_weights)
        
        if total_weight > 1.0:
            target_weights = target_weights / total_weight
        
        # Calculate target portfolio value
        total_value = self.get_total_value()
        target_values = target_weights * total_value
        
        # Calculate target holdings
        target_holdings = target_values / (self.current_prices + 1e-8)
        
        # Calculate trades needed
        trades = target_holdings - self.holdings
        
        # Calculate transaction costs
        transaction_value = np.abs(trades) * self.current_prices
        total_transaction_value = np.sum(transaction_value)
        transaction_cost_amount = total_transaction_value * self.transaction_cost
        
        # Check if we have enough cash
        # Need cash for: buying new shares + transaction costs
        buy_value = np.sum(np.maximum(0, trades) * self.current_prices)
        sell_value = np.sum(np.maximum(0, -trades) * self.current_prices)
        net_cash_needed = buy_value - sell_value + transaction_cost_amount
        
        if net_cash_needed > self.cash:
            # Insufficient cash - scale down the trade
            scale_factor = self.cash / (net_cash_needed + 1e-8)
            trades = trades * scale_factor
            transaction_value = np.abs(trades) * self.current_prices
            total_transaction_value = np.sum(transaction_value)
            transaction_cost_amount = total_transaction_value * self.transaction_cost
        
        # Execute trades
        # Sell first (to get cash)
        sell_value = np.sum(np.maximum(0, -trades) * self.current_prices)
        self.cash += sell_value
        
        # Buy new shares
        buy_value = np.sum(np.maximum(0, trades) * self.current_prices)
        self.cash -= buy_value
        
        # Pay transaction costs
        self.cash -= transaction_cost_amount
        
        # Update holdings
        self.holdings += trades
        
        # Ensure holdings are non-negative
        self.holdings = np.maximum(0, self.holdings)
        
        return transaction_cost_amount, True
    
    def get_action_size(self) -> int:
        """Get number of actions (one weight per stock)"""
        return self.num_stocks
    
    def get_performance_metrics(self) -> Dict:
        """Get portfolio performance metrics"""
        total_value = self.get_total_value()
        total_return = (total_value - self.initial_cash) / self.initial_cash
        
        # Calculate returns over time
        if len(self.price_history) > 1:
            portfolio_values = []
            for prices in self.price_history:
                portfolio_value = np.sum(self.holdings * prices) + self.cash
                portfolio_values.append(portfolio_value)
            
            returns = np.diff(portfolio_values) / (np.array(portfolio_values[:-1]) + 1e-8)
            volatility = np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0.0
            sharpe_ratio = (np.mean(returns) * 252) / (volatility + 1e-8) if volatility > 0 else 0.0
        else:
            volatility = 0.0
            sharpe_ratio = 0.0
        
        return {
            'total_value': total_value,
            'total_return': total_return,
            'cash': self.cash,
            'portfolio_value': self.get_portfolio_value(),
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'day': self.day,
            'holdings': self.holdings.copy(),
            'prices': self.current_prices.copy()
        }

