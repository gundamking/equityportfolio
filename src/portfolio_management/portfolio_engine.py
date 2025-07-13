"""
Portfolio Engine Module

Core portfolio management engine that handles position management,
order execution, and portfolio state tracking.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass, field
from .strategies import BaseStrategy

@dataclass
class Position:
    """Represents a position in a security."""
    symbol: str
    shares: float
    avg_cost: float
    current_price: float = 0.0
    market_value: float = 0.0
    unrealized_pnl: float = 0.0
    
    def update_price(self, price: float):
        """Update current price and calculate market value."""
        self.current_price = price
        self.market_value = self.shares * price
        self.unrealized_pnl = self.market_value - (self.shares * self.avg_cost)

@dataclass
class Transaction:
    """Represents a transaction."""
    date: datetime
    symbol: str
    action: str  # 'BUY' or 'SELL'
    shares: float
    price: float
    value: float
    commission: float = 0.0
    
@dataclass
class PortfolioState:
    """Represents the current state of the portfolio."""
    cash: float
    positions: Dict[str, Position] = field(default_factory=dict)
    total_value: float = 0.0
    total_equity: float = 0.0
    
    def calculate_totals(self):
        """Calculate total portfolio value and equity."""
        position_value = sum(pos.market_value for pos in self.positions.values())
        self.total_value = self.cash + position_value
        self.total_equity = self.total_value

class PortfolioEngine:
    """
    Main portfolio management engine.
    
    Handles position management, order execution, rebalancing,
    and portfolio state tracking.
    """
    
    def __init__(self, initial_capital: float = 5000000, 
                 commission_rate: float = 0.001,
                 rebalance_frequency: int = 5):
        """
        Initialize the portfolio engine.
        
        Args:
            initial_capital: Starting capital amount
            commission_rate: Commission rate per transaction
            rebalance_frequency: Days between rebalancing
        """
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.rebalance_frequency = rebalance_frequency
        
        # Initialize portfolio state
        self.portfolio_state = PortfolioState(cash=initial_capital)
        
        # Transaction and performance tracking
        self.transactions: List[Transaction] = []
        self.portfolio_history: List[Dict] = []
        
        # Logger setup
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        """Set up logging configuration."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def execute_trade(self, symbol: str, action: str, shares: float, 
                     price: float, date: datetime) -> bool:
        """
        Execute a trade (buy or sell).
        
        Args:
            symbol: Stock symbol
            action: 'BUY' or 'SELL'
            shares: Number of shares
            price: Price per share
            date: Transaction date
            
        Returns:
            True if trade executed successfully, False otherwise
        """
        trade_value = shares * price
        commission = trade_value * self.commission_rate
        total_cost = trade_value + commission
        
        if action == 'BUY':
            # Check if we have enough cash
            if self.portfolio_state.cash < total_cost:
                self.logger.warning(f"Insufficient cash for {symbol} purchase: ${total_cost:.2f}")
                return False
            
            # Execute buy order
            self.portfolio_state.cash -= total_cost
            
            if symbol in self.portfolio_state.positions:
                # Update existing position
                pos = self.portfolio_state.positions[symbol]
                total_shares = pos.shares + shares
                total_cost_basis = (pos.shares * pos.avg_cost) + trade_value
                pos.shares = total_shares
                pos.avg_cost = total_cost_basis / total_shares
                pos.update_price(price)
            else:
                # Create new position
                self.portfolio_state.positions[symbol] = Position(
                    symbol=symbol,
                    shares=shares,
                    avg_cost=price,
                    current_price=price
                )
                self.portfolio_state.positions[symbol].update_price(price)
            
        elif action == 'SELL':
            # Check if we have the position
            if symbol not in self.portfolio_state.positions:
                self.logger.warning(f"Cannot sell {symbol}: No position found")
                return False
            
            pos = self.portfolio_state.positions[symbol]
            if pos.shares < shares:
                self.logger.warning(f"Cannot sell {shares} shares of {symbol}: Only {pos.shares} available")
                return False
            
            # Execute sell order
            self.portfolio_state.cash += (trade_value - commission)
            pos.shares -= shares
            pos.update_price(price)
            
            # Remove position if completely sold
            if pos.shares == 0:
                del self.portfolio_state.positions[symbol]
        
        # Record transaction
        transaction = Transaction(
            date=date,
            symbol=symbol,
            action=action,
            shares=shares,
            price=price,
            value=trade_value,
            commission=commission
        )
        self.transactions.append(transaction)
        
        self.logger.info(f"Executed {action} {shares} shares of {symbol} at ${price:.2f}")
        return True
    
    def update_portfolio_prices(self, price_data: Dict[str, float], date: datetime):
        """
        Update current prices for all positions.
        
        Args:
            price_data: Dictionary mapping symbols to current prices
            date: Current date
        """
        for symbol, position in self.portfolio_state.positions.items():
            if symbol in price_data:
                position.update_price(price_data[symbol])
        
        # Calculate portfolio totals
        self.portfolio_state.calculate_totals()
        
        # Record portfolio state
        self._record_portfolio_state(date)
    
    def _record_portfolio_state(self, date: datetime):
        """Record current portfolio state for historical tracking."""
        state_record = {
            'date': date,
            'cash': self.portfolio_state.cash,
            'total_value': self.portfolio_state.total_value,
            'total_equity': self.portfolio_state.total_equity,
            'positions': {symbol: {
                'shares': pos.shares,
                'price': pos.current_price,
                'value': pos.market_value,
                'unrealized_pnl': pos.unrealized_pnl
            } for symbol, pos in self.portfolio_state.positions.items()}
        }
        self.portfolio_history.append(state_record)
    
    def rebalance_portfolio(self, strategy: BaseStrategy, data: Dict[str, pd.DataFrame], 
                          date: str, target_positions: int = 5) -> bool:
        """
        Rebalance portfolio based on strategy signals.
        
        Args:
            strategy: Trading strategy to use
            data: Market data
            date: Rebalancing date
            target_positions: Number of positions to hold
            
        Returns:
            True if rebalancing successful
        """
        try:
            # Get selected assets from strategy
            selected_assets = strategy.select_assets(data, date, target_positions)
            
            if not selected_assets:
                self.logger.warning(f"No assets selected for {date}")
                return False
            
            # Calculate target allocation per asset
            target_allocation = self.portfolio_state.total_equity / len(selected_assets)
            
            # Get current prices
            current_prices = {}
            for symbol in selected_assets:
                if symbol in data:
                    closest_date = min(data[symbol].index, 
                                     key=lambda x: abs(x - pd.to_datetime(date)))
                    current_prices[symbol] = data[symbol].loc[closest_date, 'adj_close']
            
            # Update current portfolio prices
            self.update_portfolio_prices(current_prices, pd.to_datetime(date))
            
            # Sell positions not in selected assets
            positions_to_sell = []
            for symbol in list(self.portfolio_state.positions.keys()):
                if symbol not in selected_assets:
                    positions_to_sell.append(symbol)
            
            for symbol in positions_to_sell:
                pos = self.portfolio_state.positions[symbol]
                if symbol in current_prices:
                    self.execute_trade(symbol, 'SELL', pos.shares, 
                                     current_prices[symbol], pd.to_datetime(date))
            
            # Buy or adjust positions for selected assets
            for symbol in selected_assets:
                if symbol not in current_prices:
                    continue
                
                current_price = current_prices[symbol]
                target_shares = target_allocation / current_price
                
                if symbol in self.portfolio_state.positions:
                    # Adjust existing position
                    current_shares = self.portfolio_state.positions[symbol].shares
                    share_diff = target_shares - current_shares
                    
                    if share_diff > 0:
                        # Buy more shares
                        self.execute_trade(symbol, 'BUY', share_diff, 
                                         current_price, pd.to_datetime(date))
                    elif share_diff < 0:
                        # Sell excess shares
                        self.execute_trade(symbol, 'SELL', abs(share_diff), 
                                         current_price, pd.to_datetime(date))
                else:
                    # Create new position
                    self.execute_trade(symbol, 'BUY', target_shares, 
                                     current_price, pd.to_datetime(date))
            
            self.logger.info(f"Portfolio rebalanced on {date}: {selected_assets}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error rebalancing portfolio on {date}: {str(e)}")
            return False
    
    def run_backtest(self, strategy: BaseStrategy, data: Dict[str, pd.DataFrame], 
                    start_date: str, end_date: str) -> pd.DataFrame:
        """
        Run a complete backtest using the specified strategy.
        
        Args:
            strategy: Trading strategy to use
            data: Market data
            start_date: Start date for backtest
            end_date: End date for backtest
            
        Returns:
            DataFrame with backtest results
        """
        self.logger.info(f"Starting backtest from {start_date} to {end_date}")
        
        # Get trading dates (every nth day based on rebalance frequency)
        all_dates = pd.date_range(start=start_date, end=end_date, freq='D')
        trading_dates = all_dates[::self.rebalance_frequency]
        
        # Initialize with first rebalance
        if len(trading_dates) > 0:
            first_date = trading_dates[0].strftime('%Y-%m-%d')
            self.rebalance_portfolio(strategy, data, first_date)
        
        # Run through all trading dates
        for date in trading_dates[1:]:
            date_str = date.strftime('%Y-%m-%d')
            
            # Update portfolio with current prices
            current_prices = {}
            for symbol in self.portfolio_state.positions.keys():
                if symbol in data:
                    try:
                        closest_date = min(data[symbol].index, 
                                         key=lambda x: abs(x - date))
                        current_prices[symbol] = data[symbol].loc[closest_date, 'adj_close']
                    except:
                        continue
            
            self.update_portfolio_prices(current_prices, date)
            
            # Rebalance portfolio
            self.rebalance_portfolio(strategy, data, date_str)
        
        # Convert portfolio history to DataFrame
        results_df = pd.DataFrame(self.portfolio_history)
        results_df.set_index('date', inplace=True)
        
        self.logger.info(f"Backtest completed. Final portfolio value: ${self.portfolio_state.total_value:.2f}")
        
        return results_df
    
    def get_portfolio_summary(self) -> Dict:
        """Get current portfolio summary."""
        return {
            'cash': self.portfolio_state.cash,
            'total_value': self.portfolio_state.total_value,
            'total_equity': self.portfolio_state.total_equity,
            'positions': len(self.portfolio_state.positions),
            'position_details': {
                symbol: {
                    'shares': pos.shares,
                    'avg_cost': pos.avg_cost,
                    'current_price': pos.current_price,
                    'market_value': pos.market_value,
                    'unrealized_pnl': pos.unrealized_pnl,
                    'weight': pos.market_value / self.portfolio_state.total_value if self.portfolio_state.total_value > 0 else 0
                }
                for symbol, pos in self.portfolio_state.positions.items()
            }
        }
    
    def get_transaction_history(self) -> pd.DataFrame:
        """Get transaction history as DataFrame."""
        if not self.transactions:
            return pd.DataFrame()
        
        transaction_data = []
        for txn in self.transactions:
            transaction_data.append({
                'date': txn.date,
                'symbol': txn.symbol,
                'action': txn.action,
                'shares': txn.shares,
                'price': txn.price,
                'value': txn.value,
                'commission': txn.commission
            })
        
        df = pd.DataFrame(transaction_data)
        df.set_index('date', inplace=True)
        return df
    
    def reset_portfolio(self):
        """Reset portfolio to initial state."""
        self.portfolio_state = PortfolioState(cash=self.initial_capital)
        self.transactions = []
        self.portfolio_history = []
        self.logger.info("Portfolio reset to initial state") 