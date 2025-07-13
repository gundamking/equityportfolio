"""
Trading Strategies Module

Implements various trading strategies for portfolio management.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from abc import ABC, abstractmethod
import logging

class BaseStrategy(ABC):
    """Abstract base class for trading strategies."""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(__name__)
    
    @abstractmethod
    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Generate trading signals based on the strategy."""
        pass
    
    @abstractmethod
    def select_assets(self, data: Dict[str, pd.DataFrame], date: str, n_assets: int = 5) -> List[str]:
        """Select assets for the portfolio at a given date."""
        pass

class MomentumStrategy(BaseStrategy):
    """
    Momentum-based trading strategy.
    
    Selects assets based on their recent performance (momentum).
    Can select either top performers (positive momentum) or bottom performers (negative momentum).
    """
    
    def __init__(self, lookback_period: int = 5, select_top: bool = False):
        """
        Initialize momentum strategy.
        
        Args:
            lookback_period: Number of days to look back for momentum calculation
            select_top: If True, select top performers; if False, select bottom performers
        """
        super().__init__("Momentum Strategy")
        self.lookback_period = lookback_period
        self.select_top = select_top
        self.strategy_type = "Top Momentum" if select_top else "Bottom Momentum"
        
    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Generate momentum signals for all assets.
        
        Args:
            data: Dictionary of asset DataFrames
            
        Returns:
            DataFrame with momentum signals for each asset
        """
        signals = {}
        
        for symbol, df in data.items():
            if 'adj_close' in df.columns:
                # Calculate momentum (percentage change over lookback period)
                momentum = df['adj_close'].pct_change(self.lookback_period) * 100
                signals[f"{symbol}_momentum"] = momentum
                
                # Generate binary signals based on momentum
                if self.select_top:
                    # Buy signal for positive momentum
                    signals[f"{symbol}_signal"] = (momentum > 0).astype(int)
                else:
                    # Buy signal for negative momentum (contrarian)
                    signals[f"{symbol}_signal"] = (momentum < 0).astype(int)
        
        return pd.DataFrame(signals)
    
    def select_assets(self, data: Dict[str, pd.DataFrame], date: str, n_assets: int = 5) -> List[str]:
        """
        Select assets based on momentum at a specific date.
        
        Args:
            data: Dictionary of asset DataFrames
            date: Date for asset selection
            n_assets: Number of assets to select
            
        Returns:
            List of selected asset symbols
        """
        momentum_scores = {}
        
        for symbol, df in data.items():
            if 'adj_close' in df.columns:
                try:
                    # Find the closest date
                    available_dates = df.index
                    closest_date = min(available_dates, key=lambda x: abs(x - pd.to_datetime(date)))
                    
                    # Calculate momentum at this date
                    momentum = df.loc[closest_date, 'adj_close'] / df.loc[closest_date - pd.Timedelta(days=self.lookback_period), 'adj_close'] - 1
                    momentum_scores[symbol] = momentum * 100
                    
                except (KeyError, IndexError):
                    self.logger.warning(f"Could not calculate momentum for {symbol} at {date}")
                    continue
        
        # Sort by momentum
        sorted_assets = sorted(momentum_scores.items(), key=lambda x: x[1], reverse=self.select_top)
        
        # Return top n_assets
        selected_assets = [asset[0] for asset in sorted_assets[:n_assets]]
        
        self.logger.info(f"Selected assets for {date}: {selected_assets}")
        return selected_assets
    
    def get_momentum_rankings(self, data: Dict[str, pd.DataFrame], date: str) -> pd.DataFrame:
        """
        Get momentum rankings for all assets at a specific date.
        
        Args:
            data: Dictionary of asset DataFrames
            date: Date for ranking
            
        Returns:
            DataFrame with momentum rankings
        """
        momentum_data = []
        
        for symbol, df in data.items():
            if 'adj_close' in df.columns:
                try:
                    available_dates = df.index
                    closest_date = min(available_dates, key=lambda x: abs(x - pd.to_datetime(date)))
                    
                    current_price = df.loc[closest_date, 'adj_close']
                    past_price = df.loc[closest_date - pd.Timedelta(days=self.lookback_period), 'adj_close']
                    momentum = (current_price / past_price - 1) * 100
                    
                    momentum_data.append({
                        'Symbol': symbol,
                        'Current_Price': current_price,
                        'Past_Price': past_price,
                        'Momentum_%': momentum,
                        'Date': closest_date
                    })
                    
                except (KeyError, IndexError):
                    continue
        
        df_momentum = pd.DataFrame(momentum_data)
        df_momentum = df_momentum.sort_values('Momentum_%', ascending=not self.select_top)
        
        return df_momentum

class DividendStrategy(BaseStrategy):
    """
    Dividend-focused trading strategy.
    
    Selects assets based on dividend yield and dividend growth.
    """
    
    def __init__(self, min_dividend_yield: float = 0.01):
        """
        Initialize dividend strategy.
        
        Args:
            min_dividend_yield: Minimum dividend yield threshold
        """
        super().__init__("Dividend Strategy")
        self.min_dividend_yield = min_dividend_yield
        
    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Generate dividend-based signals.
        
        Args:
            data: Dictionary of asset DataFrames
            
        Returns:
            DataFrame with dividend signals
        """
        signals = {}
        
        for symbol, df in data.items():
            if 'dividend_est' in df.columns:
                # Calculate dividend yield
                dividend_yield = df['dividend_est'] / df['adj_close']
                signals[f"{symbol}_dividend_yield"] = dividend_yield
                
                # Generate buy signals for assets with good dividend yield
                signals[f"{symbol}_signal"] = (dividend_yield > self.min_dividend_yield).astype(int)
        
        return pd.DataFrame(signals)
    
    def select_assets(self, data: Dict[str, pd.DataFrame], date: str, n_assets: int = 5) -> List[str]:
        """
        Select assets based on dividend characteristics.
        
        Args:
            data: Dictionary of asset DataFrames
            date: Date for asset selection
            n_assets: Number of assets to select
            
        Returns:
            List of selected asset symbols
        """
        dividend_scores = {}
        
        for symbol, df in data.items():
            if 'dividend_est' in df.columns:
                try:
                    available_dates = df.index
                    closest_date = min(available_dates, key=lambda x: abs(x - pd.to_datetime(date)))
                    
                    # Calculate recent dividend activity
                    recent_dividends = df.loc[closest_date - pd.Timedelta(days=90):closest_date, 'dividend_est'].sum()
                    current_price = df.loc[closest_date, 'adj_close']
                    
                    # Calculate annualized dividend yield
                    dividend_yield = (recent_dividends * 4) / current_price  # Approximate annual yield
                    dividend_scores[symbol] = dividend_yield
                    
                except (KeyError, IndexError):
                    self.logger.warning(f"Could not calculate dividend score for {symbol} at {date}")
                    continue
        
        # Sort by dividend yield
        sorted_assets = sorted(dividend_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Return top n_assets
        selected_assets = [asset[0] for asset in sorted_assets[:n_assets]]
        
        self.logger.info(f"Selected dividend assets for {date}: {selected_assets}")
        return selected_assets

class MeanReversionStrategy(BaseStrategy):
    """
    Mean reversion strategy using Bollinger Bands.
    """
    
    def __init__(self, lookback_period: int = 20, std_multiplier: float = 2.0):
        """
        Initialize mean reversion strategy.
        
        Args:
            lookback_period: Period for moving average calculation
            std_multiplier: Standard deviation multiplier for bands
        """
        super().__init__("Mean Reversion Strategy")
        self.lookback_period = lookback_period
        self.std_multiplier = std_multiplier
        
    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Generate mean reversion signals."""
        signals = {}
        
        for symbol, df in data.items():
            if 'adj_close' in df.columns:
                # Calculate Bollinger Bands
                sma = df['adj_close'].rolling(window=self.lookback_period).mean()
                std = df['adj_close'].rolling(window=self.lookback_period).std()
                
                upper_band = sma + (std * self.std_multiplier)
                lower_band = sma - (std * self.std_multiplier)
                
                # Generate signals
                # Buy when price touches lower band (oversold)
                # Sell when price touches upper band (overbought)
                buy_signal = (df['adj_close'] <= lower_band).astype(int)
                sell_signal = (df['adj_close'] >= upper_band).astype(int)
                
                signals[f"{symbol}_buy_signal"] = buy_signal
                signals[f"{symbol}_sell_signal"] = sell_signal
                signals[f"{symbol}_bb_position"] = (df['adj_close'] - sma) / (upper_band - lower_band)
        
        return pd.DataFrame(signals)
    
    def select_assets(self, data: Dict[str, pd.DataFrame], date: str, n_assets: int = 5) -> List[str]:
        """Select assets based on mean reversion signals."""
        reversion_scores = {}
        
        for symbol, df in data.items():
            if 'adj_close' in df.columns:
                try:
                    available_dates = df.index
                    closest_date = min(available_dates, key=lambda x: abs(x - pd.to_datetime(date)))
                    
                    # Calculate position relative to Bollinger Bands
                    recent_data = df.loc[:closest_date].tail(self.lookback_period)
                    sma = recent_data['adj_close'].mean()
                    std = recent_data['adj_close'].std()
                    
                    current_price = df.loc[closest_date, 'adj_close']
                    bb_position = (current_price - sma) / (2 * std)
                    
                    # Lower scores indicate more oversold (better buy opportunities)
                    reversion_scores[symbol] = bb_position
                    
                except (KeyError, IndexError):
                    continue
        
        # Sort by reversion score (lowest first - most oversold)
        sorted_assets = sorted(reversion_scores.items(), key=lambda x: x[1])
        
        # Return top n_assets
        selected_assets = [asset[0] for asset in sorted_assets[:n_assets]]
        
        return selected_assets

class CombinedStrategy(BaseStrategy):
    """
    Combined strategy that uses multiple signals.
    """
    
    def __init__(self, strategies: List[BaseStrategy], weights: Optional[List[float]] = None):
        """
        Initialize combined strategy.
        
        Args:
            strategies: List of base strategies to combine
            weights: Weights for each strategy (if None, equal weights)
        """
        super().__init__("Combined Strategy")
        self.strategies = strategies
        self.weights = weights or [1.0 / len(strategies)] * len(strategies)
        
        if len(self.weights) != len(self.strategies):
            raise ValueError("Number of weights must match number of strategies")
    
    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Generate combined signals from multiple strategies."""
        all_signals = []
        
        for strategy in self.strategies:
            signals = strategy.generate_signals(data)
            all_signals.append(signals)
        
        # Combine signals (simple average for now)
        combined_signals = pd.concat(all_signals, axis=1)
        
        return combined_signals
    
    def select_assets(self, data: Dict[str, pd.DataFrame], date: str, n_assets: int = 5) -> List[str]:
        """Select assets using combined strategy scores."""
        combined_scores = {}
        
        # Get selections from each strategy
        for i, strategy in enumerate(self.strategies):
            try:
                selected_assets = strategy.select_assets(data, date, len(data))
                
                # Assign scores based on ranking
                for j, asset in enumerate(selected_assets):
                    if asset not in combined_scores:
                        combined_scores[asset] = 0
                    
                    # Higher rank = lower score (better)
                    score = (len(data) - j) * self.weights[i]
                    combined_scores[asset] += score
                    
            except Exception as e:
                self.logger.warning(f"Error in strategy {strategy.name}: {str(e)}")
                continue
        
        # Sort by combined score
        sorted_assets = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Return top n_assets
        selected_assets = [asset[0] for asset in sorted_assets[:n_assets]]
        
        return selected_assets 