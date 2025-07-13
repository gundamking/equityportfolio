"""
Risk Management Module

Implements risk management tools and controls for portfolio management.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

class RiskManager:
    """
    Risk management system for portfolio controls and monitoring.
    
    Provides position sizing, risk limits, and portfolio constraints.
    """
    
    def __init__(self, max_position_size: float = 0.2, 
                 max_sector_exposure: float = 0.4,
                 max_volatility: float = 0.3):
        """
        Initialize risk manager.
        
        Args:
            max_position_size: Maximum weight per position (0.2 = 20%)
            max_sector_exposure: Maximum sector exposure
            max_volatility: Maximum portfolio volatility
        """
        self.max_position_size = max_position_size
        self.max_sector_exposure = max_sector_exposure
        self.max_volatility = max_volatility
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
    
    def check_position_limits(self, portfolio_weights: Dict[str, float]) -> Dict[str, bool]:
        """
        Check if positions exceed size limits.
        
        Args:
            portfolio_weights: Dictionary of symbol -> weight
            
        Returns:
            Dictionary of symbol -> is_within_limit
        """
        position_checks = {}
        
        for symbol, weight in portfolio_weights.items():
            is_within_limit = abs(weight) <= self.max_position_size
            position_checks[symbol] = is_within_limit
            
            if not is_within_limit:
                self.logger.warning(f"Position {symbol} exceeds limit: {weight:.2%} > {self.max_position_size:.2%}")
        
        return position_checks
    
    def calculate_portfolio_volatility(self, returns: pd.DataFrame, 
                                     weights: Dict[str, float]) -> float:
        """
        Calculate portfolio volatility.
        
        Args:
            returns: DataFrame of asset returns
            weights: Portfolio weights
            
        Returns:
            Portfolio volatility
        """
        # Align weights with returns columns
        aligned_weights = []
        aligned_returns = []
        
        for symbol in returns.columns:
            if symbol in weights:
                aligned_weights.append(weights[symbol])
                aligned_returns.append(returns[symbol])
        
        if not aligned_weights:
            return 0.0
        
        weights_array = np.array(aligned_weights)
        returns_matrix = pd.DataFrame(aligned_returns).T
        
        # Calculate covariance matrix
        cov_matrix = returns_matrix.cov().values
        
        # Calculate portfolio variance
        portfolio_variance = np.dot(weights_array.T, np.dot(cov_matrix, weights_array))
        
        # Annualize volatility
        portfolio_volatility = np.sqrt(portfolio_variance * 252)
        
        return portfolio_volatility
    
    def check_volatility_limit(self, returns: pd.DataFrame, 
                             weights: Dict[str, float]) -> bool:
        """
        Check if portfolio volatility is within limits.
        
        Args:
            returns: Asset returns
            weights: Portfolio weights
            
        Returns:
            True if within limit
        """
        portfolio_vol = self.calculate_portfolio_volatility(returns, weights)
        is_within_limit = portfolio_vol <= self.max_volatility
        
        if not is_within_limit:
            self.logger.warning(f"Portfolio volatility exceeds limit: {portfolio_vol:.2%} > {self.max_volatility:.2%}")
        
        return is_within_limit
    
    def calculate_var(self, portfolio_returns: pd.Series, 
                     confidence_level: float = 0.05) -> float:
        """
        Calculate Value at Risk.
        
        Args:
            portfolio_returns: Portfolio return series
            confidence_level: Confidence level (0.05 = 95% VaR)
            
        Returns:
            VaR value
        """
        return np.percentile(portfolio_returns, confidence_level * 100)
    
    def calculate_expected_shortfall(self, portfolio_returns: pd.Series,
                                   confidence_level: float = 0.05) -> float:
        """
        Calculate Expected Shortfall (Conditional VaR).
        
        Args:
            portfolio_returns: Portfolio return series
            confidence_level: Confidence level
            
        Returns:
            Expected shortfall
        """
        var = self.calculate_var(portfolio_returns, confidence_level)
        expected_shortfall = portfolio_returns[portfolio_returns <= var].mean()
        return expected_shortfall
    
    def check_concentration_risk(self, weights: Dict[str, float], 
                               max_concentration: float = 0.3) -> bool:
        """
        Check portfolio concentration risk.
        
        Args:
            weights: Portfolio weights
            max_concentration: Maximum single position weight
            
        Returns:
            True if concentration is acceptable
        """
        max_weight = max(abs(w) for w in weights.values()) if weights else 0
        is_acceptable = max_weight <= max_concentration
        
        if not is_acceptable:
            self.logger.warning(f"Concentration risk: max weight {max_weight:.2%} > {max_concentration:.2%}")
        
        return is_acceptable
    
    def apply_risk_controls(self, target_weights: Dict[str, float],
                          returns_data: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        """
        Apply risk controls to target weights.
        
        Args:
            target_weights: Desired portfolio weights
            returns_data: Historical returns for volatility calculation
            
        Returns:
            Risk-adjusted weights
        """
        adjusted_weights = target_weights.copy()
        
        # Apply position size limits
        for symbol in adjusted_weights:
            if abs(adjusted_weights[symbol]) > self.max_position_size:
                sign = np.sign(adjusted_weights[symbol])
                adjusted_weights[symbol] = sign * self.max_position_size
                self.logger.info(f"Capped {symbol} position at {self.max_position_size:.2%}")
        
        # Normalize weights to sum to 1
        total_weight = sum(abs(w) for w in adjusted_weights.values())
        if total_weight > 0:
            for symbol in adjusted_weights:
                adjusted_weights[symbol] /= total_weight
        
        return adjusted_weights
    
    def generate_risk_report(self, portfolio_weights: Dict[str, float],
                           portfolio_returns: pd.Series,
                           returns_data: Optional[pd.DataFrame] = None) -> Dict:
        """
        Generate comprehensive risk report.
        
        Args:
            portfolio_weights: Current portfolio weights
            portfolio_returns: Portfolio return history
            returns_data: Asset returns for calculations
            
        Returns:
            Risk report dictionary
        """
        report = {}
        
        # Position analysis
        report['max_position'] = max(abs(w) for w in portfolio_weights.values()) if portfolio_weights else 0
        report['position_count'] = len(portfolio_weights)
        report['concentration_risk'] = report['max_position'] > self.max_position_size
        
        # Volatility analysis
        if returns_data is not None:
            portfolio_vol = self.calculate_portfolio_volatility(returns_data, portfolio_weights)
            report['portfolio_volatility'] = portfolio_vol
            report['volatility_breach'] = portfolio_vol > self.max_volatility
        
        # VaR analysis
        if len(portfolio_returns) > 0:
            report['var_95'] = self.calculate_var(portfolio_returns, 0.05)
            report['var_99'] = self.calculate_var(portfolio_returns, 0.01)
            report['expected_shortfall_95'] = self.calculate_expected_shortfall(portfolio_returns, 0.05)
        
        # Risk limits
        report['position_limit'] = self.max_position_size
        report['volatility_limit'] = self.max_volatility
        
        return report 