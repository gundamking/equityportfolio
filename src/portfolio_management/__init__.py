"""
Equity Portfolio Management System

A comprehensive portfolio management system with advanced analytics,
backtesting capabilities, and risk management tools.
"""

__version__ = "1.0.0"
__author__ = "Your Name"

from .data_loader import DataLoader
from .portfolio_engine import PortfolioEngine
from .performance_analytics import PerformanceAnalytics
from .strategies import MomentumStrategy, DividendStrategy, MeanReversionStrategy, CombinedStrategy, BaseStrategy
from .risk_management import RiskManager
from .backtester import Backtester

__all__ = [
    "DataLoader",
    "PortfolioEngine", 
    "PerformanceAnalytics",
    "MomentumStrategy",
    "DividendStrategy",
    "MeanReversionStrategy",
    "CombinedStrategy",
    "BaseStrategy",
    "RiskManager",
    "Backtester",
] 