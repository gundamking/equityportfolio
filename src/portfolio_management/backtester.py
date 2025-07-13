"""
Backtesting Module

Comprehensive backtesting framework for portfolio strategies.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging
from .portfolio_engine import PortfolioEngine
from .strategies import BaseStrategy
from .performance_analytics import PerformanceAnalytics

class Backtester:
    """
    Comprehensive backtesting framework for portfolio strategies.
    
    Provides systematic testing, comparison, and optimization capabilities.
    """
    
    def __init__(self, initial_capital: float = 1000000,
                 commission_rate: float = 0.001,
                 rebalance_frequency: int = 5):
        """
        Initialize backtester.
        
        Args:
            initial_capital: Starting capital
            commission_rate: Transaction commission rate
            rebalance_frequency: Days between rebalancing
        """
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.rebalance_frequency = rebalance_frequency
        self.logger = self._setup_logger()
        
        # Results storage
        self.results = {}
        self.performance_reports = {}
        
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
    
    def run_single_backtest(self, strategy: BaseStrategy, 
                          data: Dict[str, pd.DataFrame],
                          start_date: str, end_date: str,
                          strategy_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Run backtest for a single strategy.
        
        Args:
            strategy: Trading strategy to test
            data: Market data
            start_date: Backtest start date
            end_date: Backtest end date
            strategy_name: Name for the strategy (optional)
            
        Returns:
            Backtest results dictionary
        """
        if strategy_name is None:
            strategy_name = strategy.name
        
        self.logger.info(f"Running backtest for {strategy_name}")
        
        # Initialize portfolio engine
        portfolio = PortfolioEngine(
            initial_capital=self.initial_capital,
            commission_rate=self.commission_rate,
            rebalance_frequency=self.rebalance_frequency
        )
        
        # Run backtest
        try:
            results = portfolio.run_backtest(strategy, data, start_date, end_date)
            
            # Generate performance analytics
            analytics = PerformanceAnalytics()
            performance_report = analytics.generate_performance_report(results['total_value'])
            
            # Store results
            backtest_result = {
                'strategy_name': strategy_name,
                'portfolio_values': results['total_value'],
                'portfolio_history': results,
                'performance_report': performance_report,
                'transactions': portfolio.get_transaction_history(),
                'portfolio_summary': portfolio.get_portfolio_summary(),
                'start_date': start_date,
                'end_date': end_date
            }
            
            self.results[strategy_name] = backtest_result
            self.performance_reports[strategy_name] = performance_report
            
            self.logger.info(f"Backtest completed for {strategy_name}")
            return backtest_result
            
        except Exception as e:
            self.logger.error(f"Error in backtest for {strategy_name}: {str(e)}")
            raise
    
    def run_multiple_backtests(self, strategies: List[Tuple[BaseStrategy, str]],
                             data: Dict[str, pd.DataFrame],
                             start_date: str, end_date: str) -> Dict[str, Any]:
        """
        Run backtests for multiple strategies.
        
        Args:
            strategies: List of (strategy, name) tuples
            data: Market data
            start_date: Backtest start date
            end_date: Backtest end date
            
        Returns:
            Combined results dictionary
        """
        self.logger.info(f"Running backtests for {len(strategies)} strategies")
        
        all_results = {}
        
        for strategy, name in strategies:
            try:
                result = self.run_single_backtest(strategy, data, start_date, end_date, name)
                all_results[name] = result
            except Exception as e:
                self.logger.error(f"Failed to run backtest for {name}: {str(e)}")
                continue
        
        return all_results
    
    def compare_strategies(self, strategy_names: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Compare performance of multiple strategies.
        
        Args:
            strategy_names: List of strategy names to compare (None for all)
            
        Returns:
            Comparison DataFrame
        """
        if strategy_names is None:
            strategy_names = list(self.performance_reports.keys())
        
        comparison_data = []
        
        for name in strategy_names:
            if name in self.performance_reports:
                report = self.performance_reports[name]
                comparison_data.append({
                    'Strategy': name,
                    'Total Return': report['total_return'],
                    'Annual Return': report['annual_return'],
                    'Volatility': report['annual_volatility'],
                    'Sharpe Ratio': report['sharpe_ratio'],
                    'Sortino Ratio': report['sortino_ratio'],
                    'Calmar Ratio': report['calmar_ratio'],
                    'Max Drawdown': report['max_drawdown'],
                    'VaR (95%)': report['var_95'],
                    'Win Rate': report['win_rate']
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        return comparison_df
    
    def optimize_strategy_parameters(self, strategy_class, parameter_grid: Dict,
                                   data: Dict[str, pd.DataFrame],
                                   start_date: str, end_date: str,
                                   optimization_metric: str = 'sharpe_ratio') -> Dict:
        """
        Optimize strategy parameters using grid search.
        
        Args:
            strategy_class: Strategy class to optimize
            parameter_grid: Dictionary of parameter names and values to test
            data: Market data
            start_date: Backtest start date
            end_date: Backtest end date
            optimization_metric: Metric to optimize
            
        Returns:
            Optimization results
        """
        self.logger.info(f"Optimizing {strategy_class.__name__} parameters")
        
        # Generate parameter combinations
        param_names = list(parameter_grid.keys())
        param_values = list(parameter_grid.values())
        
        from itertools import product
        param_combinations = list(product(*param_values))
        
        optimization_results = []
        
        for i, param_combo in enumerate(param_combinations):
            # Create parameter dictionary
            params = dict(zip(param_names, param_combo))
            
            try:
                # Create strategy with parameters
                strategy = strategy_class(**params)
                strategy_name = f"{strategy_class.__name__}_{i}"
                
                # Run backtest
                result = self.run_single_backtest(strategy, data, start_date, end_date, strategy_name)
                
                # Store optimization result
                opt_result = {
                    'parameters': params,
                    'strategy_name': strategy_name,
                    'metric_value': result['performance_report'][optimization_metric],
                    'total_return': result['performance_report']['total_return'],
                    'sharpe_ratio': result['performance_report']['sharpe_ratio'],
                    'max_drawdown': result['performance_report']['max_drawdown']
                }
                optimization_results.append(opt_result)
                
                self.logger.info(f"Tested parameters {params}: {optimization_metric} = {opt_result['metric_value']:.4f}")
                
            except Exception as e:
                self.logger.error(f"Error testing parameters {params}: {str(e)}")
                continue
        
        # Find best parameters
        if optimization_results:
            best_result = max(optimization_results, key=lambda x: x['metric_value'])
            
            optimization_summary = {
                'best_parameters': best_result['parameters'],
                'best_metric_value': best_result['metric_value'],
                'best_strategy_name': best_result['strategy_name'],
                'all_results': optimization_results,
                'total_combinations_tested': len(optimization_results)
            }
            
            self.logger.info(f"Optimization complete. Best {optimization_metric}: {best_result['metric_value']:.4f}")
            self.logger.info(f"Best parameters: {best_result['parameters']}")
            
            return optimization_summary
        else:
            raise ValueError("No successful parameter combinations found")
    
    def walk_forward_analysis(self, strategy: BaseStrategy,
                            data: Dict[str, pd.DataFrame],
                            start_date: str, end_date: str,
                            window_months: int = 12,
                            step_months: int = 3) -> Dict:
        """
        Perform walk-forward analysis.
        
        Args:
            strategy: Strategy to test
            data: Market data
            start_date: Analysis start date
            end_date: Analysis end date
            window_months: Training window in months
            step_months: Step size in months
            
        Returns:
            Walk-forward results
        """
        self.logger.info("Starting walk-forward analysis")
        
        # Convert dates
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        walk_forward_results = []
        current_start = start_dt
        
        while current_start + pd.DateOffset(months=window_months) <= end_dt:
            # Define training and testing periods
            train_end = current_start + pd.DateOffset(months=window_months)
            test_start = train_end
            test_end = min(test_start + pd.DateOffset(months=step_months), end_dt)
            
            train_start_str = current_start.strftime('%Y-%m-%d')
            train_end_str = train_end.strftime('%Y-%m-%d')
            test_start_str = test_start.strftime('%Y-%m-%d')
            test_end_str = test_end.strftime('%Y-%m-%d')
            
            try:
                # Run backtest on testing period
                result = self.run_single_backtest(
                    strategy, data, test_start_str, test_end_str,
                    f"WF_{test_start_str}_to_{test_end_str}"
                )
                
                walk_forward_results.append({
                    'train_start': train_start_str,
                    'train_end': train_end_str,
                    'test_start': test_start_str,
                    'test_end': test_end_str,
                    'performance': result['performance_report'],
                    'portfolio_values': result['portfolio_values']
                })
                
                self.logger.info(f"Walk-forward period {test_start_str} to {test_end_str} completed")
                
            except Exception as e:
                self.logger.error(f"Error in walk-forward period {test_start_str} to {test_end_str}: {str(e)}")
            
            # Move to next period
            current_start += pd.DateOffset(months=step_months)
        
        # Aggregate results
        if walk_forward_results:
            all_returns = []
            for result in walk_forward_results:
                all_returns.extend(result['performance']['total_return'] for performance in [result['performance']])
            
            aggregate_performance = {
                'periods_tested': len(walk_forward_results),
                'average_return': np.mean(all_returns) if all_returns else 0,
                'return_std': np.std(all_returns) if all_returns else 0,
                'win_rate': sum(1 for r in all_returns if r > 0) / len(all_returns) if all_returns else 0
            }
            
            return {
                'walk_forward_results': walk_forward_results,
                'aggregate_performance': aggregate_performance
            }
        else:
            raise ValueError("No successful walk-forward periods")
    
    def generate_backtest_report(self, strategy_names: Optional[List[str]] = None) -> str:
        """
        Generate comprehensive backtest report.
        
        Args:
            strategy_names: Strategies to include in report
            
        Returns:
            Formatted report string
        """
        if strategy_names is None:
            strategy_names = list(self.results.keys())
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("COMPREHENSIVE BACKTEST REPORT")
        report_lines.append("=" * 80)
        
        # Summary table
        comparison_df = self.compare_strategies(strategy_names)
        report_lines.append("\nSTRATEGY PERFORMANCE COMPARISON:")
        report_lines.append("-" * 50)
        report_lines.append(comparison_df.to_string(index=False, float_format='%.4f'))
        
        # Individual strategy details
        for name in strategy_names:
            if name in self.results:
                result = self.results[name]
                report_lines.append(f"\n\nDETAILED ANALYSIS: {name}")
                report_lines.append("-" * 50)
                
                # Performance metrics
                perf = result['performance_report']
                report_lines.append(f"Total Return: {perf['total_return']:.2%}")
                report_lines.append(f"Annual Return: {perf['annual_return']:.2%}")
                report_lines.append(f"Volatility: {perf['annual_volatility']:.2%}")
                report_lines.append(f"Sharpe Ratio: {perf['sharpe_ratio']:.3f}")
                report_lines.append(f"Max Drawdown: {perf['max_drawdown']:.2%}")
                
                # Transaction summary
                transactions = result['transactions']
                if not transactions.empty:
                    report_lines.append(f"Total Transactions: {len(transactions)}")
                    report_lines.append(f"Total Commissions: ${transactions['commission'].sum():.2f}")
        
        report_lines.append("\n" + "=" * 80)
        
        return "\n".join(report_lines) 