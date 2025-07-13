"""
Performance Analytics Module

Comprehensive performance analysis and risk metrics for portfolio management.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class PerformanceAnalytics:
    """
    Comprehensive performance analytics for portfolio management.
    
    Provides risk metrics, return analysis, drawdown analysis,
    and benchmarking capabilities.
    """
    
    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize performance analytics.
        
        Args:
            risk_free_rate: Annual risk-free rate for Sharpe ratio calculation
        """
        self.risk_free_rate = risk_free_rate
        self.trading_days = 252  # Annual trading days
        
    def calculate_returns(self, portfolio_values: pd.Series) -> pd.Series:
        """
        Calculate portfolio returns.
        
        Args:
            portfolio_values: Time series of portfolio values
            
        Returns:
            Series of portfolio returns
        """
        returns = portfolio_values.pct_change().dropna()
        return returns
    
    def calculate_log_returns(self, portfolio_values: pd.Series) -> pd.Series:
        """
        Calculate log returns.
        
        Args:
            portfolio_values: Time series of portfolio values
            
        Returns:
            Series of log returns
        """
        log_returns = np.log(portfolio_values / portfolio_values.shift(1)).dropna()
        return log_returns
    
    def calculate_cumulative_returns(self, returns: pd.Series) -> pd.Series:
        """
        Calculate cumulative returns.
        
        Args:
            returns: Series of returns
            
        Returns:
            Series of cumulative returns
        """
        cumulative_returns = (1 + returns).cumprod() - 1
        return cumulative_returns
    
    def calculate_sharpe_ratio(self, returns: pd.Series, 
                             risk_free_rate: Optional[float] = None) -> float:
        """
        Calculate Sharpe ratio.
        
        Args:
            returns: Series of returns
            risk_free_rate: Risk-free rate (if None, uses instance default)
            
        Returns:
            Sharpe ratio
        """
        if risk_free_rate is None:
            risk_free_rate = self.risk_free_rate
        
        # Convert to daily risk-free rate
        daily_rf = risk_free_rate / self.trading_days
        
        excess_returns = returns - daily_rf
        
        if excess_returns.std() == 0:
            return 0.0
        
        sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(self.trading_days)
        return sharpe_ratio
    
    def calculate_sortino_ratio(self, returns: pd.Series,
                              risk_free_rate: Optional[float] = None) -> float:
        """
        Calculate Sortino ratio (downside deviation).
        
        Args:
            returns: Series of returns
            risk_free_rate: Risk-free rate
            
        Returns:
            Sortino ratio
        """
        if risk_free_rate is None:
            risk_free_rate = self.risk_free_rate
        
        daily_rf = risk_free_rate / self.trading_days
        excess_returns = returns - daily_rf
        
        # Calculate downside deviation
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return np.inf if excess_returns.mean() > 0 else 0.0
        
        downside_deviation = downside_returns.std()
        sortino_ratio = excess_returns.mean() / downside_deviation * np.sqrt(self.trading_days)
        
        return sortino_ratio
    
    def calculate_max_drawdown(self, portfolio_values: pd.Series) -> Dict[str, float]:
        """
        Calculate maximum drawdown and related metrics.
        
        Args:
            portfolio_values: Time series of portfolio values
            
        Returns:
            Dictionary with drawdown metrics
        """
        # Calculate running maximum
        running_max = portfolio_values.expanding().max()
        
        # Calculate drawdown
        drawdown = (portfolio_values - running_max) / running_max
        
        # Find maximum drawdown
        max_drawdown = drawdown.min()
        
        # Find drawdown periods
        drawdown_periods = []
        in_drawdown = False
        start_date = None
        
        for date, dd in drawdown.items():
            if dd < 0 and not in_drawdown:
                # Start of drawdown
                in_drawdown = True
                start_date = date
            elif dd >= 0 and in_drawdown:
                # End of drawdown
                in_drawdown = False
                if start_date is not None:
                    drawdown_periods.append((start_date, date))
        
        # Calculate average drawdown duration
        if drawdown_periods:
            durations = [(end - start).days for start, end in drawdown_periods]
            avg_duration = np.mean(durations)
            max_duration = max(durations)
        else:
            avg_duration = 0
            max_duration = 0
        
        return {
            'max_drawdown': max_drawdown,
            'avg_drawdown_duration': avg_duration,
            'max_drawdown_duration': max_duration,
            'drawdown_periods': len(drawdown_periods),
            'drawdown_series': drawdown
        }
    
    def calculate_var(self, returns: pd.Series, confidence_level: float = 0.05) -> float:
        """
        Calculate Value at Risk (VaR).
        
        Args:
            returns: Series of returns
            confidence_level: Confidence level (e.g., 0.05 for 95% VaR)
            
        Returns:
            VaR value
        """
        var = np.percentile(returns, confidence_level * 100)
        return var
    
    def calculate_cvar(self, returns: pd.Series, confidence_level: float = 0.05) -> float:
        """
        Calculate Conditional Value at Risk (CVaR).
        
        Args:
            returns: Series of returns
            confidence_level: Confidence level
            
        Returns:
            CVaR value
        """
        var = self.calculate_var(returns, confidence_level)
        cvar = returns[returns <= var].mean()
        return cvar
    
    def calculate_beta(self, portfolio_returns: pd.Series, 
                      benchmark_returns: pd.Series) -> float:
        """
        Calculate portfolio beta relative to benchmark.
        
        Args:
            portfolio_returns: Portfolio returns
            benchmark_returns: Benchmark returns
            
        Returns:
            Beta coefficient
        """
        # Align the series
        aligned_data = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
        
        if len(aligned_data) < 2:
            return 0.0
        
        portfolio_aligned = aligned_data.iloc[:, 0]
        benchmark_aligned = aligned_data.iloc[:, 1]
        
        covariance = np.cov(portfolio_aligned, benchmark_aligned)[0, 1]
        benchmark_variance = np.var(benchmark_aligned)
        
        if benchmark_variance == 0:
            return 0.0
        
        beta = covariance / benchmark_variance
        return beta
    
    def calculate_alpha(self, portfolio_returns: pd.Series,
                       benchmark_returns: pd.Series,
                       risk_free_rate: Optional[float] = None) -> float:
        """
        Calculate Jensen's alpha.
        
        Args:
            portfolio_returns: Portfolio returns
            benchmark_returns: Benchmark returns
            risk_free_rate: Risk-free rate
            
        Returns:
            Alpha coefficient
        """
        if risk_free_rate is None:
            risk_free_rate = self.risk_free_rate
        
        daily_rf = risk_free_rate / self.trading_days
        
        beta = self.calculate_beta(portfolio_returns, benchmark_returns)
        
        # Calculate average excess returns
        portfolio_excess = portfolio_returns.mean() - daily_rf
        benchmark_excess = benchmark_returns.mean() - daily_rf
        
        alpha = portfolio_excess - beta * benchmark_excess
        
        # Annualize alpha
        alpha_annual = alpha * self.trading_days
        
        return alpha_annual
    
    def calculate_information_ratio(self, portfolio_returns: pd.Series,
                                  benchmark_returns: pd.Series) -> float:
        """
        Calculate Information Ratio.
        
        Args:
            portfolio_returns: Portfolio returns
            benchmark_returns: Benchmark returns
            
        Returns:
            Information ratio
        """
        # Align the series
        aligned_data = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
        
        if len(aligned_data) < 2:
            return 0.0
        
        portfolio_aligned = aligned_data.iloc[:, 0]
        benchmark_aligned = aligned_data.iloc[:, 1]
        
        # Calculate tracking error (active return)
        active_returns = portfolio_aligned - benchmark_aligned
        
        if active_returns.std() == 0:
            return 0.0
        
        information_ratio = active_returns.mean() / active_returns.std() * np.sqrt(self.trading_days)
        
        return information_ratio
    
    def calculate_calmar_ratio(self, portfolio_values: pd.Series) -> float:
        """
        Calculate Calmar Ratio (Annual Return / Max Drawdown).
        
        Args:
            portfolio_values: Time series of portfolio values
            
        Returns:
            Calmar ratio
        """
        returns = self.calculate_returns(portfolio_values)
        annual_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) ** (self.trading_days / len(portfolio_values)) - 1
        
        drawdown_metrics = self.calculate_max_drawdown(portfolio_values)
        max_drawdown = abs(drawdown_metrics['max_drawdown'])
        
        if max_drawdown == 0:
            return np.inf if annual_return > 0 else 0.0
        
        calmar_ratio = annual_return / max_drawdown
        return calmar_ratio
    
    def generate_performance_report(self, portfolio_values: pd.Series,
                                  benchmark_values: Optional[pd.Series] = None) -> Dict:
        """
        Generate comprehensive performance report.
        
        Args:
            portfolio_values: Portfolio values time series
            benchmark_values: Benchmark values time series (optional)
            
        Returns:
            Dictionary with performance metrics
        """
        # Calculate returns
        portfolio_returns = self.calculate_returns(portfolio_values)
        
        # Basic metrics
        total_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1
        annual_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) ** (self.trading_days / len(portfolio_values)) - 1
        annual_volatility = portfolio_returns.std() * np.sqrt(self.trading_days)
        
        # Risk metrics
        sharpe_ratio = self.calculate_sharpe_ratio(portfolio_returns)
        sortino_ratio = self.calculate_sortino_ratio(portfolio_returns)
        
        # Drawdown analysis
        drawdown_metrics = self.calculate_max_drawdown(portfolio_values)
        
        # VaR metrics
        var_95 = self.calculate_var(portfolio_returns, 0.05)
        cvar_95 = self.calculate_cvar(portfolio_returns, 0.05)
        
        # Calmar ratio
        calmar_ratio = self.calculate_calmar_ratio(portfolio_values)
        
        # Win rate
        win_rate = (portfolio_returns > 0).mean()
        
        # Best and worst days
        best_day = portfolio_returns.max()
        worst_day = portfolio_returns.min()
        
        report = {
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': drawdown_metrics['max_drawdown'],
            'avg_drawdown_duration': drawdown_metrics['avg_drawdown_duration'],
            'var_95': var_95,
            'cvar_95': cvar_95,
            'win_rate': win_rate,
            'best_day': best_day,
            'worst_day': worst_day,
            'total_trades': len(portfolio_values),
            'start_date': portfolio_values.index[0],
            'end_date': portfolio_values.index[-1]
        }
        
        # Add benchmark comparison if provided
        if benchmark_values is not None:
            benchmark_returns = self.calculate_returns(benchmark_values)
            
            # Align dates
            aligned_data = pd.concat([portfolio_values, benchmark_values], axis=1).dropna()
            if len(aligned_data) > 0:
                portfolio_aligned = aligned_data.iloc[:, 0]
                benchmark_aligned = aligned_data.iloc[:, 1]
                
                portfolio_returns_aligned = self.calculate_returns(portfolio_aligned)
                benchmark_returns_aligned = self.calculate_returns(benchmark_aligned)
                
                # Benchmark metrics
                benchmark_total_return = (benchmark_aligned.iloc[-1] / benchmark_aligned.iloc[0]) - 1
                benchmark_annual_return = (benchmark_aligned.iloc[-1] / benchmark_aligned.iloc[0]) ** (self.trading_days / len(benchmark_aligned)) - 1
                benchmark_volatility = benchmark_returns_aligned.std() * np.sqrt(self.trading_days)
                
                # Relative metrics
                beta = self.calculate_beta(portfolio_returns_aligned, benchmark_returns_aligned)
                alpha = self.calculate_alpha(portfolio_returns_aligned, benchmark_returns_aligned)
                information_ratio = self.calculate_information_ratio(portfolio_returns_aligned, benchmark_returns_aligned)
                
                report.update({
                    'benchmark_total_return': benchmark_total_return,
                    'benchmark_annual_return': benchmark_annual_return,
                    'benchmark_volatility': benchmark_volatility,
                    'beta': beta,
                    'alpha': alpha,
                    'information_ratio': information_ratio,
                    'excess_return': annual_return - benchmark_annual_return
                })
        
        return report
    
    def plot_performance_chart(self, portfolio_values: pd.Series,
                             benchmark_values: Optional[pd.Series] = None,
                             figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Create performance visualization chart.
        
        Args:
            portfolio_values: Portfolio values time series
            benchmark_values: Benchmark values time series (optional)
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Portfolio Performance Analysis', fontsize=16)
        
        # Normalize values to start at 100
        portfolio_normalized = (portfolio_values / portfolio_values.iloc[0]) * 100
        
        # Plot 1: Portfolio value over time
        axes[0, 0].plot(portfolio_normalized.index, portfolio_normalized.values, 
                       label='Portfolio', linewidth=2)
        
        if benchmark_values is not None:
            benchmark_normalized = (benchmark_values / benchmark_values.iloc[0]) * 100
            axes[0, 0].plot(benchmark_normalized.index, benchmark_normalized.values, 
                           label='Benchmark', linewidth=2, alpha=0.7)
        
        axes[0, 0].set_title('Portfolio Value Over Time')
        axes[0, 0].set_ylabel('Value (Base = 100)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Drawdown
        drawdown_metrics = self.calculate_max_drawdown(portfolio_values)
        drawdown_series = drawdown_metrics['drawdown_series']
        
        axes[0, 1].fill_between(drawdown_series.index, drawdown_series.values, 0, 
                               alpha=0.7, color='red')
        axes[0, 1].set_title('Drawdown')
        axes[0, 1].set_ylabel('Drawdown (%)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Returns distribution
        portfolio_returns = self.calculate_returns(portfolio_values)
        axes[1, 0].hist(portfolio_returns, bins=50, alpha=0.7, edgecolor='black')
        axes[1, 0].axvline(portfolio_returns.mean(), color='red', linestyle='--', 
                          label=f'Mean: {portfolio_returns.mean():.4f}')
        axes[1, 0].set_title('Returns Distribution')
        axes[1, 0].set_xlabel('Daily Returns')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Rolling Sharpe ratio
        rolling_sharpe = portfolio_returns.rolling(window=60).apply(
            lambda x: self.calculate_sharpe_ratio(x) if len(x) == 60 else np.nan
        )
        
        axes[1, 1].plot(rolling_sharpe.index, rolling_sharpe.values)
        axes[1, 1].axhline(y=1.0, color='red', linestyle='--', alpha=0.7)
        axes[1, 1].set_title('Rolling 60-Day Sharpe Ratio')
        axes[1, 1].set_ylabel('Sharpe Ratio')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def create_performance_summary_table(self, report: Dict) -> pd.DataFrame:
        """
        Create a formatted performance summary table.
        
        Args:
            report: Performance report dictionary
            
        Returns:
            Formatted DataFrame
        """
        metrics = [
            ('Total Return', f"{report['total_return']:.2%}"),
            ('Annual Return', f"{report['annual_return']:.2%}"),
            ('Annual Volatility', f"{report['annual_volatility']:.2%}"),
            ('Sharpe Ratio', f"{report['sharpe_ratio']:.3f}"),
            ('Sortino Ratio', f"{report['sortino_ratio']:.3f}"),
            ('Calmar Ratio', f"{report['calmar_ratio']:.3f}"),
            ('Max Drawdown', f"{report['max_drawdown']:.2%}"),
            ('VaR (95%)', f"{report['var_95']:.2%}"),
            ('CVaR (95%)', f"{report['cvar_95']:.2%}"),
            ('Win Rate', f"{report['win_rate']:.2%}"),
            ('Best Day', f"{report['best_day']:.2%}"),
            ('Worst Day', f"{report['worst_day']:.2%}")
        ]
        
        # Add benchmark metrics if available
        if 'beta' in report:
            metrics.extend([
                ('Beta', f"{report['beta']:.3f}"),
                ('Alpha', f"{report['alpha']:.2%}"),
                ('Information Ratio', f"{report['information_ratio']:.3f}"),
                ('Excess Return', f"{report['excess_return']:.2%}")
            ])
        
        df = pd.DataFrame(metrics, columns=['Metric', 'Value'])
        return df 