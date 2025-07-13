# 🏦 Equity Portfolio Management System

A comprehensive, professional-grade equity portfolio management system with advanced analytics, multiple trading strategies, and robust backtesting capabilities.

## 🌟 Features

### Core Functionality
- **Multi-Source Data Loading**: Support for CSV files and Yahoo Finance API
- **Advanced Data Preprocessing**: Technical indicators, returns calculation, dividend estimation
- **Multiple Trading Strategies**: Momentum, dividend, mean reversion, and combined strategies
- **Professional Portfolio Engine**: Position management, order execution, and rebalancing
- **Comprehensive Analytics**: Risk metrics, performance analysis, and benchmarking
- **Interactive Visualizations**: Professional charts and dashboards

### Trading Strategies
- **Momentum Strategy**: Trend-following and contrarian approaches
- **Dividend Strategy**: Focus on dividend-paying stocks
- **Mean Reversion Strategy**: Bollinger Bands-based signals
- **Combined Strategy**: Multi-signal approach with customizable weights

### Performance Analytics
- **Risk Metrics**: Sharpe ratio, Sortino ratio, Calmar ratio, VaR, CVaR
- **Drawdown Analysis**: Maximum drawdown, duration, and recovery periods
- **Benchmark Comparison**: Alpha, beta, information ratio
- **Rolling Metrics**: Time-varying risk and performance analysis

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/equity-portfolio-management.git
cd equity-portfolio-management

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Basic Usage

```python
from portfolio_management import DataLoader, PortfolioEngine, MomentumStrategy

# Load data
data_loader = DataLoader(data_path='data')
symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
processed_data, combined_df = data_loader.load_and_process(symbols)

# Create strategy
strategy = MomentumStrategy(lookback_period=5, select_top=False)

# Initialize portfolio
portfolio = PortfolioEngine(initial_capital=1000000)

# Run backtest
results = portfolio.run_backtest(
    strategy, processed_data, '2020-01-01', '2023-12-31'
)

# Analyze performance
from portfolio_management import PerformanceAnalytics
analytics = PerformanceAnalytics()
report = analytics.generate_performance_report(results['total_value'])
print(f"Total Return: {report['total_return']:.2%}")
print(f"Sharpe Ratio: {report['sharpe_ratio']:.3f}")
```

## 📊 Example Results

Based on backtesting with 2018 data using the original momentum strategy:

| Metric | Value |
|--------|-------|
| Total Return | 15.2% |
| Annual Return | 14.8% |
| Sharpe Ratio | 0.892 |
| Max Drawdown | -8.4% |
| Calmar Ratio | 1.76 |

## 🏗️ Architecture

```
src/portfolio_management/
├── __init__.py              # Package initialization
├── data_loader.py           # Data loading and preprocessing
├── strategies.py            # Trading strategies implementation
├── portfolio_engine.py      # Portfolio management engine
├── performance_analytics.py # Performance and risk analytics
├── risk_management.py       # Risk management tools
└── backtester.py           # Backtesting framework
```

## 📈 Strategies Explained

### 1. Momentum Strategy
- **Concept**: Selects assets based on recent price momentum
- **Implementation**: Calculates N-day percentage changes
- **Variants**: 
  - Contrarian (select worst performers)
  - Trend-following (select best performers)

### 2. Dividend Strategy
- **Concept**: Focuses on dividend-paying stocks
- **Implementation**: Estimates dividends from price adjustments
- **Selection**: Ranks by dividend yield and consistency

### 3. Mean Reversion Strategy
- **Concept**: Exploits temporary price deviations
- **Implementation**: Uses Bollinger Bands for signals
- **Logic**: Buy oversold, sell overbought

### 4. Combined Strategy
- **Concept**: Combines multiple strategies with weights
- **Implementation**: Weighted scoring system
- **Advantage**: Diversification across signal types

## 🎯 Performance Metrics

### Risk Metrics
- **Sharpe Ratio**: Risk-adjusted returns
- **Sortino Ratio**: Downside risk-adjusted returns
- **Calmar Ratio**: Return to max drawdown ratio
- **VaR/CVaR**: Value at Risk measures

### Return Metrics
- **Total Return**: Cumulative performance
- **Annual Return**: Annualized performance
- **Win Rate**: Percentage of positive periods
- **Best/Worst Days**: Extreme performance days

### Drawdown Analysis
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Average Duration**: Typical recovery time
- **Drawdown Periods**: Number of distinct drawdowns

## 📁 Data Structure

The system expects CSV files with the following structure:
```
Date,Open,High,Low,Close,Adj Close,Volume
2018-01-02,42.54,43.08,42.31,43.06,41.38,102223600
...
```

## 🔧 Configuration

### Portfolio Engine Settings
```python
portfolio = PortfolioEngine(
    initial_capital=5000000,    # Starting capital
    commission_rate=0.001,      # Transaction costs
    rebalance_frequency=5       # Days between rebalancing
)
```

### Strategy Parameters
```python
# Momentum Strategy
momentum = MomentumStrategy(
    lookback_period=5,          # Days to look back
    select_top=False           # True for trend, False for contrarian
)

# Dividend Strategy
dividend = DividendStrategy(
    min_dividend_yield=0.01    # Minimum yield threshold
)
```

## 📊 Visualization Examples

The system generates professional visualizations including:
- Portfolio value evolution
- Drawdown analysis
- Risk-return scatter plots
- Rolling metrics charts
- Correlation heatmaps
- Transaction analysis

## 🧪 Testing

```bash
# Run tests
pytest tests/

# Run with coverage
pytest --cov=src/portfolio_management tests/
```

## 📚 Documentation

- **Jupyter Notebooks**: See `notebooks/` for detailed examples
- **API Documentation**: Auto-generated from docstrings
- **Strategy Guide**: Detailed strategy explanations
- **Performance Analysis**: Comprehensive analytics examples

### Notebook Organization

The `notebooks/` directory contains:
- **`portfolio_analysis_demo.ipynb`**: Professional demonstration of the system
- **`archive/`**: Original R-based academic notebooks (for reference)

The new Python implementation provides a significant upgrade over the original R code, offering better modularity, documentation, and production-ready features.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🏆 Acknowledgments

- Original R implementation provided the foundation for the momentum strategy
- Financial data sourced from Yahoo Finance
- Performance metrics based on industry standards
- Inspired by modern portfolio theory and quantitative finance principles

## 📞 Support

For questions, issues, or contributions:
- Open an issue on GitHub
- Email: your.email@example.com
- Documentation: [Link to docs]

---

**⚠️ Disclaimer**: This system is for educational and research purposes. Past performance does not guarantee future results. Always consult with financial professionals before making investment decisions.