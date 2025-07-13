#!/usr/bin/env python3
"""
Simple Portfolio Management Example

This script demonstrates the basic usage of the equity portfolio management system.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from portfolio_management import (
    DataLoader, 
    PortfolioEngine, 
    PerformanceAnalytics,
    MomentumStrategy
)

def main():
    print("🏦 Equity Portfolio Management - Simple Example")
    print("=" * 50)
    
    # Define parameters
    SYMBOLS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    INITIAL_CAPITAL = 1_000_000
    BACKTEST_START = '2018-01-15'
    BACKTEST_END = '2018-12-31'
    
    print(f"📊 Universe: {', '.join(SYMBOLS)}")
    print(f"💰 Initial Capital: ${INITIAL_CAPITAL:,}")
    print(f"📅 Period: {BACKTEST_START} to {BACKTEST_END}")
    
    # Step 1: Load Data
    print("\n1️⃣ Loading Data...")
    data_loader = DataLoader(data_path='data')
    
    try:
        processed_data, combined_df = data_loader.load_and_process(SYMBOLS, source='csv')
        print(f"✅ Successfully loaded data for {len(processed_data)} symbols")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Step 2: Create Strategy
    print("\n2️⃣ Creating Strategy...")
    strategy = MomentumStrategy(lookback_period=5, select_top=False)
    print(f"📈 Strategy: {strategy.name} ({strategy.strategy_type})")
    
    # Step 3: Initialize Portfolio
    print("\n3️⃣ Initializing Portfolio...")
    portfolio = PortfolioEngine(
        initial_capital=INITIAL_CAPITAL,
        commission_rate=0.001,
        rebalance_frequency=5
    )
    print("✅ Portfolio engine initialized")
    
    # Step 4: Run Backtest
    print("\n4️⃣ Running Backtest...")
    try:
        results = portfolio.run_backtest(
            strategy, processed_data, BACKTEST_START, BACKTEST_END
        )
        print("✅ Backtest completed successfully")
    except Exception as e:
        print(f"❌ Error during backtest: {e}")
        return
    
    # Step 5: Analyze Performance
    print("\n5️⃣ Analyzing Performance...")
    analytics = PerformanceAnalytics()
    report = analytics.generate_performance_report(results['total_value'])
    
    # Display Results
    print("\n" + "=" * 50)
    print("📊 PERFORMANCE RESULTS")
    print("=" * 50)
    
    print(f"Final Portfolio Value: ${results['total_value'].iloc[-1]:,.2f}")
    print(f"Total Return: {report['total_return']:.2%}")
    print(f"Annual Return: {report['annual_return']:.2%}")
    print(f"Annual Volatility: {report['annual_volatility']:.2%}")
    print(f"Sharpe Ratio: {report['sharpe_ratio']:.3f}")
    print(f"Sortino Ratio: {report['sortino_ratio']:.3f}")
    print(f"Max Drawdown: {report['max_drawdown']:.2%}")
    print(f"Calmar Ratio: {report['calmar_ratio']:.3f}")
    print(f"Win Rate: {report['win_rate']:.2%}")
    
    # Transaction Summary
    transactions = portfolio.get_transaction_history()
    print(f"\nTotal Transactions: {len(transactions)}")
    print(f"Total Commissions: ${transactions['commission'].sum():,.2f}")
    
    # Portfolio Summary
    portfolio_summary = portfolio.get_portfolio_summary()
    print(f"\nFinal Positions: {len(portfolio_summary['position_details'])}")
    print(f"Cash Remaining: ${portfolio_summary['cash']:,.2f}")
    
    print("\n✅ Analysis Complete!")
    print("\n💡 Next Steps:")
    print("- Run notebooks/portfolio_analysis_demo.ipynb for detailed analysis")
    print("- Experiment with different strategies and parameters")
    print("- Add more symbols to the universe")
    print("- Try different rebalancing frequencies")

if __name__ == "__main__":
    main() 