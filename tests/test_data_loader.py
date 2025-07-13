"""
Unit tests for DataLoader module.
"""

import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from portfolio_management import DataLoader

class TestDataLoader(unittest.TestCase):
    """Test cases for DataLoader class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.data_loader = DataLoader()
        
        # Create sample data
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        self.sample_data = pd.DataFrame({
            'Date': dates,
            'Open': np.random.uniform(100, 200, 100),
            'High': np.random.uniform(100, 200, 100),
            'Low': np.random.uniform(100, 200, 100),
            'Close': np.random.uniform(100, 200, 100),
            'Adj Close': np.random.uniform(100, 200, 100),
            'Volume': np.random.randint(1000000, 10000000, 100)
        })
    
    def test_initialization(self):
        """Test DataLoader initialization."""
        loader = DataLoader()
        self.assertIsNotNone(loader.logger)
        self.assertEqual(str(loader.data_path), 'data')
        
        # Test with custom path
        loader_custom = DataLoader(data_path='custom_path')
        self.assertEqual(str(loader_custom.data_path), 'custom_path')
    
    def test_handle_missing_values(self):
        """Test missing value handling."""
        # Create data with missing values
        data_with_na = self.sample_data.copy()
        data_with_na.loc[10:15, 'Close'] = np.nan
        
        cleaned_data = self.data_loader._handle_missing_values(data_with_na)
        
        # Should forward fill missing values
        self.assertFalse(cleaned_data['Close'].isna().any())
    
    def test_add_technical_indicators(self):
        """Test technical indicator calculation."""
        # Prepare data
        data = self.sample_data.copy()
        data.set_index('Date', inplace=True)
        data.columns = data.columns.str.replace(' ', '_').str.lower()
        
        # Add technical indicators
        result = self.data_loader._add_technical_indicators(data)
        
        # Check if indicators are added
        expected_indicators = ['sma_5', 'sma_20', 'ema_12', 'ema_26', 'macd', 'rsi']
        for indicator in expected_indicators:
            self.assertIn(indicator, result.columns)
    
    def test_add_returns(self):
        """Test return calculations."""
        data = self.sample_data.copy()
        data.set_index('Date', inplace=True)
        data.columns = data.columns.str.replace(' ', '_').str.lower()
        
        result = self.data_loader._add_returns(data)
        
        # Check if return columns are added
        expected_returns = ['daily_return', 'log_return', 'return_5d', 'volatility_20d']
        for return_col in expected_returns:
            self.assertIn(return_col, result.columns)
    
    def test_calculate_rsi(self):
        """Test RSI calculation."""
        prices = pd.Series(np.random.uniform(100, 200, 50))
        rsi = self.data_loader._calculate_rsi(prices)
        
        # RSI should be between 0 and 100
        self.assertTrue((rsi >= 0).all())
        self.assertTrue((rsi <= 100).all())
    
    def test_calculate_dividends(self):
        """Test dividend calculation."""
        data = self.sample_data.copy()
        data.set_index('Date', inplace=True)
        data.columns = data.columns.str.replace(' ', '_').str.lower()
        
        result = self.data_loader._calculate_dividends(data)
        
        # Should have dividend estimation columns
        self.assertIn('dividend_est', result.columns)
    
    def test_get_data_summary(self):
        """Test data summary generation."""
        # Create mock processed data
        mock_data = {
            'AAPL': self.sample_data.set_index('Date'),
            'MSFT': self.sample_data.set_index('Date')
        }
        
        summary = self.data_loader.get_data_summary(mock_data)
        
        # Check summary structure
        self.assertIsInstance(summary, pd.DataFrame)
        self.assertEqual(len(summary), 2)  # Two symbols
        expected_columns = ['Symbol', 'Records', 'Start_Date', 'End_Date', 'Missing_Values', 'Columns']
        for col in expected_columns:
            self.assertIn(col, summary.columns)

if __name__ == '__main__':
    unittest.main() 