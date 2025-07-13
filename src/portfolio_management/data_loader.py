"""
Data Loading and Preprocessing Module

Handles loading, cleaning, and preprocessing of financial data from various sources.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import List, Dict, Optional, Tuple, Union
from datetime import datetime, timedelta
import logging
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

class DataLoader:
    """
    A comprehensive data loader for financial data with preprocessing capabilities.
    
    Supports loading from:
    - Local CSV files
    - Yahoo Finance API
    - Custom data sources
    """
    
    def __init__(self, data_path: Optional[str] = None):
        """
        Initialize the DataLoader.
        
        Args:
            data_path: Path to local data directory
        """
        self.data_path = Path(data_path) if data_path else Path("data")
        self.logger = self._setup_logger()
        self.data_cache = {}
        
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
    
    def load_csv_data(self, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Load data from local CSV files.
        
        Args:
            symbols: List of stock symbols to load
            
        Returns:
            Dictionary mapping symbols to DataFrames
        """
        data = {}
        
        for symbol in symbols:
            file_path = self.data_path / f"{symbol}.csv"
            
            if file_path.exists():
                try:
                    df = pd.read_csv(file_path)
                    df['Date'] = pd.to_datetime(df['Date'])
                    df.set_index('Date', inplace=True)
                    df.sort_index(inplace=True)
                    
                    # Standardize column names
                    df.columns = df.columns.str.replace(' ', '_').str.lower()
                    if 'adj_close' in df.columns:
                        df['adj_close'] = df['adj_close']
                    elif 'adj.close' in df.columns:
                        df['adj_close'] = df['adj.close']
                        
                    data[symbol] = df
                    self.logger.info(f"Loaded {symbol} data: {len(df)} records")
                    
                except Exception as e:
                    self.logger.error(f"Error loading {symbol}: {str(e)}")
            else:
                self.logger.warning(f"File not found: {file_path}")
                
        return data
    
    def load_yahoo_data(self, symbols: List[str], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """
        Load data from Yahoo Finance API.
        
        Args:
            symbols: List of stock symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            Dictionary mapping symbols to DataFrames
        """
        data = {}
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(start=start_date, end=end_date)
                
                if not df.empty:
                    # Standardize column names
                    df.columns = df.columns.str.replace(' ', '_').str.lower()
                    df.index.name = 'date'
                    
                    data[symbol] = df
                    self.logger.info(f"Downloaded {symbol} data: {len(df)} records")
                else:
                    self.logger.warning(f"No data found for {symbol}")
                    
            except Exception as e:
                self.logger.error(f"Error downloading {symbol}: {str(e)}")
                
        return data
    
    def preprocess_data(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Preprocess and clean the data.
        
        Args:
            data: Dictionary of raw DataFrames
            
        Returns:
            Dictionary of cleaned DataFrames
        """
        processed_data = {}
        
        for symbol, df in data.items():
            try:
                # Create a copy to avoid modifying original data
                processed_df = df.copy()
                
                # Handle missing values
                processed_df = self._handle_missing_values(processed_df)
                
                # Add technical indicators
                processed_df = self._add_technical_indicators(processed_df)
                
                # Add returns
                processed_df = self._add_returns(processed_df)
                
                # Add dividends (if available)
                processed_df = self._calculate_dividends(processed_df)
                
                processed_data[symbol] = processed_df
                self.logger.info(f"Processed {symbol} data successfully")
                
            except Exception as e:
                self.logger.error(f"Error processing {symbol}: {str(e)}")
                
        return processed_data
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        # Forward fill missing values
        df = df.fillna(method='ffill')
        
        # Drop rows with all NaN values
        df = df.dropna(how='all')
        
        return df
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to the dataset."""
        # Use adjusted close price for calculations
        price_col = 'adj_close' if 'adj_close' in df.columns else 'close'
        
        # Simple Moving Averages
        df['sma_5'] = df[price_col].rolling(window=5).mean()
        df['sma_20'] = df[price_col].rolling(window=20).mean()
        df['sma_50'] = df[price_col].rolling(window=50).mean()
        
        # Exponential Moving Averages
        df['ema_12'] = df[price_col].ewm(span=12).mean()
        df['ema_26'] = df[price_col].ewm(span=26).mean()
        
        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # RSI
        df['rsi'] = self._calculate_rsi(df[price_col])
        
        # Bollinger Bands
        df['bb_middle'] = df[price_col].rolling(window=20).mean()
        bb_std = df[price_col].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        
        # Volume indicators
        if 'volume' in df.columns:
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _add_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add various return calculations."""
        price_col = 'adj_close' if 'adj_close' in df.columns else 'close'
        
        # Daily returns
        df['daily_return'] = df[price_col].pct_change()
        
        # Log returns
        df['log_return'] = np.log(df[price_col] / df[price_col].shift(1))
        
        # Multi-period returns
        for period in [5, 10, 20, 60]:
            df[f'return_{period}d'] = df[price_col].pct_change(period)
            
        # Volatility (rolling standard deviation of returns)
        df['volatility_20d'] = df['daily_return'].rolling(window=20).std() * np.sqrt(252)
        
        return df
    
    def _calculate_dividends(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate dividend information from price data."""
        if 'close' in df.columns and 'adj_close' in df.columns:
            # Calculate ratio difference to identify dividend payments
            df['price_ratio'] = df['close'].shift(1) / df['close']
            df['adj_ratio'] = df['adj_close'].shift(1) / df['adj_close']
            df['ratio_diff'] = df['price_ratio'] - df['adj_ratio']
            
            # Estimate dividend amount
            df['dividend_est'] = df['ratio_diff'] * df['close']
            
            # Only keep significant dividend amounts (> $0.01)
            df['dividend_est'] = df['dividend_est'].where(df['dividend_est'] > 0.01, 0)
            
        return df
    
    def create_combined_dataset(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Create a combined dataset with all symbols.
        
        Args:
            data: Dictionary of processed DataFrames
            
        Returns:
            Combined DataFrame with multi-level columns
        """
        combined_data = {}
        
        for symbol, df in data.items():
            # Select key columns for combination
            key_columns = ['close', 'adj_close', 'volume', 'daily_return', 'volatility_20d']
            available_columns = [col for col in key_columns if col in df.columns]
            
            symbol_data = df[available_columns].copy()
            symbol_data.columns = [f"{symbol}_{col}" for col in symbol_data.columns]
            
            combined_data[symbol] = symbol_data
        
        # Combine all data
        combined_df = pd.concat(combined_data.values(), axis=1)
        combined_df = combined_df.sort_index()
        
        return combined_df
    
    def get_data_summary(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Generate a summary of the loaded data.
        
        Args:
            data: Dictionary of DataFrames
            
        Returns:
            Summary DataFrame
        """
        summary_data = []
        
        for symbol, df in data.items():
            summary = {
                'Symbol': symbol,
                'Records': len(df),
                'Start_Date': df.index.min(),
                'End_Date': df.index.max(),
                'Missing_Values': df.isnull().sum().sum(),
                'Columns': len(df.columns)
            }
            
            if 'adj_close' in df.columns:
                summary['Avg_Price'] = df['adj_close'].mean()
                summary['Price_Volatility'] = df['adj_close'].std()
            
            summary_data.append(summary)
        
        return pd.DataFrame(summary_data)
    
    def load_and_process(self, symbols: List[str], source: str = 'csv', 
                        start_date: Optional[str] = None, 
                        end_date: Optional[str] = None) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
        """
        Main method to load and process data.
        
        Args:
            symbols: List of stock symbols
            source: Data source ('csv' or 'yahoo')
            start_date: Start date for yahoo data
            end_date: End date for yahoo data
            
        Returns:
            Tuple of (processed_data_dict, combined_dataframe)
        """
        self.logger.info(f"Loading data for {len(symbols)} symbols from {source}")
        
        # Load raw data
        if source == 'csv':
            raw_data = self.load_csv_data(symbols)
        elif source == 'yahoo':
            if not start_date or not end_date:
                raise ValueError("start_date and end_date required for yahoo data")
            raw_data = self.load_yahoo_data(symbols, start_date, end_date)
        else:
            raise ValueError(f"Unsupported data source: {source}")
        
        # Process data
        processed_data = self.preprocess_data(raw_data)
        
        # Create combined dataset
        combined_df = self.create_combined_dataset(processed_data)
        
        # Generate summary
        summary = self.get_data_summary(processed_data)
        self.logger.info(f"Data loading completed:\n{summary}")
        
        return processed_data, combined_df 