import streamlit as st
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
from scipy.optimize import minimize
import vectorbt as vbt
from utils.processing import get_stocks

@dataclass
class IndicatorParams:
    """Data class for storing indicator parameters"""
    window: int = 20
    vol_window: int = 50
    lookback: int = 5
    price_window: int = 10
    epsilon: float = 0.01

class MarketIndicators:
    """Class to encapsulate market indicator calculations"""
    
    @staticmethod
    def price_range(prices: pd.Series, window: int) -> pd.Series:
        """Calculate normalized price range"""
        rolling_high = prices.rolling(window=window).max()
        rolling_low = prices.rolling(window=window).min()
        return (rolling_high - rolling_low) / rolling_low

    @staticmethod
    def vtp(prices: pd.Series, volumes: pd.Series, params: IndicatorParams) -> pd.Series:
        """Calculate Volume Trend Persistence"""
        ma_volume = volumes.rolling(window=params.window).mean()
        vol_volume = volumes.rolling(window=params.vol_window).std()
        return (volumes - ma_volume) / vol_volume

    @staticmethod
    def rsr(df: pd.DataFrame, window: int) -> pd.Series:
        """Calculate Range Stability Ratio"""
        rolling_range = (df['high'] - df['low']).rolling(window=window).mean()
        current_range = df['high'] - df['low']
        return rolling_range / current_range

    @staticmethod
    def vpd(df: pd.DataFrame, volumes: pd.Series, params: IndicatorParams) -> pd.Series:
        """Calculate Volume-Price Divergence"""
        ma_volume = volumes.rolling(window=params.window).mean()
        ma_price = (df['close'] - df['open']).abs().rolling(window=params.price_window).mean()
        return ((volumes / ma_volume) - 1) / ((df['close'] - df['open']).abs() / ma_price + params.epsilon)

    @staticmethod
    def cpd(df: pd.DataFrame, params: IndicatorParams) -> pd.Series:
        """Calculate Closing Price Drift"""
        price_change = df['close'].diff()
        vol_price = df['close'].rolling(window=params.vol_window).std()
        return price_change.rolling(window=params.lookback).sum() / vol_price

    @staticmethod
    def vap(df: pd.DataFrame, volumes: pd.Series, params: IndicatorParams) -> pd.Series:
        """Calculate Volume Acceleration Proxy"""
        ma_volume = volumes.rolling(window=params.window).mean()
        return (volumes - volumes.shift(1)) / ma_volume

class TradingStrategy:
    """Class to handle trading strategy logic and optimization"""
    
    def __init__(self, df: pd.DataFrame, benchmark: pd.Series):
        self.df = df
        self.benchmark = benchmark
        self.indicators = MarketIndicators()
        self.params = IndicatorParams()

    def calculate_metrics(self) -> Dict[str, pd.Series]:
        """Calculate all metrics"""
        closes = self.df['close']
        volumes = self.df['volume']
        
        vtp = self.indicators.vtp(closes, volumes, self.params)
        price_range = self.indicators.price_range(closes, self.params.window)
        
        return {
            'VTP Ratio': vtp / price_range,
            'RSR Ratio': self.indicators.rsr(self.df, self.params.window),
            'VAP Ratio': self.indicators.vap(self.df, volumes, self.params) / price_range
        }

    def calculate_returns(self, projection_period: int) -> pd.Series:
        """Calculate forward returns"""
        closes = self.df['close']
        close_ahead = closes.shift(-projection_period)
        return (close_ahead - closes) / closes

    def evaluate_signals(self, metrics: Dict[str, pd.Series], thresholds: Dict[str, float]) -> pd.DataFrame:
        """Evaluate signal performance"""
        projected_returns = self.calculate_returns(10)
        
        combined_signal = (
            (metrics['VTP Ratio'] > thresholds['vtp']) &
            (metrics['RSR Ratio'] > thresholds['rsr']) &
            (metrics['VAP Ratio'] > thresholds['vap'])
        )
        
        results = {}
        for name, signal in {**metrics, 'Combined Signal': combined_signal}.items():
            signal_mask = signal > 0
            total_signals = signal_mask.sum()
            if total_signals.empty:
                continue
                
            positive_returns = projected_returns[signal_mask]
            positive_count = (positive_returns > 0).sum()
            
            results[name] = {
                'accuracy': (positive_count / total_signals).mean(axis=0),
                'sharp_ratio': (positive_returns.mean() / positive_returns.std()).mean(axis=0),
                'avg_return': positive_returns.mean().mean(axis=0),
                'total_signals': total_signals.mean(axis=0)
            }
        
        return pd.DataFrame(results).T

    def optimize_parameters(self, initial_params: Dict[str, float]) -> Dict[str, float]:
        """Optimize thresholds and window using Sharpe ratio maximization"""
        def objective(params):
            thresholds = {
                'vtp': params[0],
                'rsr': params[1],
                'vap': params[2]
            }
            self.params.window = int(params[3])
            
            metrics = self.calculate_metrics()
            results = self.evaluate_signals(metrics, thresholds)
            
            # Negative Sharpe ratio for minimization
            return -results.loc['Combined Signal', 'sharp_ratio'] if 'Combined Signal' in results.index else -np.inf

        bounds = [(0, 5), (0, 5), (0, 5), (5, 50)]  # Thresholds and window bounds
        result = minimize(
            objective,
            [initial_params['vtp'], initial_params['rsr'], initial_params['vap'], self.params.window],
            method='L-BFGS-B',
            bounds=bounds
        )
        
        return {
            'vtp': result.x[0],
            'rsr': result.x[1],
            'vap': result.x[2],
            'window': int(result.x[3])
        }

def run(symbol_benchmark: str, symbols_date_dict: Dict):
    """Main Streamlit application"""
    if not symbols_date_dict.get('symbols'):
        st.info("Please select symbols.")
        return

    # Load data
    stocks_df = get_stocks(symbols_date_dict, stack=True)
    benchmark = get_stocks(symbols_date_dict, 'close', benchmark=True)[symbol_benchmark]
    
    # Initialize strategy
    strategy = TradingStrategy(stocks_df, benchmark)
    
    # UI controls
    window = st.slider("Window", 5, 50, 20)
    strategy.params.window = window
    
    thresholds = {
        'vtp': st.slider("VTP Threshold", 0.0, 5.0, 3.0),
        'rsr': st.slider("RSR Threshold", 0.0, 5.0, 3.0),
        'vap': st.slider("VAP Threshold", 0.0, 5.0, 3.0)
    }
    
    if st.button("Optimize Parameters"):
        optimized = strategy.optimize_parameters(thresholds)
        st.write("Optimized Parameters:", optimized)
        thresholds = {k: v for k, v in optimized.items() if k != 'window'}
        strategy.params.window = optimized['window']

    # Calculate and display results
    metrics = strategy.calculate_metrics()
    results_df = strategy.evaluate_signals(metrics, thresholds)
    
    st.write("### Results")
    st.write(results_df)
