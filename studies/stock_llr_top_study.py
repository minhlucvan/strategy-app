import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, Dict, List
from utils.processing import get_stocks
import vectorbt as vbt

from .quant_portfolio import CONFIG, Trade, FactorStrategy, PerformanceMetrics, create_visualizations, create_signal_visualizations

class LLRStrategy(FactorStrategy):
    def __init__(self, prices: pd.DataFrame, volumes: pd.DataFrame, opens: pd.DataFrame, highs: pd.DataFrame, lows: pd.DataFrame, config: Dict = CONFIG):
        super().__init__(prices, config)
        self.volumes = volumes.fillna(method='ffill')
        self.opens = opens.fillna(method='ffill')
        self.highs = highs.fillna(method='ffill')
        self.lows = lows.fillna(method='ffill')
        self.prices = prices.fillna(method='ffill')
        
    def calculate_llr(self, date: pd.Timestamp) -> pd.Series:
        # Fixed window for simplicity
        window = self.config.get('llr_window', 32)
        price_change = self.prices.loc[date] - self.prices.shift(1).loc[date]
        sign_change = np.sign(price_change)
        current_vol = self.volumes.loc[date]
        avg_vol = self.volumes.rolling(window=window).mean().loc[date]
        return (current_vol * sign_change) / avg_vol
    
    def calculate_adjusted_signal(self, date: pd.Timestamp) -> pd.Series:
        llr = self.calculate_llr(date)
        # Simplified volatility adjustment using daily range
        daily_range = (self.highs.loc[date] - self.lows.loc[date]) / self.prices.loc[date]
        # Apply threshold and non-linear transformation
        vol_factor = np.log1p(daily_range.where(daily_range > 0.02, 0))  # Only amplify if range > 2%
        adjusted_signal = llr * vol_factor
        # Filter for minimum signal strength
        return llr
    
    def select_assets(self, date: pd.Timestamp) -> List[str]:
        adjusted_signals = self.calculate_adjusted_signal(date)
        top_n = self.config.get('top_n', 3)
        return adjusted_signals.nlargest(top_n).index.tolist(), adjusted_signals

def run(symbol_benchmark: str, symbols_date_dict: Dict, strategy_type: str = "LLR", 
           extra_data: pd.DataFrame = None):
    if not symbols_date_dict.get('symbols'):
        st.info("Please select symbols.")
        return

    stocks_df = get_stocks(symbols_date_dict, stack=True)
    prices = stocks_df['close']
    volumes = stocks_df['volume']
    highs = stocks_df['high']
    lows = stocks_df['low']
    opens = stocks_df['open']
    
    benchmark = get_stocks(symbols_date_dict, 'close', benchmark=True)[symbol_benchmark]
    col1, col2 = st.columns(2)
    
    with col1:
        llr_window = st.slider("Select window for average volume calculation", 5, 504, 32)
        holding_period = st.slider("Select holding period", 1, 10, 2)
        top_n = st.slider("Select number of top assets", 1, 10, 3)
    
    with col2:
        atr_window = st.slider("Select ATR window", 5, 504, 50)
        stop_loss = st.slider("Select stop loss", 0.0, 0.1, 0.00)
        take_profit = st.slider("Select take profit", 0.0, 0.5, 0.00)
    # Configuration
    CONFIG = {
        'risk_free_rate': 0.045,
        'transaction_cost': 0.002,
        'freq': 252,
        'llr_window' : llr_window,
        'atr_window': atr_window,
        'holding_period': holding_period,
        'top_n': top_n,
        'stop_loss': stop_loss,
        'take_profit': take_profit
    }
    
    # Strategy selection unchanged
    strategy = LLRStrategy(prices, volumes, opens, highs, lows, CONFIG)
    
    portfolio_returns, trade_log = strategy.execute()
    
    benchmark_returns = (benchmark.pct_change() + 1)[portfolio_returns.index[0]:]
    
    metrics = PerformanceMetrics()
    cum_returns = pd.DataFrame({
        'Strategy': metrics.cumulative_returns(portfolio_returns),
        'Benchmark': metrics.cumulative_returns(benchmark_returns)
    }).fillna(method='ffill')
    
    cols = st.columns(4)
    for col, (label, value) in zip(cols, [
        (f"{strategy_type.capitalize()} Return", cum_returns['Strategy'].iloc[-1]),
        ("Benchmark Return", cum_returns['Benchmark'].iloc[-1]),
        ("Sharpe Ratio", metrics.sharpe_ratio(portfolio_returns, CONFIG)),
        ("Win Rate", metrics.win_rate(portfolio_returns))
    ]):
        col.metric(label, f"{value:.2%}" if 'Ratio' not in label else f"{value:.2f}")
    
    st.plotly_chart(px.line(cum_returns, labels={'value': 'Cumulative Return', 'variable': ''}))
    st.plotly_chart(create_visualizations(trade_log, 'Returns Over Time', 'exit_date', 'net_return', 'ticker'))
    st.write("### Return Distribution")
    st.plotly_chart(create_visualizations(trade_log, '', 'net_return', ''))
    
    # scatter returns vs signals
    st.plotly_chart(create_signal_visualizations(trade_log, 'Returns vs Signals', 'net_return', 'signal', 'ticker'))

    # Rest of the function remains largely unchanged
    if st.checkbox("Show Current Portfolio"):
        st.subheader("Current Portfolio")
        latest_assets = strategy.select_assets(prices.index[-2])
        current_data = [{
            'Ticker': ticker,
            'Entry': prices.loc[prices.index[-2], ticker],
            'Current': prices.loc[prices.index[-1], ticker],
            'Return': (prices.loc[prices.index[-1], ticker] / prices.loc[prices.index[-2], ticker]) - 1
        } for ticker in latest_assets]
        st.write(pd.DataFrame(current_data))

    if st.checkbox("Show Trade Log"):
        st.subheader("Trade History")
        display_log = trade_log.copy()
        display_log['net_return'] = display_log['net_return'].apply(lambda x: f"{x:.2%}")
        st.dataframe(display_log.rename(columns={'net_return': 'Return'}))

    if st.checkbox("Show Statistics"):
        st.subheader("Strategy Statistics")
        stats = pd.DataFrame({
            'Metric': ['Total Trades', 'Avg Return', 'Total Return', 'Sharpe', 'Win Rate'],
            'Value': [
                len(trade_log),
                trade_log['net_return'].mean(),
                cum_returns['Strategy'].iloc[-1],
                metrics.sharpe_ratio(portfolio_returns, CONFIG),
                metrics.win_rate(portfolio_returns)
            ]
        })
        st.dataframe(stats)
