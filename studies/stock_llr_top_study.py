import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, Dict, List
from utils.processing import get_stocks

from .quant_portfolio import CONFIG, Trade, FactorStrategy, PerformanceMetrics, create_visualizations

class LLRStrategy(FactorStrategy):
    def __init__(self, prices: pd.DataFrame, volumes: pd.DataFrame, config: Dict = CONFIG):
        super().__init__(prices, config)
        self.volumes = volumes.fillna(method='ffill')
        
    def calculate_llr(self, date: pd.Timestamp) -> pd.Series:
        """Calculate LLR = (Volume × Sign(Price Change)) / Avg Volume over Window"""
        # Price change sign
        price_change = self.prices.loc[date] - self.prices.shift(1).loc[date]
        sign_change = np.sign(price_change)
        
        # Current volume
        current_vol = self.volumes.loc[date]
        
        # Average volume over window
        window_start = max(0, self.prices.index.get_loc(date) - self.config['llr_window'] + 1)
        avg_vol = self.volumes.iloc[window_start:self.prices.index.get_loc(date) + 1].mean()
        
        # LLR calculation
        llr = (current_vol * sign_change) / avg_vol
        return llr
    
    def select_assets(self, date: pd.Timestamp) -> List[str]:
        """Select the stock with the highest LLR"""
        llr_scores = self.calculate_llr(date)
        return [llr_scores.nlargest(1).index[0]]


def run(symbol_benchmark: str, symbols_date_dict: Dict, strategy_type: str = "LLR", 
           extra_data: pd.DataFrame = None):
    if not symbols_date_dict.get('symbols'):
        st.info("Please select symbols.")
        return

    prices = get_stocks(symbols_date_dict, 'close')
    volumes = get_stocks(symbols_date_dict, 'volume')
    benchmark = get_stocks(symbols_date_dict, 'close', benchmark=True)[symbol_benchmark]
    
    llr_window = st.slider("Select window for average volume calculation", 5, 504, 20)
    
    # Configuration
    CONFIG = {
        'risk_free_rate': 0.045,
        'transaction_cost': 0.002,
        'freq': 52,
        'llr_window' : llr_window
    }
    
    # Strategy selection unchanged
    strategy = LLRStrategy(prices, volumes, CONFIG)
    
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
