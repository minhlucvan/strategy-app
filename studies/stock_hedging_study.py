import streamlit as st
import pandas as pd
import numpy as np
import talib as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.processing import get_stocks
import utils.stock_utils as su
import utils.plot_utils as pu


def calculate_fear_greed_indicator(benchmark_price, vol_window=20, rsi_window=14, lookback=252):
    """
    Constructs a fear and greed indicator using volatility and RSI.
    """
    daily_returns = benchmark_price.pct_change()
    volatility = daily_returns.rolling(window=vol_window).std()
    min_vol = volatility.rolling(window=lookback).min()
    max_vol = volatility.rolling(window=lookback).max()
    normalized_volatility = 100 * (volatility - min_vol) / (max_vol - min_vol)
    greed_from_vol = 100 - normalized_volatility
    rsi = ta.RSI(benchmark_price, timeperiod=rsi_window)
    return (rsi + greed_from_vol) / 2


def calculate_correlation_and_beta(index_price, benchmark_price):
    """
    Computes the correlation and beta of index price against the benchmark price using returns.
    """
    # Calculate daily returns (percentage change)
    index_returns = index_price.pct_change().dropna()
    benchmark_returns = benchmark_price.pct_change().dropna()
    
    # Concatenate returns into a DataFrame
    corr_df = pd.concat([index_returns, benchmark_returns], axis=1)
    corr_df.columns = ['index', 'benchmark']
    
    # Calculate Pearson correlation
    correlation = corr_df['index'].corr(corr_df['benchmark'])
    
    # Calculate beta: Cov(index, benchmark) / Var(benchmark)
    beta = corr_df['index'].cov(corr_df['benchmark']) / corr_df['benchmark'].var()
    
    return correlation, beta

def calculate_rolling_beta(index_price, benchmark_price, window=252):
    """
    Computes the rolling beta of index price against the benchmark price using returns.
    """
    # Calculate daily returns (percentage change)
    index_returns = index_price.pct_change().dropna()
    benchmark_returns = benchmark_price.pct_change().dropna()
    
    # Concatenate returns into a DataFrame
    corr_df = pd.concat([index_returns, benchmark_returns], axis=1)
    corr_df.columns = ['index', 'benchmark']
    
    # Calculate rolling beta
    rolling_beta = corr_df['index'].rolling(window=window).cov(corr_df['benchmark']) / corr_df['benchmark'].rolling(window=window).var()
    
    return rolling_beta

def calculate_rolling_correlation(index_price, benchmark_price, window=252):
    """
    Computes the rolling correlation of index price against the benchmark price using returns.
    """
    # Calculate daily returns (percentage change)
    index_returns = index_price.pct_change().dropna()
    benchmark_returns = benchmark_price.pct_change().dropna()
    
    # Concatenate returns into a DataFrame
    corr_df = pd.concat([index_returns, benchmark_returns], axis=1)
    corr_df.columns = ['index', 'benchmark']
    
    # Calculate rolling correlation
    rolling_correlation = corr_df['index'].rolling(window=window).corr(corr_df['benchmark'])
    
    return rolling_correlation

def plot_price_and_indicator(price_series, indicator_series, title, indicator_name):
    """
    Creates a dual-plot figure with price and a given indicator.
    """
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02)
    fig.add_trace(go.Scatter(x=price_series.index, y=price_series, name='Price'), row=1, col=1)
    fig.add_trace(go.Scatter(x=indicator_series.index, y=indicator_series, name=indicator_name), row=2, col=1)
    fig.update_layout(title_text=title)
    st.plotly_chart(fig)


def run(symbol_benchmark, symbolsDate_dict):
    """
    Main function to run stock analysis and visualization.
    """
    if not symbolsDate_dict['symbols']:
        st.info("Please select symbols.")
        st.stop()

    prices_df = get_stocks(symbolsDate_dict, 'close')
    benchmark_df = get_stocks(symbolsDate_dict, 'close', benchmark=True)
    index_price = su.construct_index_df(prices_df)
    benchmark_price = benchmark_df[symbol_benchmark] * 100_000
    

    # normalize the prices
    index_price = index_price / index_price.iloc[0]
    benchmark_price = benchmark_price / benchmark_price.iloc[0]
    
    index_over_benchmark = index_price / benchmark_price
    pu.plot_single_line(index_over_benchmark, "Index Over Benchmark", "Date", "Index Over Benchmark")
    
    
    pu.plot_single_line_with_price(index_price, "Stocks Price", "Date", "Price", "Stock", benchmark_price, "VN30")
    
    rolling_correlation = calculate_rolling_correlation(index_price, benchmark_price)
    
    pu.plot_single_line(rolling_correlation, "Rolling Correlation", "Date", "Correlation")
    
    rolling_beta = calculate_rolling_beta(index_price, benchmark_price)
    pu.plot_single_line(rolling_beta, "Rolling Beta", "Date", "Beta")
    
    correlation, beta = calculate_correlation_and_beta(index_price, benchmark_price)
    st.write(f"Correlation: {correlation}")
    st.write(f"Beta: {beta}")
    
    vol_period = st.slider("Volatility Period", 1, 100, 20)
    benchmark_vol = benchmark_price.pct_change().rolling(vol_period).std()
    plot_price_and_indicator(benchmark_price, benchmark_vol, "Benchmark Price and Volatility", "Volatility")
    
    benchmark_rsi = ta.RSI(benchmark_price, timeperiod=14)
    plot_price_and_indicator(benchmark_price, benchmark_rsi, "Benchmark Price and RSI", "RSI")
    
    lookback = st.slider("Lookback Period", 1, 365, 252)
    rsi_window = st.slider("RSI Window", 1, 30, 14)
    fear_greed_index = calculate_fear_greed_indicator(benchmark_price, vol_window=vol_period, rsi_window=rsi_window, lookback=lookback)
    plot_price_and_indicator(benchmark_price, fear_greed_index, "Benchmark Price and Fear and Greed Index", "Fear and Greed Index")
    
    fear_greed_stock = calculate_fear_greed_indicator(index_price, vol_window=vol_period, rsi_window=rsi_window, lookback=lookback)
    plot_price_and_indicator(index_price, fear_greed_stock, "Stocks Price and Fear and Greed Index", "Fear and Greed Index")
    
    # Spread=Price of Stock−(beta x Price of VN30)
    
    spread = index_price - beta * benchmark_price
    
    pu.plot_single_line(spread, "Spread", "Date", "Spread")
    
    # zscorespread=(spread−mean(spread))/std(spread)
    # zscore_spread = (spread - spread.mean()) / spread.std()
    
    # pu.plot_single_line(zscore_spread, "Z-Score Spread", "Date", "Z-Score Spread")