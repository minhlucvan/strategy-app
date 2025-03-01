import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
from studies.cash_flow_study import add_price_changes_to_df, calculate_price_changes, filter_prices, plot_correlation_matrix, plot_scatter_matrix, prepare_dims_df
from utils.plot_utils import plot_multi_bar, plot_multi_line, plot_multi_scatter, plot_single_bar, plot_single_line
from utils.processing import get_stocks
import utils.stock_utils as stock_utils
import plotly.graph_objects as go
import talib as ta
from plotly.subplots import make_subplots


def get_stocks_symbols(symbolsDate_dict, symbol):
    symbolsDate_dict_cp = symbolsDate_dict.copy()
    symbolsDate_dict_cp['symbols'] = [symbol]
    
    return symbolsDate_dict_cp

def plot_double_bar(bearish_vix, bullish_vix, benchmark):
    # plot bearish vix and bullish vix
    # bearish vix is red bar
    # bullish vix is green bar
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02)
    fig.add_trace(go.Bar(x=bearish_vix.index, y=bearish_vix, name='Bearish VIX', marker_color='red'), row=1, col=1)
    fig.add_trace(go.Bar(x=bullish_vix.index, y=bullish_vix, name='Bullish VIX', marker_color='green'), row=1, col=1)
    
    # add the price 
    fig.add_trace(go.Scatter(x=benchmark.index, y=benchmark, mode='lines', name='Price'), row=2, col=1)
    
    # hide the legend
    fig.update_layout(barmode='stack', showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

# Calculate 30-day variance by interpolating the two variances,
# depending on the time to expiration of each. Take the square root to get volatility as standard deviation.
# Multiply the volatility (standard deviation) by 100. The result is the VIX index value.
# https://www.macroption.com/vix-calculation/#:~:text=VIX%20Calculation%20Step%20by%20Step,-Select%20the%20options&text=Calculate%2030%2Dday%20variance%20by,is%20the%20VIX%20index%20value.
def calculate_vix_index(prices, window=21):
    # calculate the log returns
    log_returns = np.log(prices / prices.shift(1))
    
    # calculate the variance
    variance = log_returns.rolling(window=window).std() ** 2
    
    # calculate the variance of the variance
    variance_of_variance = variance.rolling(window=2).std()
    
    # calculate the VIX index
    vix_index = 100 * variance_of_variance
    
    return vix_index

def run(symbol_benchmark, symbolsDate_dict):
    symbol_benchmark = 'VN30F1M'
    
    if len(symbolsDate_dict['symbols']) > 0:
        symbol_benchmark = symbolsDate_dict['symbols'][0]
        
    st.write(f"Symbol Benchmark: {symbol_benchmark}")
    
    benchmark_df = get_stocks(get_stocks_symbols(symbolsDate_dict, symbol_benchmark), single=True, timeframe=symbolsDate_dict['timeframe'])
    # st.write(benchmark_df)
    
    # st.write(benchmark_df)
    
    benchmark = benchmark_df['close']
    
    returns = benchmark_df['price_change']
    
    plot_single_line(benchmark, title='Benchmark', x_title='Date', y_title='Price')
    
    vix_index = calculate_vix_index(benchmark)
    
    vix_index = vix_index.rolling(window=5).mean()
    
    plot_single_bar(vix_index, title='VIX Index', x_title='Date', y_title='VIX Index')
    
    # bearish_vix where negative returns
    bearish_vix = vix_index[returns < 0]
    bullish_vix = vix_index[returns > 0]
    
    plot_double_bar(bearish_vix, bullish_vix, benchmark)

    # vix z-score
    vix_z_score = (vix_index - vix_index.mean()) / vix_index.std()
    bearish_vix_z_score = vix_z_score[returns < 0]
    bullish_vix_z_score = vix_z_score[returns > 0]
    
    plot_double_bar(bearish_vix_z_score, bullish_vix_z_score, benchmark)
    
    # ivix = 1/vix
    ivix = 1 / vix_index
    # rolling ivix 10
    # ivix = ivix.rolling(window=10).mean()
    
    bearish_ivix = ivix[returns < 0]
    bullish_ivix = ivix[returns > 0]
    
    plot_double_bar(bearish_ivix, bullish_ivix, benchmark)
    
    atr = ta.ATR(benchmark_df['high'], benchmark_df['low'], benchmark_df['close'], timeperiod=14)
    
    bearish_atr = atr[returns < 0]
    bullish_atr = atr[returns > 0]
    plot_double_bar(bearish_atr, bullish_atr, benchmark)
    
    atr_z_score = (atr - atr.mean()) / atr.std()
    bearish_atr_z_score = atr_z_score[returns < 0]
    bullish_atr_z_score = atr_z_score[returns > 0]
    
    plot_double_bar(bearish_atr_z_score, bullish_atr_z_score, benchmark)