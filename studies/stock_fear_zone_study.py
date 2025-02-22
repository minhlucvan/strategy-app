import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import talib as ta
from utils.plot_utils import plot_multi_bar, plot_multi_line, plot_multi_scatter, plot_single_bar, plot_single_line
from utils.processing import get_stocks

def get_stocks_symbols(symbolsDate_dict, symbol):
    symbolsDate_dict_cp = symbolsDate_dict.copy()
    symbolsDate_dict_cp['symbols'] = [symbol]
    return symbolsDate_dict_cp

def calculate_moving_average(series, period, ma_type="WMA"):
    if ma_type == "SMA":
        return ta.SMA(series, timeperiod=period)
    elif ma_type == "EMA":
        return ta.EMA(series, timeperiod=period)
    elif ma_type == "WMA":
        return ta.WMA(series, timeperiod=period)
    elif ma_type == "HMA":
        # HMA is not directly available in TA-Lib, using WMA as fallback
        return ta.WMA(series, timeperiod=period)
    elif ma_type == "RMA":
        return ta.RMA(series, timeperiod=period)
    return ta.WMA(series)

def calculate_fz1(df, source_col, high_period, stdev_period, ma_type):
    source = df[source_col]
    highest_high = source.rolling(window=high_period).max()
    fz1 = (highest_high - source) / highest_high
    avg1 = calculate_moving_average(fz1, stdev_period, ma_type)
    stdev1 = fz1.rolling(window=stdev_period).std()
    fz1_limit = avg1 + stdev1
    return fz1, fz1_limit

def calculate_fz2(df, source_col, high_period, stdev_period, ma_type):
    source = df[source_col]
    fz2 = calculate_moving_average(source, high_period, ma_type)
    avg2 = calculate_moving_average(fz2, stdev_period, ma_type)
    stdev2 = fz2.rolling(window=stdev_period).std()
    fz2_limit = avg2 - stdev2
    return fz2, fz2_limit

def calculate_greed(df, source_col, high_period, stdev_period, ma_type):
    source = df[source_col]
    highest = source.rolling(window=high_period).max()
    lowest = source.rolling(window=high_period).min()
    gz1 = (source - lowest) / (highest - lowest)
    avg_gz1 = calculate_moving_average(gz1, stdev_period, ma_type)
    stdev_gz1 = gz1.rolling(window=stdev_period).std()
    gz1_limit = avg_gz1 + stdev_gz1
    return gz1, gz1_limit

def plot_zones(df, fearzone_open, fearzone_close, greedzone_open, greedzone_close, benchmark):
    fig = make_subplots(rows=1, cols=1)
    
    # Add benchmark price
    fig.add_trace(go.Scatter(x=df.index, y=benchmark, mode='lines', name='Price', line=dict(color='gray')))
    
    # Add FearZone candlesticks
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=fearzone_open,
        high=fearzone_open,
        low=fearzone_close,
        close=fearzone_close,
        name='FearZone',
        increasing_line_color='#FC6C85',
        decreasing_line_color='#FC6C85'
    ))
    
    # Add GreedZone candlesticks
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=greedzone_open,
        high=greedzone_open,
        low=greedzone_close,
        close=greedzone_close,
        name='GreedZone',
        increasing_line_color='#32CD32',
        decreasing_line_color='#32CD32'
    ))
    
    fig.update_layout(
        title='Fear & Greed Zones with Price',
        yaxis_title='Price',
        showlegend=True
    )
    st.plotly_chart(fig, use_container_width=True)

def run(symbol_benchmark, symbolsDate_dict):
    symbol_benchmark = symbolsDate_dict['symbols'][0] if len(symbolsDate_dict['symbols']) > 0 else 'VN30F1M'
    
    if len(symbolsDate_dict['symbols']) > 0:
        symbol_benchmark = symbolsDate_dict['symbols'][0]
        
    st.write(f"Symbol Benchmark: {symbol_benchmark}")
    
    # Settings (matching Pine Script inputs)
    high_period = 30
    stdev_period = 50
    ma_type = "WMA"
    source_col = 'ohlc4'  # Calculated as (open + high + low + close)/4
    
    # Get data
    df = get_stocks(get_stocks_symbols(symbolsDate_dict, symbol_benchmark), 
                   single=True, 
                   timeframe=symbolsDate_dict['timeframe'])
    
    # Calculate OHLC4
    df[source_col] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    benchmark = df['close']
    
    # Calculate True Range
    df['tr'] = ta.TRANGE(df['high'], df['low'], df['close'])
    
    # Calculate Zones
    fz1, fz1_limit = calculate_fz1(df, source_col, high_period, stdev_period, ma_type)
    fz2, fz2_limit = calculate_fz2(df, source_col, high_period, stdev_period, ma_type)
    gz1, gz1_limit = calculate_greed(df, source_col, high_period, stdev_period, ma_type)
    
    # Zone Conditions
    fearzone_con = (fz1 > fz1_limit) & (fz2 < fz2_limit)
    greedzone_con = gz1 > gz1_limit
    
    # Calculate Zone Levels
    fearzone_open = np.where(fearzone_con, df['low'] - df['tr'], np.nan)
    fearzone_close = np.where(fearzone_con, df['low'] - 2 * df['tr'], np.nan)
    greedzone_open = np.where(greedzone_con, df['high'] + df['tr'], np.nan)
    greedzone_close = np.where(greedzone_con, df['high'] + 2 * df['tr'], np.nan)
    
    # Plot results
    plot_zones(df, fearzone_open, fearzone_close, greedzone_open, greedzone_close, benchmark)
    
    # Plot benchmark separately
    # plot_single_line(benchmark, title='Benchmark Price', x_title='Date', y_title='Price')

if __name__ == "__main__":
    # Example usage
    symbols_date_dict = {
        'symbols': ['VN30F1M'],
        'timeframe': '1D'
    }
    run('VN30F1M', symbols_date_dict)