import pandas as pd
import numpy as np
import streamlit as st
import vectorbt as vbt
import talib as ta
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
from utils.plot_utils import plot_multi_line, plot_single_line
from utils.processing import get_stocks

def calculate_volatility(prices, window=21):
    close = prices['close']
    daily_returns = close.pct_change()
    volatility = (daily_returns.rolling(window=window).std() * np.sqrt(252)) * 100
    volatility = volatility - 50
    return volatility

def caculate_momentum(stock_df, window=14):
    """
    Calculate momentum for a given stock.
    """
    # price over average price over the last 14 days
    close = stock_df['close']
    momentum = (close - close.rolling(window=window).mean()) / close.rolling(window=window).mean()
    return momentum

def caculete_liquidity(stock_df, window=14):
    """
    Calculate liquidity for a given stock.
    """
    # liquidity = stock_df['volume'].rolling(window=10).mean(
    volume = stock_df['volume']
    liquidity = (volume - volume.rolling(window=window).mean()) / volume.rolling(window=window).mean()
    return liquidity

def run(symbol_benchmark, symbolsDate_dict):
    if not symbolsDate_dict.get('symbols'):
        st.info("Please select symbols.")
        return

    # Fetch stock & benchmark data
    stock_df = get_stocks(symbolsDate_dict, single=True)
    
    rolling_window = st.slider('Select Rolling Window', 5, 200, 20)
    
    volatility = calculate_volatility(stock_df, window=rolling_window)
    momentum = caculate_momentum(stock_df, window=rolling_window)
    liquidity = caculete_liquidity(stock_df, window=rolling_window)
    
    
    # Plot ATR
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, subplot_titles=("Stock Price", "ATR"))
    fig.add_trace(go.Scatter(x=stock_df.index, y=stock_df['close'], name='Price'), row=1, col=1)
    fig.add_trace(go.Bar(x=volatility.index, y=volatility, name='Volatility'), row=2, col=1)
    fig.add_trace(go.Bar(x=momentum.index, y=momentum, name='Momentum'), row=3, col=1)
    fig.add_trace(go.Bar(x=liquidity.index, y=liquidity, name='Liquidity'), row=4, col=1)
    fig.update_layout(title_text='Stock Price and LVM', height=800)
    st.plotly_chart(fig)
    
    # signal generation
    buy_signal = (volatility > 0)
    sell_signal = (volatility < 0)
    
    buy_signal = buy_signal[buy_signal == True]
    sell_signal = sell_signal[sell_signal == True]
    
    
    # plot signals along with stock price
    fig = make_subplots(rows=1, cols=1, shared_xaxes=True, subplot_titles=("Stock Price", "Buy/Sell Signals"))
    fig.add_trace(go.Scatter(x=stock_df.index, y=stock_df['close'], name='Price'), row=1, col=1)
    fig.add_trace(go.Scatter(x=buy_signal.index, y=stock_df.loc[buy_signal.index, 'close'], mode='markers', marker=dict(color='green', size=10), name='Buy Signal'), row=1, col=1)
    fig.add_trace(go.Scatter(x=sell_signal.index, y=stock_df.loc[sell_signal.index, 'close'], mode='markers', marker=dict(color='red', size=10), name='Sell Signal'), row=1, col=1)
    fig.update_layout(title_text='Stock Price and LVM Signals', height=600)
    st.plotly_chart(fig)