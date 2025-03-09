import streamlit as st
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
from scipy.optimize import minimize
import vectorbt as vbt
from utils.processing import get_stocks
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import utils.plot_utils as pu


def run(symbol_benchmark: str, symbols_date_dict: Dict):
    """Main Streamlit application"""
    if not symbols_date_dict.get('symbols'):
        st.info("Please select symbols.")
        return

    # Load data
    stocks_df = get_stocks(symbols_date_dict, stack=True)
    closes = stocks_df['close']
    volumes = stocks_df['volume']
    benchmark = get_stocks(symbols_date_dict, 'close', benchmark=True)[symbol_benchmark]
    
    window = st.slider("Select lookback period for average volume", 5, 504, 20)
    
    price_ahead = st.slider("Select price ahead", 1, 20, 3)
    
    # price std
    price_std = stocks_df['close'].rolling(window=window).std()
    
    # volume mean
    volume_mean = stocks_df['volume'].rolling(window=window).mean()
    
    # volume std
    volume_std = stocks_df['volume'].rolling(window=window).std()
        
    # price std vs volume std
    price_ahead_3 = closes.shift(-price_ahead)
    price_ahead_mean = price_ahead_3
    price_ahead_change = (price_ahead_mean - closes) / closes
    price_ahead_change_sign = np.sign(price_ahead_change)
    
    # threshold
    price_threshold = st.slider("Select price threshold", -3.0, 3.0, -1.0)
    volume_threshold = st.slider("Select volume threshold", -3.0, 3.0, -1.0)
    
    # normalize price and volume
    price_norm = (closes - closes.rolling(window=window).mean()) / closes.rolling(window=window).std()
    volume_norm = (volumes - volumes.rolling(window=window).mean()) / volumes.rolling(window=window).std()
    
    # calculate EMA
    ema = closes.ewm(span=100, adjust=False).mean()
    
    # Analysis
    price_filter = price_norm < price_threshold
    volume_filter = volume_norm < volume_threshold
    trend_filter = closes > ema
    combined_filter = price_filter & volume_filter & trend_filter
    
    # price_ahead_change analysis
    metrics = {}
    
    signal = price_ahead_change[combined_filter]
    
    daily_return = signal.mean(axis=1)
    
    positive_signal = signal[signal > 0].count(axis=1)
    negative_signal = signal[signal < 0].count(axis=1)
    
    positive_signal_count = positive_signal.sum()
    negative_signal_count = negative_signal.sum()
    total_signal = positive_signal_count + negative_signal_count
    
    metrics['total_signal'] = total_signal
    metrics['positive_signal'] = positive_signal_count
    metrics['negative_signal'] = negative_signal_count
    metrics['accuracy'] = positive_signal_count / total_signal
    metrics['daily_return'] = daily_return.mean()
    avg_return = signal.mean().mean()
    metrics['avg_return'] = avg_return
    
    metrics_df = pd.DataFrame(metrics, index=[0])
    
    # Tickers
    signal_count_by_ticker = combined_filter.sum(axis=0)
    return_by_ticker = signal.mean(axis=0)
    
    ticker_metrics = pd.DataFrame({'signal_count': signal_count_by_ticker, 'return': return_by_ticker})
    
    st.dataframe(ticker_metrics)
    
    st.write(metrics_df)    
    
    # plot signal
    fig = go.Figure()
    fig.add_trace(go.Bar(x=signal.index, y=signal.sum(axis=1)))
    fig.update_layout(title_text='Signal', xaxis_title='Date', yaxis_title='Signal')
    st.plotly_chart(fig)
    
    
    # scatter signal price vs volume color by return
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=price_norm[combined_filter].stack(), y=volume_norm[combined_filter].stack(), mode='markers'))
    fig.update_layout(title_text='Normalized Price vs Normalized Volume', xaxis_title='Normalized Price', yaxis_title='Normalized Volume', coloraxis_colorbar=dict(title="Return"))
    fig.update_traces(marker=dict(color=price_ahead_change_sign[combined_filter].stack(), colorscale='Viridis', colorbar=dict(title="Return")))
    fig.update_traces(marker=dict(colorbar=dict(title="Return"), colorscale='RdYlGn'))
    st.plotly_chart(fig)
    
    
    #=============================
    
    enable_ticker = st.checkbox("Enable ticker")
    
    #=============================
    # Ticker analysis
    #=============================
    
    if not enable_ticker:
        return
    
    selected_ticker = st.selectbox("Select ticker", stocks_df.columns.levels[1])
    
    ticker_price = stocks_df['close'][selected_ticker]
    ticker_volume = stocks_df['volume'][selected_ticker]
    ticker_price_std = price_std[selected_ticker]
    ticker_price_ahead_change = price_ahead_change[selected_ticker]
    ticker_price_norm = price_norm[selected_ticker]
    ticker_volume_norm = volume_norm[selected_ticker]
    
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.02)
    fig.add_trace(go.Scatter(x=ticker_price.index, y=ticker_price, mode='lines', name='Price'), row=1, col=1)
    fig.add_trace(go.Scatter(x=ticker_price_std.index, y=ticker_price_std, mode='lines', name='Price std'), row=2, col=1)
    fig.add_trace(go.Scatter(x=volume_std.index, y=volume_std[selected_ticker], mode='lines', name='Volume std'), row=3, col=1)
    fig.update_layout(title_text=f'{selected_ticker} Price and Volume', xaxis_title='Date', yaxis_title='Price/Volume')
    st.plotly_chart(fig)
    

    # scatter plot normalized price vs normalized volume
    # fig = go.Figure()
    # fig.add_trace(go.Scatter(x=ticker_price_norm, y=ticker_volume_norm, mode='markers'))
    # fig.update_layout(title_text=f'{selected_ticker} Normalized Price vs Normalized Volume', xaxis_title='Normalized Price', yaxis_title='Normalized Volume', coloraxis_colorbar=dict(title="Return"))
    # fig.update_traces(marker=dict(color=ticker_price_ahead_change, colorscale='Viridis', colorbar=dict(title="Return")))
    # fig.update_traces(marker=dict(colorbar=dict(title="Return"), colorscale='RdYlGn'))
    # st.plotly_chart(fig)

    
    # filter
    price_filter = ticker_price_norm < price_threshold
    volume_filter = ticker_volume_norm < volume_threshold
    combined_filter = price_filter & volume_filter 
    
    ticker_ema = ema[selected_ticker]
    
    # trend filter
    trend_filter = ticker_price > ticker_ema
    
    combined_filter = combined_filter & trend_filter
        
    # calculate EMA 10
    ema_10 = ticker_price.ewm(span=10, adjust=False).mean()

    # plot filtered data with EMA
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ticker_price.index, y=ticker_price, mode='lines', name='Price'))
    fig.add_trace(go.Scatter(x=ema.index, y=ticker_ema, mode='lines', name='EMA 100'))
    fig.add_trace(go.Scatter(x=ema_10.index, y=ema_10, mode='lines', name='EMA 10'))
    fig.add_trace(go.Scatter(x=ticker_price[combined_filter].index, y=ticker_price[combined_filter], mode='markers', name='Combined Signal', marker=dict(color='green')))
    fig.update_layout(title_text=f'{selected_ticker} Price, EMA 100, EMA 10 and Combined Signal', xaxis_title='Date', yaxis_title='Price')
    st.plotly_chart(fig)
