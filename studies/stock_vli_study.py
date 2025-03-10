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
    
    # volume std
    volume_std = stocks_df['volume'].rolling(window=window).std()
        
    # price std vs volume std
    price_ahead_3 = closes.shift(-price_ahead)
    price_ahead_mean = price_ahead_3
    price_ahead_change = (price_ahead_mean - closes) / closes
    price_ahead_change_sign = np.sign(price_ahead_change)
    
    # threshold
    price_threshold = st.slider("Select price threshold", 0.0, 1.0, 0.02, step=0.02)
    volume_threshold = st.slider("Select volume threshold", 0.0, 1.0, 0.02, step=0.02)
    
    # normalize price and volume
    price_norm = price_std / closes.rolling(window=252).mean()
    volume_norm = volume_std / volumes.rolling(window=252).mean()
    
    price_norm_percentile = price_norm.rolling(window=252).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    volume_norm_percentile = volume_norm.rolling(window=252).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    price_norm_filter = price_norm_percentile < price_threshold
    volume_norm_filter = volume_norm_percentile < volume_threshold
    
    # calculate EMA
    ema = closes.ewm(span=20, adjust=False).mean()
    
    ema_filter = closes < ema
    
    # confirmaion sgnal roc(1) > 0.02
    roc = (closes - closes.shift(1)) / closes.shift(1)
    roc_filter = roc > 0.02
    
    # Analysis
    price_filter = price_norm_filter
    volume_filter = volume_norm_filter
    combined_filter_present = price_filter & volume_filter & ema_filter
    
    # shift filter to the previous day
    combined_filter_past = combined_filter_present.shift(1)
    
    # entry signal = past filter & roc filter
    combined_filter = combined_filter_past & roc_filter
    
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
    ticker_volume_std = volume_std[selected_ticker]
    ticker_price_ahead_change = price_ahead_change[selected_ticker]
    ticker_price_norm = price_norm[selected_ticker]
    ticker_volume_norm = volume_norm[selected_ticker]        
        
    # filter norm < 0
    ticker_price_norm_filter = price_filter[selected_ticker]
    ticker_volume_norm_filter = volume_filter[selected_ticker]
    
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.02)
    fig.add_trace(go.Scatter(x=ticker_price.index, y=ticker_price, mode='lines', name='Price'), row=1, col=1)
    fig.add_trace(go.Scatter(x=ticker_price_norm.index, y=ticker_price_norm, mode='lines', name='Price Norm'), row=2, col=1)
    fig.add_trace(go.Scatter(x=ticker_volume_norm.index, y=ticker_volume_norm, mode='lines', name='Volume Norm'), row=3, col=1)
    fig.add_trace(go.Scatter(x=ticker_price_norm[ticker_price_norm_filter].index, y=ticker_price_norm[ticker_price_norm_filter], mode='markers', name='Price Norm Filter', marker=dict(color='green')), row=2, col=1)
    fig.add_trace(go.Scatter(x=ticker_volume_norm[ticker_volume_norm_filter].index, y=ticker_volume_norm[ticker_volume_norm_filter], mode='markers', name='Volume Norm Filter', marker=dict(color='green')), row=3, col=1)
    fig.update_layout(title_text=f'{selected_ticker} Price and Volume Normalized', xaxis_title='Date', yaxis_title='Price/Volume')
    st.plotly_chart(fig)

    # =============================
    # Ticker analysis
    # =============================
    
    # trend filter    
    ticker_combined_filter = combined_filter[selected_ticker]
        
    # calculate EMA 10
    ema_10 = ticker_price.ewm(span=10, adjust=False).mean()

    # plot filtered data with EMA
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ticker_price.index, y=ticker_price, mode='lines', name='Price'))
    fig.add_trace(go.Scatter(x=ema_10.index, y=ema_10, mode='lines', name='EMA 10'))
    fig.add_trace(go.Scatter(x=ticker_price[ticker_combined_filter].index, y=ticker_price[ticker_combined_filter], mode='markers', name='Combined Signal', marker=dict(color='green')))
    fig.update_layout(title_text=f'{selected_ticker} Price, EMA 100, EMA 10 and Combined Signal', xaxis_title='Date', yaxis_title='Price')
    st.plotly_chart(fig)

    show_raw = st.checkbox("Show raw price and volume")
    if show_raw:
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.02)
        fig.add_trace(go.Scatter(x=ticker_price.index, y=ticker_price, mode='lines', name='Price'), row=1, col=1)
        fig.add_trace(go.Scatter(x=ticker_price_std.index, y=ticker_price_std, mode='lines', name='Price std'), row=2, col=1)
        fig.add_trace(go.Scatter(x=volume_std.index, y=volume_std[selected_ticker], mode='lines', name='Volume std'), row=3, col=1)
        fig.update_layout(title_text=f'{selected_ticker} Price and Volume', xaxis_title='Date', yaxis_title='Price/Volume')
        st.plotly_chart(fig)
    
    show_norm = st.checkbox("Show normalized price and volume")
    if show_norm:    
        # Normalized price and volume
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.02)
        fig.add_trace(go.Scatter(x=ticker_price.index, y=ticker_price, mode='lines', name='Price'), row=1, col=1)
        fig.add_trace(go.Scatter(x=ticker_price_norm.index, y=ticker_price_norm, mode='lines', name='Price Norm'), row=2, col=1)
        fig.add_trace(go.Scatter(x=ticker_volume_norm.index, y=ticker_volume_norm, mode='lines', name='Volume Norm'), row=3, col=1)
        fig.update_layout(title_text=f'{selected_ticker} Price and Volume Normalized', xaxis_title='Date', yaxis_title='Price/Volume')
        st.plotly_chart(fig)