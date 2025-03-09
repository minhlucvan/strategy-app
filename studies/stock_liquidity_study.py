import streamlit as st
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
import plotly.express as px
from plotly import graph_objects as go
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, Dict, List
from utils.processing import get_stocks
import vectorbt as vb

def caulate_price_range(prices: pd.DataFrame, window: int = 20) -> pd.Series:
    rolling_high = prices.rolling(window=window).max()
    rolling_low = prices.rolling(window=window).min()
    return (rolling_high - rolling_low) / rolling_low

# #### 1. Volume Trend Persistence (VTP)
# - **Definition**: Cumulative deviation of daily volume from its moving average, normalized by historical volatility.
# - **Formula**:  
#   \[
#   VTP_t = \frac{\sum_{i=t-n}^{t} (Volume_i - MA_n(Volume))}{\sigma_{Volume}}
#   \]
#   - \(MA_n(Volume)\): n-day moving average of volume (e.g., 20 days).
#   - \(\sigma_{Volume}\): Std dev of volume over a longer lookback (e.g., 50 days).
#   - \(n\): Lookback period (e.g., 5 days).
# - **Intuition**: Institutions accumulating over days push volume above average consistently. A rising VTP with stable prices suggests they’re loading up quietly.
# - **Threshold**: VTP > 1 AND daily price change < 1% over the period.
def calculate_vtp(prices: pd.Series, volumes: pd.Series, window: int = 20, vol_window: int = 50, lookback: int = 5) -> pd.Series:
    # Calculate moving average and volatility
    ma_volume = volumes.rolling(window=window).mean()
    vol_volume = volumes.rolling(window=vol_window).std()
    
    # Calculate VTP
    vtp = (volumes - ma_volume) / vol_volume
    return vtp


# #### 2. Range Stability Ratio (RSR)
# - **Definition**: Ratio of the current daily range to its recent average, inverted to emphasize compression.
# - **Formula**:  
#   \[
#   RSR_t = \frac{\frac{1}{m} \sum_{i=t-m}^{t-1} (High_i - Low_i)}{High_t - Low_t}
#   \]
#   - \(High_t - Low_t\): Current day’s range.
#   - \(m\): Lookback (e.g., 10 days).
# - **Intuition**: Tight ranges (high RSR) mean someone’s absorbing supply without letting price escape. Institutions love this during buildup phases.
# - **Threshold**: RSR > 1.5 AND volume > 50-day median.
def calculate_rsr(prices: pd.DataFrame, window: int = 10) -> pd.Series:
    rolling_range = (prices['high'] - prices['low']).rolling(window=window).mean()
    current_range = prices['high'] - prices['low']
    return rolling_range / current_range

# #### 3. Volume-Price Divergence (VPD)
# - **Definition**: Discrepancy between volume growth and price movement.
# - **Formula**:  
#   \[
#   VPD_t = \frac{\frac{Volume_t}{MA_n(Volume)} - 1}{\frac{|Close_t - Open_t|}{MA_m(Close - Open)} + \epsilon}
#   \]
#   - \(MA_n(Volume)\): 20-day volume MA.
#   - \(MA_m(Close - Open)\): 10-day MA of absolute open-to-close change.
#   - \(\epsilon\): Small constant (e.g., 0.01) to avoid division by zero.
# - **Intuition**: High volume with low price movement (big VPD) screams institutional absorption—retail would’ve moved the needle more.
# - **Threshold**: VPD > 2 AND price change < 0.5%.
def calculate_vpd(prices: pd.DataFrame, volumes: pd.DataFrame, window: int = 20, price_window: int = 10, epsilon: float = 0.01) -> pd.Series:
    ma_volume = volumes.rolling(window=window).mean()
    ma_price = (prices['close'] - prices['open']).abs().rolling(window=price_window).mean()
    
    vpd = ((volumes / ma_volume) - 1) / ((prices['close'] - prices['open']).abs() / ma_price + epsilon)
    return vpd

# #### 4. Closing Price Drift (CPD)
# - **Definition**: Cumulative small upward drift in closing prices, normalized by volatility.
# - **Formula**:  
#   \[
#   CPD_t = \frac{\sum_{i=t-k}^{t} (Close_i - Close_{i-1})}{\sigma_{Close}}
#   \]
#   - \(k\): Short lookback (e.g., 5 days).
#   - \(\sigma_{Close}\): Std dev of closes over 20 days.
# - **Intuition**: Institutions nearing the end of accumulation might let prices creep up slightly. Positive CPD with low volatility hints at their presence.
# - **Threshold**: CPD > 0.75 AND daily range < 50-day average.
def calculate_cpd(prices: pd.DataFrame, window: int = 5, vol_window: int = 20) -> pd.Series:
    price_change = prices['close'].diff()
    vol_price = prices['close'].rolling(window=vol_window).std()
    
    cpd = price_change.rolling(window=window).sum() / vol_price
    return cpd

# #### 5. Volume Acceleration Proxy (VAP)
# - **Definition**: Change in volume relative to a longer-term trend.
# - **Formula**:  
#   \[
#   VAP_t = \frac{Volume_t - Volume_{t-1}}{MA_n(Volume)}
#   \]
#   - \(MA_n(Volume)\): 20-day volume MA.
# - **Intuition**: A sudden volume jump (high VAP) without a big price move could mean institutions are ramping up before a breakout.
# - **Threshold**: VAP > 1 AND price change < 1%.
def calculate_vap(prices: pd.DataFrame, volumes: pd.DataFrame, window: int = 20) -> pd.Series:
    ma_volume = volumes.rolling(window=window).mean()
    vap = (volumes - volumes.shift(1)) / ma_volume
    return vap

def plot_price_and_signal(prices: pd.DataFrame, signal: pd.Series, symbol: str):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("Price", "Signal"))
    fig.add_trace(go.Scatter
                  (x=prices.index, y=prices[symbol], name=symbol), row=1, col=1)
    fig.add_trace(go.Scatter(x=signal.index, y=signal[symbol], name=symbol), row=2, col=1)
    st.plotly_chart(fig)

def run(symbol_benchmark: str, symbols_date_dict: Dict, strategy_type: str = "LLR", 
           extra_data: pd.DataFrame = None):
    if not symbols_date_dict.get('symbols'):
        st.info("Please select symbols.")
        return

    stocks_df = get_stocks(symbols_date_dict, stack=True)
    closes = stocks_df['close']
    volumes = stocks_df['volume']
    highs = stocks_df['high']
    lows = stocks_df['low']
    opens = stocks_df['open']
    
    benchmark = get_stocks(symbols_date_dict, 'close', benchmark=True)[symbol_benchmark]
    
    window = st.slider("Select lookback period for average volume", 5, 504, 20)
    
    st.write("Volume Trend Persistence (VTP) Ratio")
    vtp = calculate_vtp(closes, volumes, window=window)
    price_range = caulate_price_range(closes, window=window)
    vtp_ratio = vtp / price_range

    
    rsr = calculate_rsr(stocks_df, window=window)    
    vpd = calculate_vpd(stocks_df, volumes, window=window, price_window=window)
    cpd = calculate_cpd(stocks_df, window=5, vol_window=window)
    vap = calculate_vap(stocks_df, volumes, window=window)
    vap_ratio = vap / price_range
    
    projection_period = st.slider("Select projection period", 1, 20, 10)
    
    close_ahead  = closes.shift(-10)
    projected_returns = (close_ahead - closes) / closes
    
    metrics = {
        'VTP Ratio': vtp_ratio,
        'RSR Ratio': rsr,
        # 'VPD Ratio': vpd,
        # 'CPD Ratio': cpd,
        'VAP Ratio': vap_ratio
    }
   
    vtp_threshold = st.slider("Select VTP threshold", 0.0, 5.0, 3.0)
    rsr_threshold = st.slider("Select RSR threshold", 0.0, 5.0, 3.0)
    vap_threshold = st.slider("Select VAP threshold", 0.0, 5.0, 3.0)
    
    combined_signal = (vtp_ratio > vtp_threshold) & (rsr > rsr_threshold) & (vap_ratio > vap_threshold)
    
    # Stats
    # Calculate of signals and returns
    valiable_metrics = {
        'VTP Ratio': vtp[vtp > vtp_threshold],
        'RSR Ratio': rsr[rsr > rsr_threshold],
        'VAP Ratio': vap[vap > vap_threshold],
        'Combined Signal': combined_signal[combined_signal]
    }
    
    # calculate returns > 0 when signal > 0
    results = {}
    
    for metric, values in valiable_metrics.items():
        signal_mask = values > 0
        total_signals = np.sum(signal_mask)
        if total_signals.empty:
            continue
        
        positive_signals = projected_returns[signal_mask]
        positive_signals_count = np.sum(positive_signals > 0)
        accuracy = positive_signals_count / total_signals
        avg_return = np.mean(positive_signals)
        sharp_ratio = avg_return / np.std(positive_signals)
        
        results[metric] = {
            'accuracy': accuracy.mean(axis=0),
            'sharp_ratio': sharp_ratio.mean(axis=0),
            'avg_return': avg_return,
            'total_signals': total_signals.mean(axis=0)
        }
        
    st.write("### Results")
    results_df = pd.DataFrame(results).T
    st.write(results_df)
    
    # signal by ticker
    st.write("### Signals by Ticker")
    st.write(combined_signal.sum().sort_values(ascending=False).head(10))
    
    selected_stock = st.selectbox("Select stock to plot", closes.columns)
    combined_signal = combined_signal[selected_stock].astype(int)
    buy_signal = combined_signal[combined_signal == 1]
    sell_signal = combined_signal[combined_signal == 0]
    # plot price and signal as marker
    fig = make_subplots(rows=1, cols=1, shared_xaxes=True, subplot_titles=("Price", "Signal"))
    fig.add_trace(go.Scatter(x=closes.index, y=closes[selected_stock], name=selected_stock), row=1, col=1)
    # plot buy signal as green marker y = price
    fig.add_trace(go.Scatter(x=buy_signal.index, y=closes[selected_stock][buy_signal.index], mode='markers', marker=dict(color='green', size=10), name='Buy Signal'), row=1, col=1)
    # plot sell signal as red marker y = price
    # fig.add_trace(go.Scatter(x=sell_signal.index, y=closes[selected_stock][sell_signal.index], mode='markers', marker=dict(color='red', size=10), name='Sell Signal'), row=1, col=1)
    st.plotly_chart(fig)
    