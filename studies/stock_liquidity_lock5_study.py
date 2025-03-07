import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objs as go
from itertools import product

from utils.processing import get_stocks

def run_simulation(stocks_df, stocks_volume_df, windows, thresholds, period=2):
    """
    Simulate short squeeze detection using Liquidity Lock Ratio (LLR)
    LLR = (Volume × Sign(Price Change)) / Avg Volume over Window
    Period is fixed to ~2.5 days (2-3 days for simplicity)
    """
    results = []
    
    # Pre-calculate price changes over T+2 period
    price_ahead_df = stocks_df.shift(-period)  # T+2 as proxy for T+2.5
    price_change_df = (price_ahead_df - stocks_df) / stocks_df
    price_change_sign = np.sign(price_change_df)
    
    # Calculate directional liquidity
    liquidity_df = stocks_volume_df * price_change_sign
    
    for window, threshold in product(windows, thresholds):
        # Calculate LLR
        avg_volume = stocks_volume_df.rolling(window=window).mean()
        llr_df = liquidity_df / avg_volume  # Liquidity Lock Ratio
        
        # Detect squeeze signals (high LLR with positive price change)
        squeeze_signals_df = (llr_df > threshold) & (price_change_df > 0)
        
        # Calculate metrics
        signals_mask = squeeze_signals_df.values
        price_changes = price_change_df.values
        
        total_signals = np.sum(signals_mask)
        if total_signals == 0:
            continue
            
        positive_returns = np.sum(signals_mask & (price_changes > 0))
        accuracy = positive_returns / total_signals if total_signals > 0 else 0
        avg_return = np.nanmean(price_changes[signals_mask]) if total_signals > 0 else 0
        
        results.append({
            'window': window,
            'threshold': threshold,
            'accuracy': accuracy,
            'avg_return': avg_return,
            'total_signals': total_signals
        })
    
    return pd.DataFrame(results)

def run(symbol_benchmark, symbolsDate_dict):    
    with st.expander("Short Squeeze Detector (T+2.5 Liquidity Lock)"):
        st.write("Detects short squeeze-like behavior using Liquidity Lock Ratio (LLR): "
                "(Volume × Sign(Price Change)) / Avg Volume. Signals trigger when "
                "LLR exceeds threshold and price increases over ~2.5 days.")
    
    if len(symbolsDate_dict['symbols']) < 1:
        st.info("Please select symbols.")
        st.stop()
    
    # Load data
    stocks_df = get_stocks(symbolsDate_dict, 'close')
    stocks_volume_df = get_stocks(symbolsDate_dict, 'volume')

    # Fixed T+2 period (closest integer proxy for T+2.5)
    stats_period = 2
    st.write(f"Using T+{stats_period} period as proxy for T+2.5 settlement")
    
    # Window and threshold options
    window_options = [5, 10, 20, 30, 50]  # Shorter windows for quicker signals
    windows = st.multiselect("Select lookback windows for avg volume", 
                           options=window_options, 
                           default=[10, 20, 30])

    thresholds = st.multiselect("Select LLR thresholds", 
                              options=[2.0, 3.0, 4.0, 5.0],
                              default=[3.0, 4.0])
    
    # Run simulation
    results_df = run_simulation(stocks_df, stocks_volume_df, windows, thresholds, stats_period)
    
    if not results_df.empty:        
        # Scatter plots
        for metric, title in [
            ('accuracy', 'Accuracy'),
            ('total_signals', 'Total Signals'),
            ('avg_return', 'Avg Return')
        ]:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=results_df['window'],
                y=results_df['threshold'],
                mode='markers',
                marker=dict(
                    size=12,
                    color=results_df[metric],
                    colorscale='Viridis',
                    showscale=True
                )
            ))
            fig.update_layout(
                title=f"Window vs Threshold - {title}",
                xaxis_title="Lookback Window",
                yaxis_title="LLR Threshold"
            )
            st.plotly_chart(fig)
        
        # Optimal parameters
        optimal = results_df.loc[results_df['avg_return'].idxmax()]  # Optimize for return
        st.write(f"Optimal parameters: Window={optimal['window']}, "
                f"Threshold={optimal['threshold']:.1f}, "
                f"Accuracy={optimal['accuracy']:.2%}, "
                f"Avg Return={optimal['avg_return']:.2%}")
    
        # Results table
        st.write("### Simulation Results")
        st.dataframe(results_df.style.format({
            'accuracy': "{:.2%}",
            'avg_return': "{:.2%}"
        }))
    else:
        st.write("No squeeze signals detected with current parameters.")