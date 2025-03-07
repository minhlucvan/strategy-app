import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objs as go
from itertools import product

from utils.processing import get_stocks

def run_simulation(stocks_df, stocks_volume_df, windows, thresholds, period):
    """
    Simulate strategy using liquidity as volume * sign(price_change)
    Returns DataFrame with window, threshold, accuracy, and avg_return
    """
    results = []
    
    # Pre-calculate price changes
    price_ahead_df = stocks_df.shift(-period)
    price_change_df = (price_ahead_df - stocks_df) / stocks_df
    # Create sign of price change (-1 or 1)
    price_change_sign = np.sign(price_change_df)
    
    # Calculate liquidity (volume * sign of price change)
    liquidity_df = stocks_volume_df * price_change_sign
    
    # Vectorized simulation across all combinations
    for window, threshold in product(windows, thresholds):
        # Calculate liquidity z-scores
        liq_rolling_mean = liquidity_df.rolling(window=window).mean()
        liq_rolling_std = liquidity_df.rolling(window=window).std()
        liquidity_z_score_df = (liquidity_df - liq_rolling_mean) / liq_rolling_std
        
        # Calculate breakout signals based on liquidity z-score
        liquidity_breakout_df = liquidity_z_score_df > threshold
        
        # Calculate metrics efficiently
        signals_mask = liquidity_breakout_df.values
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
    with st.expander("Liquidity Lock Study (Volume * Sign(Price Change))"):
        st.write("This section analyzes liquidity breakouts (volume * sign(price change)) "
                "and subsequent price movements. Liquidity is defined as volume when "
                "price change > 0 and -volume when price change < 0.")
    
    if len(symbolsDate_dict['symbols']) < 1:
        st.info("Please select symbols.")
        st.stop()
    
    # Load data
    stocks_df = get_stocks(symbolsDate_dict, 'close')
    stocks_volume_df = get_stocks(symbolsDate_dict, 'volume')

    # Parameter selection
    stats_period = st.slider("Select period for stats", 1, 10, 2)
    
    # Window and threshold options
    window_options = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200]
    windows = st.multiselect("Select window sizes to simulate", 
                           options=window_options, 
                           default=[20, 50, 100])

    fibonacci_thresholds = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    thresholds = st.multiselect("Select liquidity z-score thresholds", 
                              options=fibonacci_thresholds,
                              default=[2.0, 3.0, 4.0])
    
    # Run simulation and output results
    results_df = run_simulation(stocks_df, stocks_volume_df, windows, thresholds, stats_period)
    
    if not results_df.empty:        
        # Create scatter plots
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
                    size=10,
                    color=results_df[metric],
                    colorscale='Viridis',
                    showscale=True
                )
            ))
            fig.update_layout(
                title=f"Window vs Threshold - {title}",
                xaxis_title="Window",
                yaxis_title="Threshold"
            )
            st.plotly_chart(fig)
        
        # Display optimal parameters
        optimal = results_df.loc[results_df['accuracy'].idxmax()]
        st.write(f"Optimal parameters: Window={optimal['window']}, "
                f"Threshold={optimal['threshold']:.1f}, "
                f"Accuracy={optimal['accuracy']:.2%}, "
                f"Avg Return={optimal['avg_return']:.2%}")
    
        # Display results table
        st.write("### Simulation Results")
        st.dataframe(results_df.style.format({
            'accuracy': "{:.2%}",
            'avg_return': "{:.2%}"
        }))
    else:
        st.write("No results generated with current parameters.")