import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objs as go
from itertools import product

from utils.processing import get_stocks

def run_simulation(stocks_df, stocks_volume_df, windows, thresholds, period):
    """
    Simulate strategy across different windows and thresholds for a fixed period
    Returns DataFrame with window, threshold, accuracy, and avg_return
    """
    results = []
    
    # Pre-calculate price changes once since period is fixed
    price_ahead_df = stocks_df.shift(-period)
    price_change_df = (price_ahead_df - stocks_df) / stocks_df
    
    # Vectorized simulation across all combinations
    for window, threshold in product(windows, thresholds):
        # Calculate volume z-scores
        vol_rolling_mean = stocks_volume_df.rolling(window=window).mean()
        vol_rolling_std = stocks_volume_df.rolling(window=window).std()
        volume_z_score_df = (stocks_volume_df - vol_rolling_mean) / vol_rolling_std
        
        # Calculate breakout signals
        volume_breakout_df = volume_z_score_df > threshold
        
        # Calculate metrics efficiently
        signals_mask = volume_breakout_df.values
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
    with st.expander("Liquidity lock study"):
        st.write("This section analyzes volume breakouts and subsequent price movements.")
    
    if len(symbolsDate_dict['symbols']) < 1:
        st.info("Please select symbols.")
        st.stop()
    
    # Load data
    stocks_df = get_stocks(symbolsDate_dict, 'close')
    stocks_volume_df = get_stocks(symbolsDate_dict, 'volume')

    # Parameter selection
    stats_period = st.slider("Select period for stats", 1, 10, 2)
    
    # Generate Fibonacci sequence for thresholds
    def fibonacci(n, u0 = 0, u1 = 1):
        fib_sequence = [u0, u1]
        while len(fib_sequence) < n:
            fib_sequence.append(fib_sequence[-1] + fib_sequence[-2])
        return fib_sequence
    
    # Simulation parameters
    window_options =  [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340, 360, 380, 400, 420, 440]
    windows = st.multiselect("Select window sizes to simulate", 
                           options=window_options, 
                           default=window_options)

    # Generate Fibonacci thresholds
    fibonacci_thresholds = [2.0,  2.5,  3.0,  3.5,  4.0,  4.5,  5.0,  5.5,  6.0,  6.5,  7.0,  7.5,  8.0,  8.5,  9.0,  9.5, 10.0]
    thresholds_options = [float(fib) for fib in fibonacci_thresholds if fib > 0]  # Convert to float and exclude 0
    thresholds = st.multiselect("Select volume z-score thresholds", 
                              options=thresholds_options,
                              default=thresholds_options)
    
    # Run simulation and output results
    results_df = run_simulation(stocks_df, stocks_volume_df, windows, thresholds, stats_period)
    
    # Find and display optimal parameters
    if not results_df.empty:        
        # Create scatter plot of window vs threshold, color by accuracy
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=results_df['window'],
            y=results_df['threshold'],
            mode='markers',
            marker=dict(
                size=10,
                color=results_df['accuracy'],
                colorscale='Viridis',
                showscale=True
            )
        ))
        fig.update_layout(title="Window vs Threshold - Accuracy", xaxis_title="Window", yaxis_title="Threshold")
        st.plotly_chart(fig)
        
        # Create scatter plot of window vs threshold, color by total signals
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=results_df['window'],
            y=results_df['threshold'],
            mode='markers',
            marker=dict(
                size=10,
                color=results_df['total_signals'],
                colorscale='Viridis',
                showscale=True
            )
        ))
        fig.update_layout(title="Window vs Threshold - Total Signals", xaxis_title="Window", yaxis_title="Threshold")
        st.plotly_chart(fig)
        
        # Create scatter plot of window vs threshold, color by total signals
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=results_df['window'],
            y=results_df['threshold'],
            mode='markers',
            marker=dict(
                size=10,
                color=results_df['avg_return'],
                colorscale='Viridis',
                showscale=True
            )
        ))
        fig.update_layout(title="Window vs Threshold - Avg Return", xaxis_title="Window", yaxis_title="Threshold")
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