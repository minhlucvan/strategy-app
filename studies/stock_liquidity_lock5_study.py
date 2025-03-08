import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objs as go
from itertools import product

from utils.processing import get_stocks

import numpy as np
import pandas as pd
from itertools import product

@st.cache_data
def run_simulation(stocks_df, stocks_volume_df, windows, thresholds, period=2):
    """
    Simulate short squeeze detection using Liquidity Lock Ratio (LLR)
    LLR = (Volume × Sign(Price Change)) / Avg Volume over Window
    """
    results = []
    
    # Calculate daily price change and sign
    price_change_df = stocks_df.pct_change()
    price_change_sign_df = np.sign(price_change_df)
    
    for window, threshold in product(windows, thresholds):
        # Compute rolling average volume
        avg_volume_df = stocks_volume_df.rolling(window=window).mean()
        
        # Calculate LLR
        llr_df = (stocks_volume_df * price_change_sign_df) / avg_volume_df
        
        # Detect squeeze signals (LLR exceeding threshold)
        llr_breakout_df = llr_df > threshold
        
        # Price change over T+period days
        price_ahead_df = stocks_df.shift(-period)
        future_price_change_df = (price_ahead_df - stocks_df) / stocks_df
        
        # Extract signals
        signals_mask = llr_breakout_df.values
        total_signals = np.sum(signals_mask).astype(int)
        if total_signals == 0:
            continue
        
        signals_returns = np.nan_to_num(future_price_change_df.values[signals_mask])
        positive_signals_count = np.sum(signals_returns > 0)
        avg_loss = np.mean(signals_returns[signals_returns < 0]) if len(signals_returns[signals_returns < 0]) > 0 else 0
        avg_win = np.mean(signals_returns[signals_returns > 0]) if len(signals_returns[signals_returns > 0]) > 0 else 0
        risk_reward_ratio = avg_win / abs(avg_loss) if abs(avg_loss) > 0 else 0
        accuracy = positive_signals_count / total_signals
        avg_return = np.mean(signals_returns) if len(signals_returns) > 0 else 0
        sharp_ratio = avg_return / np.std(signals_returns) if len(signals_returns) > 0 else 0
        signal_by_day = np.sum(signals_mask, axis=1)
        signal_density = np.mean(signal_by_day)
        
        results.append({
            'window': window,
            'threshold': threshold,
            'accuracy': accuracy,
            'sharp_ratio': sharp_ratio,
            'avg_return': avg_return,
            'avg_loss': avg_loss,
            'avg_win': avg_win,
            'risk_reward_ratio': risk_reward_ratio,
            'total_signals': total_signals,
            'signal_density': signal_density
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
    
    target_density = st.slider("Select target density (signals per year)", 1, 252, 100)
    target_density = target_density / 252  # Convert to daily signals

    # Simulation parameters
    window_options =  [5, 10, 20, 30, 40, 50, 60, 70, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340, 360, 380, 400, 420, 440, 460, 480, 500]
    windows = st.multiselect("Select window sizes to simulate", 
                           options=window_options, 
                           default=window_options)

    # Generate Fibonacci thresholds
    fibonacci_thresholds = [5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0, 120.0, 125.0, 130.0, 135.0, 140.0, 145.0, 150.0]
    thresholds_options = [float(fib) for fib in fibonacci_thresholds if fib > 0]  # Convert to float and exclude 0
    thresholds = st.multiselect("Select volume z-score thresholds", 
                              options=thresholds_options,
                              default=thresholds_options)
    
    # Run simulation and output results
    results_df = run_simulation(stocks_df, stocks_volume_df, windows, thresholds, stats_period)
    
    # Filter results by target density
    results_df = results_df[results_df['signal_density'] >= target_density]
    
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
        
        # Create scatter plot of window vs window, color by sharp ratio
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=results_df['window'],
            y=results_df['threshold'],
            mode='markers',
            marker=dict(
                size=10,
                color=results_df['sharp_ratio'],
                colorscale='Viridis',
                showscale=True
            )
        ))
        fig.update_layout(title="Window vs Threshold - Sharp Ratio", xaxis_title="Window", yaxis_title="Threshold")
        st.plotly_chart(fig)
        
        # Create scatter plot of window vs threshold, color by avg return
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
        
        #  Create scatter plot of window vs threshold, color by total signals
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
        
        # Display optimal parameters
        rank_accuracy = results_df.sort_values(by='accuracy', ascending=False)
        rank_sharp_ratio = results_df.sort_values(by='sharp_ratio', ascending=False)
        rank_avg_return = results_df.sort_values(by='avg_return', ascending=False)
        rank_total_signals = results_df.sort_values(by='total_signals', ascending=False)
        rank_risk_reward_ratio = results_df.sort_values(by='risk_reward_ratio', ascending=False)
        
        st.write("### Optimal Parameters")
        optimal_df = pd.DataFrame({
            'Accuracy': rank_accuracy.iloc[0],
            'Sharp Ratio': rank_sharp_ratio.iloc[0],
            'Avg Return': rank_avg_return.iloc[0],
            'Total Signals': rank_total_signals.iloc[0],
            'Risk Reward Ratio': rank_risk_reward_ratio.iloc[0]
        }).T
        st.dataframe(optimal_df.style.format({
            'Accuracy': "{:.2%}",
            'Sharp Ratio': "{:.2f}",
            'Avg Return': "{:.2%}",
            'Total Signals': "{:.0f}",
            'Risk Reward Ratio': "{:.2f}",
        }), use_container_width=True)
    
        # Display results table

        st.write("### Simulation Results")
        st.dataframe(results_df.style.format({
            'accuracy': "{:.2%}",
            'avg_return': "{:.2%}"
        }), use_container_width=True)
    else:
        st.write("No results generated with current parameters.")