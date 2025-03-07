import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# Assuming these are your existing utility functions
from utils.processing import get_stocks

def run(symbol_benchmark, symbolsDate_dict):    
    with st.expander("Liquidity lock study"):
        st.write("This section analyzes volume breakouts and subsequent price movements.")
    
    if len(symbolsDate_dict['symbols']) < 1:
        st.info("Please select symbols.")
        st.stop()
    
    # Load data
    stocks_df = get_stocks(symbolsDate_dict, 'close')
    stocks_volume_df = get_stocks(symbolsDate_dict, 'volume')

    # Volume breakout detection
    window = st.slider("Select the window size for rolling avg volume", 1, 252, 100)
    volume_z_score_df = (stocks_volume_df - stocks_volume_df.rolling(window=window).mean()) / stocks_volume_df.rolling(window=window).std()
    liquidity_threshold = st.slider("Select volume z-score threshold", 0.0, 10.0, 5.0)
    volume_breakout_df = volume_z_score_df > liquidity_threshold

    # Price change over next X days
    stats_period = st.slider("Select period for stats", 1, 10, 2)
    # shift price from the future to the present
    price_ahead_df = stocks_df.shift(-stats_period)
    price_change_df = (price_ahead_df - stocks_df) / stocks_df

    # Create a DataFrame of all breakout signals
    signals_list = []
    for symbol in volume_breakout_df.columns:
        breakout_dates = volume_breakout_df.index[volume_breakout_df[symbol]]
        for date in breakout_dates:
            entry_price = stocks_df[symbol].loc[date]
            exit_price = price_ahead_df[symbol].loc[date]
            signal_data = {
                'Date': date,
                'Symbol': symbol,
                'Volume_Z_Score': volume_z_score_df[symbol].loc[date],
                'Price': stocks_df[symbol].loc[date],
                'Price_ahead': exit_price,
                'Price_Change_After_{}_Days'.format(stats_period): price_change_df[symbol].loc[date] * 100,  # In percentage
            }
            signals_list.append(signal_data)
    
    signals_df = pd.DataFrame(signals_list)
    
    # Existing metrics (unchanged)
    total_signals = volume_breakout_df.sum().sum()
    accuracy = (volume_breakout_df & (price_change_df > 0)).sum() / volume_breakout_df.sum()
    avg_price_change = price_change_df[volume_breakout_df].mean()
    transaction_cost = 0.0016
    profitable = avg_price_change.mean() - transaction_cost

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total signals", int(total_signals))
    with col2:
        st.metric("Accuracy", f"{accuracy.mean() * 100:.2f}%")
    with col3:
        st.metric("Avg price change", f"{avg_price_change.mean() * 100:.2f}%")
    st.write(f"Average profitable: {profitable * 100:.2f}%")


    # Display the signals DataFrame
    show_signals = st.checkbox("Show all signals")
    if show_signals:
        st.write("### All Volume Breakout Signals")
        if not signals_df.empty:
            st.dataframe(signals_df.style.format({
                'Volume_Z_Score': "{:.2f}",
                'Price': "{:.2f}",
                'Price_Change_After_{}_Days'.format(stats_period): "{:.2f}%"
            }))
        else:
            st.write("No breakout signals detected with the current settings.")

    # Optional: Plot price with signals for cross-checking
    if len(symbolsDate_dict['symbols']) == 1:
        symbol = symbolsDate_dict['symbols'][0]
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02)
        
        # Price plot
        fig.add_trace(go.Scatter(x=stocks_df.index, y=stocks_df[symbol], mode='lines', name=f"{symbol} Price"), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=signals_df['Date'], 
            y=signals_df['Price'], 
            mode='markers', 
            name="Breakout Signals", 
            marker=dict(size=10, color='red')
        ), row=1, col=1)
        
        # Volume z-score plot
        fig.add_trace(go.Scatter(x=volume_z_score_df.index, y=volume_z_score_df[symbol], mode='lines', name="Volume Z-Score"), row=2, col=1)
        
        fig.update_layout(title_text=f"{symbol} Price with Volume Breakout Signals", height=600)
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Volume Z-Score", row=2, col=1)
        st.plotly_chart(fig)