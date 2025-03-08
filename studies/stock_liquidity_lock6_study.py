import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# Assuming these are your existing utility functions
from utils.processing import get_stocks

def run(symbol_benchmark, symbolsDate_dict):    
    with st.expander("Liquidity Lock Ratio (LLR) Study"):
        st.write("This section analyzes the Liquidity Lock Ratio (LLR) to detect short squeeze-like behavior over a T+2.5 period.")
    
    if len(symbolsDate_dict['symbols']) < 1:
        st.info("Please select symbols.")
        st.stop()
    
    # Load data
    stocks_df = get_stocks(symbolsDate_dict, 'close')
    stocks_volume_df = get_stocks(symbolsDate_dict, 'volume')

    # Calculate LLR
    lookback_period = st.slider("Select lookback period for average volume", 5, 504, 400)
    avg_volume_df = stocks_volume_df.rolling(window=lookback_period).mean()
    
    # Daily price change sign
    price_change_df = stocks_df.pct_change()
    price_change_sign_df = np.sign(price_change_df)
    
    # LLR = (Current Day Volume × Sign(Price Change)) / Average Volume
    llr_df = (stocks_volume_df * price_change_sign_df) / avg_volume_df
    
    # Define threshold for LLR breakout
    llr_threshold = st.slider("Select LLR threshold for breakout", 1.0,50.0, 20.0)
    llr_breakout_df = llr_df > llr_threshold  # Positive spikes indicate potential squeezes

    # Price change over next 2-3 days (approximating T+2.5)
    stats_period = st.slider("Select period for post-breakout price stats (T+2.5 approx)", 2, 15, 2)
    price_ahead_df = stocks_df.shift(-stats_period)

    # Create a DataFrame of all breakout signals
    signals_list = []
    for symbol in llr_breakout_df.columns:
        breakout_dates = llr_breakout_df.index[llr_breakout_df[symbol]]
        for date in breakout_dates:
            entry_price = stocks_df[symbol].loc[date]
            exit_price = price_ahead_df[symbol].loc[date]
            Returns = (exit_price - entry_price) / entry_price
            Avg_Volume = avg_volume_df[symbol].loc[date]
            Target_LLR = llr_threshold
            Target_Volume = int(Target_LLR * Avg_Volume)
            signal_data = {
                'Date': date,
                'Symbol': symbol,
                'LLR': llr_df[symbol].loc[date],
                'Target_Volume': Target_Volume,
                'Price': entry_price,
                'Price_Ahead': exit_price,
                'Returns': Returns
            }
            signals_list.append(signal_data)
    
    signals_df = pd.DataFrame(signals_list)
    signals_df = signals_df.sort_values(by='Date', ascending=False)
    
    
    # Metrics
    total_signals = llr_breakout_df.sum().sum()
    accuracy = signals_df['Returns'] > 0
    avg_price_change = signals_df['Returns']
    transaction_cost = 0.0016
    profitable = avg_price_change.mean() - transaction_cost

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total LLR Signals", int(total_signals))
    with col2:
        st.metric("Accuracy (Price Up)", f"{accuracy.mean() * 100:.2f}%")
    with col3:
        st.metric("Avg Price Change", f"{avg_price_change.mean() * 100:.2f}%")
    st.write(f"Average profitable after transaction cost: {profitable * 100:.2f}%")
            
    # daily stats  
    daily_returns = signals_df.groupby('Date').agg({
        'Returns': ['mean', 'count']
    })
    
    # avg signal per day
    avg_signals_per_day = daily_returns['Returns']['count'].mean()
    st.write(f"Average signals per day: {avg_signals_per_day:.2f}")
    
    acc_daily_returns = (1 + daily_returns['Returns']['mean']).cumprod() - 1
    
    # signals daily count
    st.write("### Daily Signals Count")
    fig = go.Figure()
    fig.add_trace(go.Bar(x=daily_returns.index, y=daily_returns['Returns']['count'], name='Signals Count'))
    fig.update_layout(title_text="Daily Signals Count", height=400)
    st.plotly_chart(fig)
    
    st.write("### Accumulated Daily Returns")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=acc_daily_returns.index, y=acc_daily_returns, mode='lines', name='Accumulated Daily Returns'))
    fig.update_layout(title_text="Accumulated Daily Returns", height=400)
    st.plotly_chart(fig)
    
    # return distribution
    st.write("### Returns Distribution")
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=signals_df['Returns'], nbinsx=20))
    fig.update_layout(title_text="Returns Distribution", height=400)
    st.plotly_chart(fig)
        
    # Display signals
    show_signals = st.checkbox("Show signals")
    if show_signals:
        st.write("### All LLR Breakout Signals")
        if not signals_df.empty:
            st.dataframe(signals_df.style.format({
                'LLR': "{:.2f}",
                'Price': "{:.2f}",
                'Price_Ahead': "{:.2f}",
                'Returns': "{:.2%}",
                'Target_Volume': "{:.0f}"
            }))
        else:
            st.write("No LLR breakout signals detected with the current settings.")
        
    # Tickers stats
    show_tickers_stats = st.checkbox("Show tickers stats")
    if show_tickers_stats:
        # group by symbol
        symbol_stats = signals_df.groupby('Symbol').agg({
            'Returns': ['mean', 'count', lambda x: (x > 0).mean()]
        })
                 
        symbol_stats.columns = ['Avg Returns', 'Count', 'Accuracy']
        symbol_stats = symbol_stats.sort_values(by='Avg Returns', ascending=False)
        st.dataframe(symbol_stats, use_container_width=True)
        

    # Plot for single symbol
    if len(symbolsDate_dict['symbols']) == 1:
        symbol = symbolsDate_dict['symbols'][0]
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02)
        
        # Price plot with signals
        fig.add_trace(go.Scatter(x=stocks_df.index, y=stocks_df[symbol], mode='lines', name=f"{symbol} Price"), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=signals_df['Date'], 
            y=signals_df['Price'], 
            mode='markers', 
            name="LLR Breakouts", 
            marker=dict(size=10, color='red')
        ), row=1, col=1)
        
        # LLR plot
        fig.add_trace(go.Scatter(x=llr_df.index, y=llr_df[symbol], mode='lines', name="LLR"), row=2, col=1)
        fig.add_hline(y=llr_threshold, line_dash="dash", line_color="red", row=2, col=1)
        
        fig.update_layout(title_text=f"{symbol} Price with LLR Breakout Signals", height=600)
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Liquidity Lock Ratio", row=2, col=1)
        st.plotly_chart(fig)
