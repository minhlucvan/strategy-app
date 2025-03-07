import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objs as go
from plotly.subplots import make_subplots

from utils.processing import get_stocks

def run(symbol_benchmark, symbolsDate_dict):    
    with st.expander("Optimized ELLR Study - Vietnam T+2.5"):
        st.write("""Here's the updated strategy description:  

**Optimized ELLR Study - Vietnam T+2.5**  

This strategy refines the Enhanced Liquidity-Linked Ratio (ELLR) model by incorporating a dynamic threshold and additional market filters to identify potential breakout stocks in Vietnam’s T+2.5 settlement environment.  

### Key Components:  
1. **ELLR Calculation**:  
   - Uses intraday volatility, trading volume, and price movement signals.  
   - Adjusted for float availability to enhance accuracy.  

2. **Dynamic Thresholding**:  
   - Identifies breakout stocks using a rolling percentile filter (e.g., 90th percentile).  

3. **Market Trend & Volatility Filters**:  
   - Trend confirmation via short-term (10-day) and long-term (20-day) SMA crossovers.  
   - Volatility filter ensures ATR is above its historical average.  

4. **T+2.5 Profitability Assessment**:  
   - Tracks price movements 2–3 days post-breakout.  
   - Applies a minimum gain filter to refine entry points.  

5. **Performance Metrics & Visualization**:  
   - Reports signal count, accuracy, and net profitability after costs.  
   - Interactive stock charts highlight key breakouts.  

This approach enhances breakout detection while filtering out false signals, offering a systematic way to capture short-term price movements in Vietnam’s stock market.""")
    
    if len(symbolsDate_dict['symbols']) < 1:
        st.info("Please select symbols.")
        st.stop()
    
    # Load data (assuming OHLCV available)
    stocks_full_df = get_stocks(symbolsDate_dict, stack=True)
    
    stocks_df = stocks_full_df['close']
    stocks_volume_df = stocks_full_df['volume']
    stocks_high_df = stocks_full_df['high']
    stocks_low_df = stocks_full_df['low']

    # ELLR Components
    lookback_period = st.slider("Lookback period", 5, 50, 20)
    avg_volume_df = stocks_volume_df.rolling(window=lookback_period).mean()
    price_change_df = stocks_df.pct_change()
    price_change_sign_df = np.sign(price_change_df)
    intraday_volatility_df = (stocks_high_df - stocks_low_df) / stocks_df
    float_availability_df = 1 - (stocks_volume_df.rolling(window=5).sum() / stocks_volume_df.rolling(window=lookback_period).sum()).clip(0, 1)
    ellr_df = (stocks_volume_df * price_change_sign_df * intraday_volatility_df) / (avg_volume_df * float_availability_df)
    
    # Dynamic Threshold: Top 10% of ELLR values
    ellr_threshold_percentile = st.slider("ELLR percentile threshold", 80, 95, 90)  # 90th percentile default
    ellr_threshold_df = ellr_df.rolling(window=lookback_period).quantile(ellr_threshold_percentile / 100)
    ellr_breakout_df = ellr_df > ellr_threshold_df
    
    # Trend Filter: 10-day SMA > 20-day SMA
    sma_short_df = stocks_df.rolling(window=10).mean()
    sma_long_df = stocks_df.rolling(window=20).mean()
    trend_filter_df = sma_short_df > sma_long_df
    
    # Volatility Filter: ATR > Avg ATR
    atr_df = ((stocks_high_df - stocks_low_df) + price_change_df.abs() * stocks_df).rolling(window=14).mean()
    atr_avg_df = atr_df.rolling(window=lookback_period).mean()
    volatility_filter_df = atr_df > atr_avg_df
    
    # Combined Breakout
    breakout_df = ellr_breakout_df & trend_filter_df & volatility_filter_df

    # T+2.5 Price Movement
    stats_period = st.slider("T+2.5 period (days)", 2, 3, 2)
    price_ahead_df = stocks_df.shift(-stats_period)
    future_price_change_df = (price_ahead_df - stocks_df) / stocks_df

    # Signals with Minimum Gain Filter
    min_gain = st.slider("Minimum gain filter (%)", 0.0, 1.0, 0.5) / 100
    signals_list = []
    for symbol in breakout_df.columns:
        breakout_dates = breakout_df.index[breakout_df[symbol]]
        for date in breakout_dates:
            entry_price = stocks_df[symbol].loc[date]
            exit_price = price_ahead_df[symbol].loc[date]
            price_change = future_price_change_df[symbol].loc[date]
            if price_change >= min_gain:
                signal_data = {
                    'Date': date,
                    'Symbol': symbol,
                    'ELLR': ellr_df[symbol].loc[date],
                    'Price': entry_price,
                    'Price_Ahead': exit_price,
                    'Price_Change (%)': price_change * 100,
                }
                signals_list.append(signal_data)
    
    signals_df = pd.DataFrame(signals_list)
    
    # Metrics
    total_signals = len(signals_df)
    accuracy = (signals_df['Price_Change (%)'] > 0).mean()
    avg_price_change = signals_df['Price_Change (%)'].mean() / 100
    transaction_cost = 0.002
    profitable = avg_price_change - transaction_cost

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total ELLR Signals", total_signals)
    with col2:
        st.metric("Accuracy (Price Up)", f"{accuracy * 100:.2f}%")
    with col3:
        st.metric("Avg Price Change", f"{avg_price_change * 100:.2f}%")
    st.write(f"Net profitable after costs: {profitable * 100:.2f}%")

    # Display signals
    if st.checkbox("Show signals"):
        st.write("### ELLR Breakout Signals")
        # sort by date
        signals_df = signals_df.sort_values(by='Date', ascending=False)
        if not signals_df.empty:
            st.dataframe(signals_df.style.format({
                'ELLR': "{:.2f}",
                'Price': "{:.2f}",
                'Price_Ahead': "{:.2f}",
                'Price_Change (%)': "{:.2f}%"
            }))
        else:
            st.write("No signals detected.")

    # Plot
    if len(symbolsDate_dict['symbols']) == 1:
        symbol = symbolsDate_dict['symbols'][0]
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05)
        fig.add_trace(go.Scatter(x=stocks_df.index, y=stocks_df[symbol], mode='lines', name=f"{symbol} Price"), row=1, col=1)
        fig.add_trace(go.Scatter(x=signals_df['Date'], y=signals_df['Price'], mode='markers', name="Breakouts", marker=dict(size=10, color='red')), row=1, col=1)
        fig.add_trace(go.Scatter(x=ellr_df.index, y=ellr_df[symbol], mode='lines', name="ELLR"), row=2, col=1)
        fig.add_trace(go.Scatter(x=ellr_threshold_df.index, y=ellr_threshold_df[symbol], mode='lines', name="Threshold", line=dict(dash='dash', color='red')), row=2, col=1)
        fig.add_trace(go.Scatter(x=stocks_volume_df.index, y=stocks_volume_df[symbol], mode='lines', name="Volume"), row=3, col=1)
        fig.update_layout(title_text=f"{symbol} - ELLR Analysis", height=800)
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="ELLR", row=2, col=1)
        fig.update_yaxes(title_text="Volume", row=3, col=1)
        st.plotly_chart(fig)
