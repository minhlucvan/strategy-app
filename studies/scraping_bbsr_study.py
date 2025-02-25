import streamlit as st
import pandas as pd
import numpy as np
import json
import talib as ta
from streamlit_lightweight_charts import renderLightweightCharts
from utils.processing import get_stocks

# Calculate consecutive lows
def calculate_consecutive_lows(df, n=3):
    lows = df['low']
    return (lows.shift(1) > lows).rolling(n).sum()

# Chart options
chart_options = {
    "layout": {
        "textColor": 'black',
        "background": {
            "type": 'solid',
            "color": 'white'
        }
    },
    "timeScale": {
        "timeVisible": True,
        "secondsVisible": False
    }
}

def prepare_chart_data(df, value_column):
    """Convert DataFrame to Lightweight Charts format"""
    return [
        {"time": int(index.timestamp()), "value": float(value)}
        for index, value in df[value_column].items()
    ]

def run(symbol_benchmark, symbolsDate_dict):
    # if len(symbolsDate_dict['symbols']) < 1:
    #     st.info("Please select symbols.")
    #     st.stop()
        
    symbolsDate_dict['symbols'] = ['VN30f1M']
            
    # Get stock prices
    prices_df = get_stocks(symbolsDate_dict, stack=True)
    
    # Calculate indicators
    rsi = ta.RSI(prices_df['close'], timeperiod=14)
    slowk, slowd = ta.STOCH(
        prices_df['high'],
        prices_df['low'],
        prices_df['close'],
        fastk_period=14,
        slowk_period=3,
        slowk_matype=0,
        slowd_period=3,
        slowd_matype=0
    )
    upper, middle, lower = ta.BBANDS(
        prices_df['close'],
        timeperiod=20,
        nbdevup=2,
        nbdevdn=2,
        matype=0
    )
    cons_lows = calculate_consecutive_lows(prices_df)
    
    # Prepare chart data
    price_data = prepare_chart_data(prices_df, 'close')
    upper_data = prepare_chart_data(upper, 'close')
    middle_data = prepare_chart_data(middle, 'close')
    lower_data = prepare_chart_data(lower, 'close')
    rsi_data = prepare_chart_data(rsi, 'close')
    slowk_data = prepare_chart_data(slowk, 'close')
    slowd_data = prepare_chart_data(slowd, 'close')
    
    # Render charts
    st.subheader(f"Technical Analysis for {symbolsDate_dict['symbols'][0]}")
    
    # Price chart with Bollinger Bands
    st.write("Price with Bollinger Bands")
    renderLightweightCharts([{
        "chart": chart_options,
        "series": [
            {"type": 'Line', "data": price_data, "options": {"color": 'blue'}},
            {"type": 'Line', "data": upper_data, "options": {"color": 'red'}},
            {"type": 'Line', "data": middle_data, "options": {"color": 'gray'}},
            {"type": 'Line', "data": lower_data, "options": {"color": 'red'}}
        ]
    }], 'multi')
    
    # RSI chart
    st.write("RSI")
    renderLightweightCharts([{
        "chart": chart_options,
        "series": [{"type": 'Line', "data": rsi_data, "options": {"color": 'purple'}}]
    }], 'rsi')
    
    # Stochastic chart
    st.write("Stochastic Oscillator")
    renderLightweightCharts([{
        "chart": chart_options,
        "series": [
            {"type": 'Line', "data": slowk_data, "options": {"color": 'blue'}},
            {"type": 'Line', "data": slowd_data, "options": {"color": 'red'}}
        ]
    }], 'stochastic')
    
    # Display metrics
    st.subheader("Key Metrics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Latest Price", f"{prices_df['close'][-1]:.2f}")
    with col2:
        st.metric("Latest RSI", f"{rsi[-1]:.2f}")
    with col3:
        st.metric("Consecutive Lows", f"{cons_lows[-1]:.0f}")

