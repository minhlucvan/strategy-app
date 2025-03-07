import pandas as pd
import numpy as np
import json

import streamlit as st
import vectorbt as vbt

import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots # creating subplots


from utils.component import  check_password, input_dates, input_SymbolsDate
from utils.plot_utils import plot_multi_line
from utils.processing import get_stocks, get_stocks_events, get_stocks_foregin_flow, get_stocks_valuation
from utils.vbt import plot_pf
from vbt_strategy.MOM_D import get_MomDInd

from studies.market_wide import MarketWide_Strategy

    
def run(symbol_benchmark, symbolsDate_dict):    
    with st.expander("Liquidity lock study"):
        st.write("This section analyzes the behavior of the liquidity breakout of the selected stocks and examines the price movements following the breakout.")
    
    if len(symbolsDate_dict['symbols']) < 1:
        st.info("Please select symbols.")
        st.stop()
    
    benchmark_df = get_stocks(symbolsDate_dict,'close', benchmark=True)[symbol_benchmark]
    stocks_df = get_stocks(symbolsDate_dict,'close')
    
    stocks_volume_df = get_stocks(symbolsDate_dict, 'volume')
    stocks_change_df = stocks_df.pct_change()
    
    liquidity_df = stocks_change_df * stocks_volume_df
                    
    # plot liquidity and price
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02)
    for symbol in stocks_df.columns:
        fig.add_trace(go.Scatter(x=stocks_df.index, y=stocks_df[symbol], mode='lines', name=symbol), row=1, col=1)
    for symbol in liquidity_df.columns:
        fig.add_trace(go.Scatter(x=liquidity_df.index, y=liquidity_df[symbol], mode='lines', name=symbol), row=2, col=1)
    fig.update_layout(title_text="Stocks Price and Liquidity", height=600)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Liquidity", row=2, col=1)
    st.plotly_chart(fig)
    
    # Volume breakout detection
    window = st.slider("Select the window size for rolling avg volume", 1, 100, 30)
    volume_z_score_df = (stocks_volume_df - stocks_volume_df.rolling(window=window).mean()) / stocks_volume_df.rolling(window=window).std()
    liquidity_threshold = st.slider("Select volume z-score threshold", 0.0, 5.0, 2.0)
    volume_breakout_df = volume_z_score_df > liquidity_threshold

    # Price change over next 3 days
    stats_period = st.slider("Select period for stats", 1, 10, 3)
    price_change_df = stocks_df.pct_change().shift(-1).rolling(stats_period).sum()

    # Metrics
    total_signals = volume_breakout_df.sum().sum()
    accuracy = (volume_breakout_df & (price_change_df > 0)).sum() / volume_breakout_df.sum()
    avg_price_change = price_change_df[volume_breakout_df].mean()
    transaction_cost = 0.0016
    profitable = avg_price_change.mean() - transaction_cost

    # Display
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total signals", int(total_signals))
    with col2:
        st.metric("Accuracy", f"{accuracy.mean() * 100:.2f}%")
    with col3:
        st.metric("Avg price change", f"{avg_price_change.mean() * 100:.2f}%")
    st.write(f"Average profitable: {profitable * 100:.2f}%")
    
