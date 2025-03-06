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
    with st.expander("Liquidity breakout study"):
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
    
    window = st.slider("Select the window size for the rolling z-score", 1, 100, 14)
    liquidity_z_score_df = liquidity_df.rolling(window).apply(lambda x: (x[-1] - x.mean()) / x.std())
    
    if len(symbolsDate_dict['symbols']) == 1:
        # plot liquidity z-score
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02)
        for symbol in stocks_df.columns:
            fig.add_trace(go.Scatter(x=stocks_df.index, y=stocks_df[symbol], mode='lines', name=symbol), row=1, col=1)
        for symbol in liquidity_z_score_df.columns:
            fig.add_trace(go.Scatter(x=liquidity_z_score_df.index, y=liquidity_z_score_df[symbol], mode='lines', name=symbol), row=2, col=1)
        fig.update_layout(title_text="Stocks Price and Liquidity Z-Score", height=600)
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Liquidity Z-Score", row=2, col=1)
        st.plotly_chart(fig)
        
    liquidity_threshold = st.slider("Select the liquidity z-score threshold", 0.0, 5.0, 2.0)
    
    liquidity_breakout_df = liquidity_z_score_df > liquidity_threshold
    
    # analyze signals accuracy by comparing with price movements of the day after
    stats_period = st.slider("Select the period for calculating the stats", 1, 10, 3)
    # Calculate the percentage change between today and `stats_period` days later
    price_change_df = (stocks_df.shift(-stats_period) - stocks_df) / stocks_df

    price_change_df = price_change_df[liquidity_breakout_df.columns]
    
    # caculate the accuracy of the signals
    accuracy_df = ((liquidity_breakout_df & (price_change_df > 0)).count() / liquidity_breakout_df.count())
    
    # calculate the average price change following the signals
    price_change_mean_df = price_change_df[liquidity_breakout_df].mean()
    
    # stats
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(label="Total signals", value=int(liquidity_breakout_df.sum().sum()))
    
    with col2:
        st.metric(label="Signals accuracy", value=f"{accuracy_df.mean() * 100:.2f}%")
    
    with col3:
        st.metric(label="Average price change", value=f"{price_change_mean_df.mean() * 100:.2f}%")
        
    # profitable = mean price change - transaction cost 
    # transaction cost = 0.16% (average for VN30 stocks)
    transaction_cost = 0.0016
    profitable_df = price_change_mean_df.mean() - transaction_cost
    
    st.write(f"Average profitable: {profitable_df * 100:.2f}%")