import pandas as pd
import numpy as np
import json

import streamlit as st
import vectorbt as vbt

import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots # creating subplots


from utils.component import  check_password, input_dates, input_SymbolsDate
import matplotlib.pyplot as plt

from utils.plot_utils import plot_multi_line
from utils.processing import get_stocks, get_stocks_events, get_stocks_foregin_flow, get_stocks_valuation
from studies.rrg import plot_RRG, rs_ratio, RRG_Strategy
from utils.vbt import plot_pf
from vbt_strategy.MOM_D import get_MomDInd

from studies.market_wide import MarketWide_Strategy

def plot_liquidity_bars(liquidity_series, title, x_title, y_title, legend_title):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=liquidity_series.index, y=liquidity_series, name=legend_title))
    # color green if positive, red if negative
    fig.update_traces(marker_color=['green' if x >= 0 else 'red' for x in liquidity_series])
    fig.update_layout(title=title, xaxis_title=x_title, yaxis_title=y_title)
    st.plotly_chart(fig)
    
def run(symbol_benchmark, symbolsDate_dict):
    
    if len(symbolsDate_dict['symbols']) < 1:
        st.info("Please select symbols.")
        st.stop()
    
    # copy the symbolsDate_dict
    benchmark_dict = symbolsDate_dict.copy()
    benchmark_dict['symbols'] = [symbol_benchmark]
    
    benchmark_df = get_stocks(benchmark_dict,'close')
    stocks_df = get_stocks(symbolsDate_dict,'close')
   
    liquidity_flow_df = get_stocks(symbolsDate_dict, 'value')
        
    
    # reindex the stocks_df for the liquidity_flow_df
    # stocks_df = stocks_df.reindex(liquidity_flow_df.index)
    
    window = st.slider('Rolling window', min_value=1, max_value=100, value=10)
    
    first_event_date = liquidity_flow_df.index[0]
    
    stocks_df = stocks_df.loc[first_event_date:]
    
    liquidity_flow_df = liquidity_flow_df.rolling(window=window).mean()
        
    plot_multi_line(stocks_df, title='Stocks Close Price', x_title='Date', y_title='Close Price', legend_title='Stocks')
        
    plot_multi_line(liquidity_flow_df, title='Stocks Liquidity Flow', x_title='Date', y_title='Value', legend_title='Stocks')
    
    liquidity_change_flow_df = get_stocks(symbolsDate_dict, 'value_change_weighted')

    liquidity_change_flow_df = liquidity_change_flow_df.rolling(window=window).sum()
    
    plot_multi_line(liquidity_change_flow_df, title='Stocks Foregin Flow Change', x_title='Date', y_title='Value change', legend_title='Stocks')
    
    for stock in liquidity_change_flow_df.columns:
        liquidity_df = liquidity_change_flow_df[stock]
        
        plot_liquidity_bars(liquidity_df, title=f'{stock} Liquidity Flow', x_title='Date', y_title='Value', legend_title='Stocks')