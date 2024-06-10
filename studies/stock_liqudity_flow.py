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
    
def run(symbol_benchmark, symbolsDate_dict):
    
    if len(symbolsDate_dict['symbols']) < 1:
        st.info("Please select symbols.")
        st.stop()
    
    # copy the symbolsDate_dict
    benchmark_dict = symbolsDate_dict.copy()
    benchmark_dict['symbols'] = [symbol_benchmark]
    
    benchmark_df = get_stocks(benchmark_dict,'close')
    stocks_df = get_stocks(symbolsDate_dict,'close')
   
    liquidity_flow_df = get_stocks(symbolsDate_dict, 'value_change_weighted')
        
    
    # reindex the stocks_df for the liquidity_flow_df
    # stocks_df = stocks_df.reindex(liquidity_flow_df.index)
    
    first_event_date = liquidity_flow_df.index[0]
    
    stocks_df = stocks_df.loc[first_event_date:]
        
    plot_multi_line(stocks_df, title='Stocks Close Price', x_title='Date', y_title='Close Price', legend_title='Stocks')
        
    plot_multi_line(liquidity_flow_df, title='Stocks Foregin Flow', x_title='Date', y_title='Net Foreign Volume', legend_title='Stocks')
    
    liquidity_flow_smth_df = liquidity_flow_df.rolling(window=5).mean()
    
    plot_multi_line(liquidity_flow_smth_df, title='Stocks Foregin Flow Smoothed', x_title='Date', y_title='Net Foreign Volume', legend_title='Stocks')
    
    liquidity_flow_cum_df = liquidity_flow_df.rolling(window=21).sum()
        
    plot_multi_line(liquidity_flow_cum_df, title='Stocks Foregin Flow Cumulative', x_title='Date', y_title='Net Foreign Volume', legend_title='Stocks')