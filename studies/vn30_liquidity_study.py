
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import utils.algotrade as at

from studies.stock_news_momentum_study import calculate_price_changes
from studies.stock_overnight_study import plot_scatter_2_sources
from utils.plot_utils import plot_double_side_bars, plot_multi_bar, plot_multi_line, plot_single_bar, plot_single_line, plot_multi_scatter, plot_events
from utils.processing import get_stocks, get_stocks_foregin_flow
import utils.plot_utils as pu
import utils.stock_utils as su

def run(symbol_benchmark, symbolsDate_dict):
    
    if len(symbolsDate_dict['symbols']) < 1:
        st.info("Please select symbols.")
        st.stop()
        
    
    stocks_df = get_stocks(symbolsDate_dict, stack=True, benchmark=True, merge_benchmark=True)
    
    
    close_df = stocks_df['close']
    volume_df = stocks_df['volume']
    
    close_vn30_df = close_df[symbol_benchmark]
    volume_vn30_df = volume_df[symbol_benchmark]
    
    close_stocks_df = close_df.drop(columns=[symbol_benchmark])
    volume_stocks_df = volume_df.drop(columns=[symbol_benchmark])
    
    stock_index_df = su.construct_index_df(close_stocks_df)
    
    liquidity_df = volume_vn30_df
    
    smooth_window = st.slider('Smooth Window', 1, 100, 63)
    
    liquidity_sm_df = liquidity_df.rolling(window=smooth_window).mean()
    pu.plot_single_bar_with_price(liquidity_sm_df, title='Liquidity', x_title='Date', y_title='Liquidity', legend_title='Liquidity', price_df=stock_index_df)
