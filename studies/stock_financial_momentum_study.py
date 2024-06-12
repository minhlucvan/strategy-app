import pandas as pd
import numpy as np
import json

import streamlit as st
import vectorbt as vbt

import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots # creating subplots


from studies.stock_news_momentum_study import calculate_price_changes, filter_events, plot_correlation, plot_scatter_matrix, run_simulation
from utils.component import  check_password, input_dates, input_SymbolsDate
import matplotlib.pyplot as plt

from utils.plot_utils import plot_events
from utils.processing import get_stocks, get_stocks_document
from studies.rrg import plot_RRG, rs_ratio, RRG_Strategy
from utils.vbt import plot_pf
from vbt_strategy.MOM_D import get_MomDInd

from studies.stock_custom_event_study import run as run_custom_event_study

import utils.vietstock as vietstock
import utils.google as google

def run(symbol_benchmark, symbolsDate_dict):
    
    if len(symbolsDate_dict['symbols']) < 1:
        st.info("Please select symbols.")
        st.stop()
        
    benchmark_dict = symbolsDate_dict.copy()
    benchmark_dict['symbols'] = [symbol_benchmark]
    
    benchmark_df = get_stocks(benchmark_dict, 'close')
    stocks_df = get_stocks(symbolsDate_dict, 'close')
    
    if benchmark_df.empty or stocks_df.empty:
        st.warning("No data available.")
        st.stop()
    
    news_df = get_stocks_document(symbolsDate_dict, 'Title', doc_type='1')
    
    # Localize index to None
    for df in [news_df, stocks_df, benchmark_df]:
        df.index = df.index.tz_localize(None)
        
    price_changes_flat_df = calculate_price_changes(stocks_df, news_df)
    
    st.write("Select the column and threshold to filter the news. negative column means you are looking into the future")
    column = st.selectbox("Select column", price_changes_flat_df.columns, index=1)
    threshold = st.number_input('Threshold', min_value=-5.0, max_value=5.0, value=0.0)
    
    news_df, original_news_df  = filter_events(news_df, price_changes_flat_df, threshold, column=column)
    
    show_data = st.checkbox("Show data")
    if show_data:
        display_df = original_news_df[original_news_df.notnull().all(axis=1)]
        st.dataframe(display_df, use_container_width=True)
    
    if len(news_df.columns) == 1:
        plot_events(stocks_df.iloc[:, 0], original_news_df.iloc[:, 0], label="")
        
    plot_correlation(price_changes_flat_df)
    plot_scatter_matrix(price_changes_flat_df)
    
    enable_simulate = st.checkbox("Enable simulate")
    if enable_simulate:
        run_simulation(symbol_benchmark, symbolsDate_dict, benchmark_df, stocks_df, news_df)

