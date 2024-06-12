from datetime import timezone
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

from utils.processing import get_stocks, get_stocks_events, get_stocks_news, get_stocks_valuation
from studies.rrg import plot_RRG, rs_ratio, RRG_Strategy
from utils.vbt import plot_pf
from vbt_strategy.MOM_D import get_MomDInd
import utils.vietstock as vietstock
import utils.plot_utils as plot_utils
from studies.stock_custom_event_study import run as run_custom_event_study

import utils.vnstock as vnstock

def run(symbol_benchmark, symbolsDate_dict):
    
    if len(symbolsDate_dict['symbols']) < 1:
        st.info("Please select symbols.")
        st.stop()
    
    # copy the symbolsDate_dict
    benchmark_dict = symbolsDate_dict.copy()
    benchmark_dict['symbols'] = [symbol_benchmark]
        
    benchmark_df = get_stocks(benchmark_dict,'close')
    stocks_df = get_stocks(symbolsDate_dict,'close')
    # value_change_weighted_df = get_stocks(symbolsDate_dict, 'volume_weighted')
    news_df = get_stocks_news(symbolsDate_dict, 'title')
        
    news_df.index = news_df.index.tz_localize(None)
    stocks_df.index = stocks_df.index.tz_localize(None)
    benchmark_df.index = benchmark_df.index.tz_localize(None)
    # value_change_weighted_df.index = value_change_weighted_df.index.tz_localize(None)
    
    # reindex stocks_df to news_df ffilled
    stocks_df = stocks_df.reindex(news_df.index, method='ffill')
        
    price_change_dfs = {}
    
    for i in range(-4, 2):
        price_change_df = (stocks_df.shift(i) - stocks_df) / stocks_df
        price_change_dfs[f"change_{i}"] = price_change_df
        
    price_changes_df = pd.concat(price_change_dfs, axis=1)
    
    # reindex to news_df
    price_changes_df.index = news_df.index
        
    price_changes_flat_df = price_changes_df.stack().reset_index()
    
    # set index to date
    price_changes_flat_df = price_changes_flat_df.set_index('date')
    
    threshold = st.number_input('Threshold', min_value=0.0, max_value=5.0, value=0.0)
    
    # filter events
    for symbol in news_df.columns.get_level_values(0).unique():
        for index in news_df.index:
            price_change_df = price_changes_flat_df.loc[index]
            price_change_symbol = price_change_df[price_change_df['level_1'] == symbol]
            price_change = price_change_symbol['change_1'].values[0]
            if price_change > threshold:
                news_df.loc[index][symbol] = price_change
            else:
                news_df.loc[index][symbol] = np.nan
        
    # plot correlation
    corr_df = price_changes_flat_df.copy()
    corr_df = corr_df[['change_1', 'change_-1']]
    corr = corr_df.corr()
    fig = px.imshow(corr)
    st.plotly_chart(fig, use_container_width=True)
    
    # plot scatter matrix
    selected_dims = corr_df.columns
    
    fig = px.scatter_matrix(price_changes_flat_df,
        dimensions=selected_dims,
        # color by ticker
        color='level_1',
    )
    st.plotly_chart(fig, use_container_width=True, height=800)
    
    # plot_utils.plot_events(stocks_df[symbol], news_df_positive['title'], label="")
    
                
    enable_simulate = st.checkbox("Enable simulate")
    
    if enable_simulate:
        run_custom_event_study(symbol_benchmark,
            symbolsDate_dict,
            benchmark_df=benchmark_df,
            stocks_df=stocks_df,
            events_df=news_df,
            def_days_before=0,
            def_days_after=3)
    