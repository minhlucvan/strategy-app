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

from utils.processing import get_stocks, get_stocks_events, get_stocks_valuation
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
    
    symbol = symbolsDate_dict['symbols'][0]
    
    benchmark_df = get_stocks(benchmark_dict,'close')
    stocks_df = get_stocks(symbolsDate_dict,'close')
    value_change_weighted_df = get_stocks(symbolsDate_dict, 'price_change')
    news_df = vietstock.get_stock_news_all(symbol)    
    
    news_df.index = news_df.index.tz_localize(None)
    stocks_df.index = stocks_df.index.tz_localize(None)
    value_change_weighted_df.index = value_change_weighted_df.index.tz_localize(None)
    
    # reindex stocks_df to news_df ffilled
    stocks_df = stocks_df.reindex(news_df.index, method='ffill')
    
    # reindex value_change_weighted_df to news_df ffilled
    value_change_weighted_df = value_change_weighted_df.reindex(news_df.index)
    
    value_change_weighted_2d_df = value_change_weighted_df.rolling(window=2).sum()
    value_change_weighted_3d_df = value_change_weighted_df.rolling(window=3).sum()
    value_change_weighted_4d_df = value_change_weighted_df.rolling(window=4).sum()
    value_change_weighted_5d_df = value_change_weighted_df.rolling(window=5).sum()
    value_change_weighted_6d_df = value_change_weighted_df.rolling(window=6).sum()
    value_change_weighted_7d_df = value_change_weighted_df.rolling(window=7).sum()
    value_change_weighted_8d_df = value_change_weighted_df.rolling(window=8).sum()
    value_change_weighted_9d_df = value_change_weighted_df.rolling(window=9).sum()
    value_change_weighted_10d_df = value_change_weighted_df.rolling(window=10).sum()
    
    price_forward_df = stocks_df.shift(-3)
    price_change_forward_df = (price_forward_df - stocks_df) / stocks_df
    
    # calculate the price change for the event
    for index, row in news_df.iterrows():
        news_df.loc[index, 'price_change'] = value_change_weighted_df.loc[index, symbol]
        news_df.loc[index, 'price_change_2d'] = value_change_weighted_2d_df.loc[index, symbol]
        news_df.loc[index, 'price_change_3d'] = value_change_weighted_3d_df.loc[index, symbol]
        news_df.loc[index, 'price_change_4d'] = value_change_weighted_4d_df.loc[index, symbol]
        news_df.loc[index, 'price_change_5d'] = value_change_weighted_5d_df.loc[index, symbol]
        news_df.loc[index, 'price_change_6d'] = value_change_weighted_6d_df.loc[index, symbol]
        news_df.loc[index, 'price_change_7d'] = value_change_weighted_7d_df.loc[index, symbol]
        news_df.loc[index, 'price_change_8d'] = value_change_weighted_8d_df.loc[index, symbol]
        news_df.loc[index, 'price_change_9d'] = value_change_weighted_9d_df.loc[index, symbol]
        news_df.loc[index, 'price_change_10d'] = value_change_weighted_10d_df.loc[index, symbol]
        
        
        news_df.loc[index, 'price_change_forward'] = price_change_forward_df.loc[index, symbol]
    
    news_df_positive = news_df[news_df['price_change_forward'] > 0]
    
    st.write(news_df_positive)
    
    # plot correlation
    corr_df = news_df.copy()
    corr_df = corr_df.drop(columns='title')
    corr = corr_df.corr()
    fig = px.imshow(corr)
    st.plotly_chart(fig, use_container_width=True)
    
    plot_utils.plot_events(stocks_df[symbol], news_df_positive['title'], label="")

    # run_custom_event_study(symbol_benchmark,
    #     symbolsDate_dict,
    #     benchmark_df=benchmark_df,
    #     stocks_df=stocks_df,
    #     events_df=news_df,
    #     def_days_before=0,
    #     def_days_after=6)
   