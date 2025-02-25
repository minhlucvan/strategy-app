import pandas as pd
import numpy as np
import json

import streamlit as st
import vectorbt as vbt

import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots # creating subplots


from studies.stock_news_momentum_study import calculate_price_changes, filter_events, filter_news_by_pe_change, plot_correlation, plot_price_changes_agg, plot_scatter_matrix, run_simulation
from utils.component import  check_password, input_dates, input_SymbolsDate
import matplotlib.pyplot as plt

from utils.plot_utils import plot_events, plot_multi_bar, plot_multi_line, plot_multi_scatter, plot_single_bar
from utils.processing import get_stocks, get_stocks_document, get_stocks_financial
from utils.vbt import plot_pf
from vbt_strategy.MOM_D import get_MomDInd
import utils.fin_utils as fin_utils

from studies.stock_custom_event_study import run as run_custom_event_study

import utils.vietstock as vietstock
import utils.google as google

def calculate_real_ratios(stocks_df, financials_dfs):
    eps_df = pd.DataFrame()
    pe_df = pd.DataFrame()
    pb_df = pd.DataFrame()
    real_pe_df = pd.DataFrame()
    real_pb_df = pd.DataFrame()

    for symbol in financials_dfs:
        financials_df = financials_dfs[symbol].copy()
        financials_df.index = pd.to_datetime(financials_df.index).tz_localize(None)
        financials_df = financials_df[financials_df.index >= stocks_df.index[0]]

        if financials_df.empty:
            continue
        
        if symbol not in stocks_df.columns:
            continue
        
        stock_df = stocks_df[symbol]
        union_df = financials_df.reindex(stock_df.index, method='ffill')
        
        union_df['close'] = stock_df.astype(float)
        union_df['realPriceToEarning'] = union_df['close'] / union_df['earningPerShare']
        union_df['realPriceToBook'] = union_df['close'] / union_df['bookValuePerShare']

        eps_df = pd.concat([eps_df, pd.DataFrame({symbol: union_df['earningPerShare']})], axis=1)
        pe_df = pd.concat([pe_df, pd.DataFrame({symbol: union_df['priceToEarning']})], axis=1)
        pb_df = pd.concat([pb_df, pd.DataFrame({symbol: union_df['priceToBook']})], axis=1)
        real_pe_df = pd.concat([real_pe_df, pd.DataFrame({symbol: union_df['realPriceToEarning']})], axis=1)
        real_pb_df = pd.concat([real_pb_df, pd.DataFrame({symbol: union_df['realPriceToBook']})], axis=1)

    return eps_df, pe_df, pb_df, real_pe_df, real_pb_df


def run(symbol_benchmark, symbolsDate_dict):
    
    if len(symbolsDate_dict['symbols']) < 1:
        st.info("Please select symbols.")
        st.stop()
        
    benchmark_dict = symbolsDate_dict.copy()
    benchmark_dict['symbols'] = [symbol_benchmark]
    
    benchmark_df = get_stocks(benchmark_dict, 'close')
    stocks_df = get_stocks(symbolsDate_dict, 'close')
    stocks_df.index = pd.to_datetime(stocks_df.index).tz_localize(None)

    
    if benchmark_df.empty or stocks_df.empty:
        st.warning("No data available.")
        st.stop()
    
    news_df = get_stocks_document(symbolsDate_dict, 'Title', doc_type='1', group_by_date=True)
    
    # financials_dfs = get_stocks_financial(symbolsDate_dict, raw=True)

    # eps_df, pe_df, pb_df, real_pe_df, real_pb_df = calculate_real_ratios(stocks_df, financials_dfs)
    # st.write(news_df)
    # Localize index to None
    for df in [news_df, stocks_df, benchmark_df]:
        df.index = df.index.tz_localize(None)
    
    # eps_df = eps_df.reindex(news_df.index, method='nearest')
    
    # real_pe_change_df = eps_df.pct_change() * 100
    # st.write(real_pe_change_df)
    
    # if len(news_df.columns) == 1:
    # plot_multi_bar(real_pe_change_df, title="Real PE Change", y_title="Real PE Change", x_title="Date")
        
    st.write("Filter the news by the real PE change")

    # news_df = filter_news_by_pe_change(news_df, real_pe_change_df)
        
    price_changes_flat_df = calculate_price_changes(stocks_df, news_df, lower_bound=-20, upper_bound=20)
    
    st.write("Select the column and threshold to filter the news. negative column means you are looking into the future")
    column = st.selectbox("Select column", price_changes_flat_df.columns, index=0)
    threshold = st.number_input('Threshold', min_value=-5.0, max_value=5.0, value=0.0)
    # no_filter = st.checkbox("No filter")
    
    # threshold = None if no_filter else threshold
    
    news_df, original_news_df  = filter_events(news_df, price_changes_flat_df, threshold, column=column)
    
    show_data = st.checkbox("Show data")
    if show_data:
        display_df = original_news_df[original_news_df.notnull().any(axis=1)]
        st.dataframe(display_df, use_container_width=True)
    
    if len(news_df.columns) == 1:
        plot_events(stocks_df.iloc[:, 0], original_news_df.iloc[:, 0], label="")


    columns = [f"change_{i}" for i in range(-20, 20)]
    plot_correlation(price_changes_flat_df, columns=columns)
    
    # reduce half of the columns, mod 3 == 0
    dims = [f"change_{i}" for i in range(-10, 9) if i % 3 == 0 and i != 0]
    plot_scatter_matrix(price_changes_flat_df, selected_dims=dims)
    
    plot_price_changes_agg(price_changes_flat_df)
    
    enable_simulate = st.checkbox("Enable simulate")
    if enable_simulate:
        run_simulation(symbol_benchmark, symbolsDate_dict, benchmark_df, stocks_df, news_df)

