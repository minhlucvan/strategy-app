import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

from utils.component import input_SymbolsDate
from utils.processing import get_stocks, get_stocks_news
from studies.stock_custom_event_study import run as run_custom_event_study

def calculate_price_changes(stocks_df, news_df):
    stocks_df = stocks_df.reindex(news_df.index, method='ffill')
    
    price_change_dfs = {}
    for i in range(-4, 2):
        price_change_df = (stocks_df.shift(i) - stocks_df) / stocks_df
        price_change_dfs[f"change_{i}"] = price_change_df
        
    price_changes_df = pd.concat(price_change_dfs, axis=1)
    price_changes_df.index = news_df.index
    
    price_changes_flat_df = price_changes_df.stack().reset_index()
    price_changes_flat_df = price_changes_flat_df.set_index('date')
    
    return price_changes_flat_df

def filter_events(news_df, price_changes_flat_df, threshold):
    for symbol in news_df.columns.get_level_values(0).unique():
        for index in news_df.index:
            price_change_df = price_changes_flat_df.loc[index]
            if price_change_df['level_1'] == symbol:
                price_change_symbol  = price_change_df
                price_change = price_change_symbol['change_1']
            else:
                price_change_symbol = price_change_df[price_change_df['level_1'] == symbol]
                price_change_1 = price_change_symbol['change_1']
                price_change = price_change_1.values[0] if not price_change_1.empty else np.nan
            if price_change > threshold:
                news_df.loc[index, symbol] = price_change
            else:
                news_df.loc[index, symbol] = np.nan
    return news_df

def plot_correlation(price_changes_flat_df):
    corr_df = price_changes_flat_df[['change_1', 'change_-1']]
    corr = corr_df.corr()
    fig = px.imshow(corr)
    st.plotly_chart(fig, use_container_width=True)

def plot_scatter_matrix(price_changes_flat_df):
    selected_dims = price_changes_flat_df[['change_1', 'change_-1']].columns
    fig = px.scatter_matrix(price_changes_flat_df, dimensions=selected_dims, color='level_1')
    st.plotly_chart(fig, use_container_width=True, height=800)

def run_simulation(symbol_benchmark, symbolsDate_dict, benchmark_df, stocks_df, news_df):
    run_custom_event_study(
        symbol_benchmark,
        symbolsDate_dict,
        benchmark_df=benchmark_df,
        stocks_df=stocks_df,
        events_df=news_df,
        def_days_before=0,
        def_days_after=3
    )

def run(symbol_benchmark, symbolsDate_dict):

    if len(symbolsDate_dict['symbols']) < 1:
        st.info("Please select symbols.")
        st.stop()
        
    benchmark_dict = symbolsDate_dict.copy()
    benchmark_dict['symbols'] = [symbol_benchmark]
    
    benchmark_df = get_stocks(benchmark_dict, 'close')
    stocks_df = get_stocks(symbolsDate_dict, 'close')
    news_df = get_stocks_news(symbolsDate_dict, 'title')
    
    # Localize index to None
    for df in [news_df, stocks_df, benchmark_df]:
        df.index = df.index.tz_localize(None)
        
    price_changes_flat_df = calculate_price_changes(stocks_df, news_df)
    
    threshold = st.number_input('Threshold', min_value=0.0, max_value=5.0, value=0.0)
    news_df = filter_events(news_df, price_changes_flat_df, threshold)
    
    plot_correlation(price_changes_flat_df)
    plot_scatter_matrix(price_changes_flat_df)
    
    enable_simulate = st.checkbox("Enable simulate")
    if enable_simulate:
        run_simulation(symbol_benchmark, symbolsDate_dict, benchmark_df, stocks_df, news_df)

