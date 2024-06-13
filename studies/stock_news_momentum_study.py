import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

from utils.component import input_SymbolsDate
from utils.plot_utils import plot_events, plot_single_bar
from utils.processing import get_stocks, get_stocks_news
from studies.stock_custom_event_study import run as run_custom_event_study

def calculate_price_changes(stocks_df, news_df, lower_bound=-4, upper_bound=2):
    stocks_df = stocks_df.reindex(news_df.index, method='ffill')
    
    price_change_dfs = {}
    
    for i in range(lower_bound, upper_bound):
        price_change_df = (stocks_df.shift(i) - stocks_df) / stocks_df
        
        # set type to float
        price_change_df = price_change_df.astype(float)
        
        price_change_dfs[f"change_{i}"] = price_change_df
        
    price_changes_df = pd.concat(price_change_dfs, axis=1)
    price_changes_df.index = news_df.index
    
    price_changes_flat_df = price_changes_df.stack().reset_index()
    price_changes_flat_df = price_changes_flat_df.set_index('level_0')
    
    return price_changes_flat_df

def filter_news_by_pe_change(news_df, pe_change_df):
    news_df = news_df.copy()
    for symbol in news_df.columns.get_level_values(0).unique():
        for index in news_df.index:
            if symbol not in pe_change_df.columns:
                news_df.loc[index, symbol] = np.nan
                continue
            real_pe_change = pe_change_df.loc[index, symbol]
            if real_pe_change < 0.0:
                news_df.loc[index, symbol] = np.nan
                # st.write(f"Filtering {symbol} at {index}")
    return news_df

def filter_events(news_df, price_changes_flat_df, threshold=0.0, column='change_1', text_filter=""):
    
    original_news = news_df.copy()
    for symbol in news_df.columns.get_level_values(0).unique():
        for index in news_df.index:
            price_change_df = price_changes_flat_df.loc[index]
            if threshold is None or column == 'level_1':
                price_change = 0
            elif isinstance(price_change_df['level_1'], str) and price_change_df['level_1'] == symbol:
                price_change_symbol  = price_change_df
                price_change = price_change_symbol[column]
            else:
                price_change_symbol = price_change_df[price_change_df['level_1'] == symbol]
                price_change_1 = price_change_symbol[column]
                price_change = price_change_1.values[0] if not price_change_1.empty else np.nan
            
            if price_change >= threshold:
                news_df.loc[index, symbol] = price_change
            else:
                news_df.loc[index, symbol] = np.nan
                original_news.loc[index, symbol] = np.nan
            
            if text_filter and text_filter != "" and isinstance(original_news.loc[index, symbol], str):
                slugs = text_filter.split(",")
                
                for slug in slugs:
                    if slug in original_news.loc[index, symbol]:
                        continue
                
                news_df.loc[index, symbol] = np.nan
                original_news.loc[index, symbol] = np.nan
                
                
    return news_df, original_news

def plot_correlation(price_changes_flat_df, columns=['change_1', 'change_-1']):
    corr_df = price_changes_flat_df[columns]
    corr = corr_df.corr()
    fig = px.imshow(corr)
    fig.update_layout(width=600, height=600)
    st.plotly_chart(fig, use_container_width=True)

def plot_scatter_matrix(price_changes_flat_df, selected_dims=['change_1', 'change_-1']):
    fig = px.scatter_matrix(price_changes_flat_df, dimensions=selected_dims, color='level_1')
    fig.update_layout(width=600, height=600)
    st.plotly_chart(fig, use_container_width=True, height=800)
    
def plot_price_changes_agg(price_changes_flat_df):
    price_changes_agg_df = price_changes_flat_df.mean(axis=0, numeric_only=True)
    # rename the index change_i to i
    price_changes_agg_df.index = price_changes_agg_df.index.map(lambda x: x.split('_')[1])
    # convert index to int
    price_changes_agg_df.index = price_changes_agg_df.index.astype(int)
    
    # invert the index i = -i
    price_changes_agg_df.index = price_changes_agg_df.index * -1
    
    plot_single_bar(price_changes_agg_df, title="Price Changes", y_title="Price Change", x_title="Date")
    

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
    
    if benchmark_df.empty or stocks_df.empty:
        st.warning("No data available.")
        st.stop()
    
    news_df = get_stocks_news(symbolsDate_dict, 'title')
    
    # Localize index to None
    for df in [news_df, stocks_df, benchmark_df]:
        df.index = df.index.tz_localize(None)
        
    price_changes_flat_df = calculate_price_changes(stocks_df, news_df)
    
    threshold = st.number_input('Threshold', min_value=0.0, max_value=5.0, value=0.0)
    news_df, original_news_df  = filter_events(news_df, price_changes_flat_df, threshold)
    
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

