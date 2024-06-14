
import streamlit as st
import pandas as pd
import plotly.express as px

from studies.stock_overnight_study import plot_scatter_2_sources
from utils.plot_utils import plot_multi_bar, plot_multi_line, plot_single_bar, plot_single_line, plot_multi_scatter
from utils.processing import get_stocks, get_stocks_foregin_flow


    
def run(symbol_benchmark, symbolsDate_dict):
    
    if len(symbolsDate_dict['symbols']) < 1:
        st.info("Please select symbols.")
        st.stop()
    
    # copy the symbolsDate_dict
    benchmark_dict = symbolsDate_dict.copy()
    benchmark_dict['symbols'] = [symbol_benchmark]
    
    benchmark_df = get_stocks(benchmark_dict,'close')
    closes_df = get_stocks(symbolsDate_dict,'close')
    opens_df = get_stocks(symbolsDate_dict,'open')
    
    gap_pct = st.slider('Gap Percentage', min_value=0.0, max_value=0.2, value=0.07, step=0.01)
    days_after = st.slider('Days After', min_value=1, max_value=30, value=3, step=1)
    
    gaps_df = (opens_df - closes_df.shift(1)) / closes_df.shift(1)
    
    opens_future_df = opens_df.shift(-days_after)
    
    opens_change_df = (opens_future_df - opens_df) / opens_df
    
    # filter the gaps_df > 0.02 or < -0.02
    gaps_df = gaps_df[(gaps_df < -gap_pct)]
    
    # filter the opens_change_df where gaps_df is not NaN
    opens_change_df = opens_change_df[gaps_df.notna()]
    
    plot_multi_bar(gaps_df, title='Stocks Gaps', x_title='Date', y_title='Gap', legend_title='Stocks')
    
    plot_multi_line(closes_df, title='Stocks Prices', x_title='Date', y_title='Price', legend_title='Stocks')
        
    # gaps_down_df = gaps_df[gaps_df < -0.07]

    # gaps_df = gaps_down_df

    # Prepare data for plotting
    # Flatten the DataFrames to make them suitable for Plotly

    plot_scatter_2_sources(gaps_df, opens_change_df, title='Correlation between Price Gaps and Changes', x_title='Price Gap', y_title='Price Change after 3 Days', legend_title='Ticker')
    
    # plot mean price change after
    opens_change_mean_df = opens_change_df.mean(axis=1)
    plot_single_bar(opens_change_mean_df, title='Mean Price Change', x_title='Date', y_title='Price Change', legend_title='Mean')
    
    opens_change_daily_df = opens_change_df.mean(axis=1)
    opens_change_cumsum_df = opens_change_daily_df.cumsum()
    plot_single_line(opens_change_cumsum_df, title='Cumulative Price Change', x_title='Date', y_title='Price Change', legend_title='Stocks')
    
    # plot mean price gap
    gaps_mean = opens_change_mean_df.mean() * 100
    
    st.write(f'Mean Price Change: {gaps_mean}')
    
    total_signal = len(gaps_df)
    st.write(f'Total Signal: {total_signal}')
    
    final_return = opens_change_cumsum_df.dropna().iloc[-1]
    st.write(f'Final Return: {final_return}')
    
    start_date = gaps_df.index[0]
    end_date = gaps_df.index[-1]
    
    annual_return = final_return / total_signal * 252 / (end_date - start_date).days
    st.write(f'Annual Return: {annual_return}')
    
    