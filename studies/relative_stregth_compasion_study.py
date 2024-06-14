
import streamlit as st



from utils.plot_utils import plot_multi_line
from utils.processing import get_stocks

import numpy as np
import pandas as pd
import talib as ta
import vectorbt as vbt

def calculate_benchmark(stocks_df):
    benchmark = stocks_df.mean(axis=1)
    return benchmark

def calculate_relative_strength(stocks_df, benchmark):
    rsc = stocks_df.div(benchmark, axis=0)
    return rsc

# Define the magic formula function
def magic_formula(metrics):
    value_score = metrics['priceToEarning']
    return value_score

def run(symbol_benchmark, symbolsDate_dict):    
    if len(symbolsDate_dict['symbols']) < 1:
        st.info("Please select symbols.")
        st.stop()
    
    # Assuming get_stocks and get_stocks_funamental are defined elsewhere
    stocks_df = get_stocks(symbolsDate_dict, "close")
    high_df = get_stocks(symbolsDate_dict, "high")
    low_df = get_stocks(symbolsDate_dict, "low")
    # st.write(stocks_df)
    
    # reindex the stocks_df to high_df and low_df
    high_df = high_df.reindex(stocks_df.index, method='nearest')
    low_df = low_df.reindex(stocks_df.index, method='nearest')
    
    benchmart_df = calculate_benchmark(stocks_df)
    # st.write(benchmart_df)
    
    rs_df = calculate_relative_strength(stocks_df, benchmart_df)
    st.write("Relative Strength Comparison")
    plot_multi_line(rs_df, title='Relative Strength Comparison', x_title='Date', y_title='Relative Strength', legend_title='Stocks')
    
    st.write("Relative Return")
    return_df = stocks_df.pct_change().cumsum()
    plot_multi_line(return_df, title='Relative Strength Change', x_title='Date', y_title='Relative Strength Change', legend_title='Stocks')
    
    st.write("RRS")
    return_benchmark_df = benchmart_df.pct_change().cumsum()
    rrs_df = return_df / return_benchmark_df
    plot_multi_line(rrs_df, title='Relative Return Strength', x_title='Date', y_title='Relative Return Strength', legend_title='Stocks')
    
    # st.write(high_df)
    indicator_atr = vbt.ATR.run(high_df, low_df, stocks_df, window=14)
    
    atr = indicator_atr.atr[14]
    
    st.write("Average True Range")
    plot_multi_line(atr, title='Average True Range', x_title='Date', y_title='Average True Range', legend_title='Stocks')
    
    atr_change = atr.pct_change().cumsum()
    st.write("Average True Range Change")
    plot_multi_line(atr_change, title='Average True Range Change', x_title='Date', y_title='Average True Range Change', legend_title='Stocks')