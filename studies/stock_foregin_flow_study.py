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

from utils.plot_utils import plot_multi_bar, plot_multi_line, plot_single_bar, plot_single_line, plot_multi_scatter
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
   
    foregin_flow_df = get_stocks_foregin_flow(symbolsDate_dict, 'netForeignVol')
    
    # reindex the stocks_df for the foregin_flow_df
    # stocks_df = stocks_df.reindex(foregin_flow_df.index)
    
    first_event_date = foregin_flow_df.index[0]
    
    stocks_df = stocks_df.loc[first_event_date:]
    benchmark_df = benchmark_df.loc[first_event_date:]
    
        
    plot_multi_bar(foregin_flow_df, title='Stocks Foregin Flow', x_title='Date', y_title='Net Foreign Volume', legend_title='Stocks')
    
    foregin_flow_smth_df = foregin_flow_df.rolling(window=5).mean()
    plot_multi_line(foregin_flow_smth_df, title='Stocks Foregin Flow Smoothed', x_title='Date', y_title='Net Foreign Volume', legend_title='Stocks')
    
    plot_multi_line(stocks_df, title='Stocks Prices', x_title='Date', y_title='Price', legend_title='Stocks')
    
    
    foregin_flow_total_df = foregin_flow_df.sum(axis=1)
    plot_single_line(foregin_flow_total_df, title='Total Foregin Flow', x_title='Date', y_title='Net Foreign Volume', legend_title='Total')
    

    plot_multi_line(benchmark_df, title='Benchmark Prices', x_title='Date', y_title='Price', legend_title='Benchmark')
    
    foregin_flow_total_change_df = foregin_flow_total_df.pct_change()
    plot_single_bar(foregin_flow_total_change_df, title='Total Foregin Flow Change', x_title='Date', y_title='Net Foreign Volume', legend_title='Total')
    
    benchmark_change_df = benchmark_df.pct_change()
    plot_multi_bar(benchmark_change_df, title='Benchmark Prices Change', x_title='Date', y_title='Price Change', legend_title='Benchmark')
    
    benchmark_change_udf = benchmark_change_df.copy()
    benchmark_change_udf['flow_change'] = foregin_flow_total_change_df / 300
    
    benchmark_change_sum_df = benchmark_change_udf.cumsum()
    
    plot_multi_line(benchmark_change_sum_df, title='Benchmark Prices Change vs Foregin Flow Change', x_title='Date', y_title='Price Change', legend_title='Benchmark')
    
    stocks_next_day_df = stocks_df.shift(-1)
    stocks_change_df = (stocks_next_day_df - stocks_df) / stocks_df
    
    foregin_flow_change_df = foregin_flow_df.pct_change()
    
    ratio_df = stocks_change_df / foregin_flow_change_df
        
    # scatter ratio 
    plot_multi_scatter(ratio_df, title='Stocks Price Change vs Foregin Flow Change', x_title='Price Change', y_title='Net Foreign Volume Change', legend_title='Stocks')
    
    
    ratio_mean_df = ratio_df.mean(axis=0)
    
    plot_single_bar(ratio_mean_df, title='Stocks Price Change vs Foregin Flow Change Mean', x_title='Date', y_title='Ratio', legend_title='Mean')
    
    