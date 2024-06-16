import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
from studies.cash_flow_study import add_price_changes_to_df, calculate_price_changes, filter_prices, plot_correlation_matrix, plot_scatter_matrix, prepare_dims_df
from utils.plot_utils import plot_multi_bar, plot_multi_line, plot_multi_scatter, plot_single_bar, plot_single_line
from utils.processing import get_stocks
import utils.stock_utils as stock_utils


def display_description():
    with st.expander("Description"):
        st.markdown("""    
- "bp": 0.418,                 // buyup percent
- "op": 0.500,                 // over percent
- "p": 23268.00,               // Price
- "pp": 0.025,                 // price percent
- "vol": 4913057,              // Volume
- "val": 135520044300,         // Value
- "mc": 107588194916600.00,    // Market Capitalization
- "nstp": 4.200,               // Performance indicator (e.g., Net Profit Margin)
- "peravgVolume5d": 0.86,      // Average volume over 5 days
- "rsi14": 56.04,              // RSI (14)
- "rs3d": 57.00,               // Relative Strength Index over 3 days
- "avgrs": 56.00,              // Average relative strength
- "pPerMa5": 0.84,             // Price/MA(5)
- "pPerMa20": 0.86,            // Price/MA(20)
- "isignal": 0.000,            // Signal indicator
- "fnetVol": -55653.00,        // Foreign net volume
- "t": "10/05/24",             // Date
- "seq": 1715299200            // Sequence number or timestamp""")
        

def get_stocks_symbols(symbolsDate_dict, symbol):
    symbolsDate_dict_cp = symbolsDate_dict.copy()
    symbolsDate_dict_cp['symbols'] = [symbol]
    
    return symbolsDate_dict_cp

def run(symbol_benchmark, symbolsDate_dict):
    symbol_benchmark = 'VN30F1M'
    timeframes = st.selectbox('Select timeframes', ['1Y', '1M', '1W', '1D'])
    
    price_timeframe = 'D'
    
    if timeframes == '1D':
        price_timeframe = '5'
    elif timeframes == '1W':
        price_timeframe = '60'

    # prices_df = get_stocks(symbolsDate_dict, 'close')
    data = stock_utils.get_intraday_cash_flow_list_all(tickers=symbolsDate_dict['symbols'], timeFrame=timeframes)
    df = stock_utils.load_intraday_cash_flow_latest_to_dataframe(data, timeFrame=timeframes)
    
    benchmark_df = get_stocks(get_stocks_symbols(symbolsDate_dict, symbol_benchmark), 'close', timeframe=price_timeframe)
   
   
    benchmark_df = filter_prices(benchmark_df, df)
    
    # prices_df = filter_prices(prices_df, df)
        
    benchmark = benchmark_df[symbol_benchmark]
    
    
    df = df.groupby(df.index).mean(numeric_only=True)
    # st.write(df)
    
    for col in df.columns:
        plot_single_bar(df[col], title=col, x_title='Date', y_title=col, legend_title='Stocks', price_df=benchmark)
   
    