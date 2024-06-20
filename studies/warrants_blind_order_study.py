import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import utils.algotrade as at

from studies.stock_news_momentum_study import calculate_price_changes
from studies.stock_overnight_study import plot_scatter_2_sources
from utils.plot_utils import plot_double_side_bars, plot_multi_bar, plot_multi_line, plot_single_bar, plot_single_line, plot_multi_scatter, plot_events
from utils.processing import get_stocks, get_stocks_foregin_flow
import utils.stock_utils as su

import plotly.graph_objects as go
import streamlit as st
from studies.stock_gaps_recover_study import run as stock_gaps_recover_study

@st.cache_data
def fetch_warrants_data():
    data =  su.get_warrants_data()

    data_df = pd.DataFrame(data)

    # keep the first word of the period
    data_df['period'] = data_df['period'].str.split().str[0]

    # convert to date
    data_df['listedDate'] = pd.to_datetime(data_df['listedDate'])
    data_df['issuedDate'] = pd.to_datetime(data_df['issuedDate'])
    data_df['expiredDate'] = pd.to_datetime(data_df['expiredDate'])

    return data_df

@st.cache_data
def fetch_data():
    warrants_df = fetch_warrants_data()
    
    warrants_df['days_to_expire'] = (warrants_df['expiredDate'] - pd.Timestamp.today()).dt.days
    
    # filter out the expired warrants > 60
    
    warrants_df = warrants_df[warrants_df['days_to_expire'] > 30]
    
    tickers = warrants_df['cw'].unique()
    
    warrants_intraday_df = su.get_last_trading_history(tickers=tickers)
    
    return warrants_df, warrants_intraday_df

def run(symbol_benchmark, symbolsDate_dict):
    
    # copy the symbolsDate_dict
    # benchmark_dict = symbolsDate_dict.copy()
    warrants_df, warrants_intraday_df = fetch_data()

    warrants_intraday_df['value'] = warrants_intraday_df['volume'] * warrants_intraday_df['close']
    
    value_filter = st.slider('Value Filter', min_value=0, max_value=1_000_000_000, value=100_000_000, step=1_000, format="%d")
    
    warrants_intraday_value_df = warrants_intraday_df[warrants_intraday_df['value'] > value_filter]
    
    tickers = warrants_intraday_value_df.index.get_level_values(0).unique().values.tolist()
        
    st.write(f"Number of Warrants: {len(tickers)}")
    
    select_all = st.checkbox('Select All')
    
    selected_tickers = st.multiselect('Select Tickers', tickers, default=tickers if select_all else [])
    
        
    if len(selected_tickers) < 1:
        st.info("Please select symbols.")
        st.stop()
    
    symbolsDate_dict['symbols'] = selected_tickers
    
    
    closes_df = get_stocks(symbolsDate_dict, 'close', stock_type='warrant')
    
    st.write(closes_df)
    
    
    plot_multi_line(closes_df, title="Warrants Close Price")
    
    warrants_selected_df = warrants_df[warrants_df['cw'].isin(selected_tickers)]
    
    stocks_tickers = warrants_selected_df['underlyingStock'].unique()
    
    symbolsDate_dict_copy = symbolsDate_dict.copy()
    symbolsDate_dict_copy['symbols'] = stocks_tickers
    
    stocks_df = get_stocks(symbolsDate_dict_copy, 'close')
    
    first_date = closes_df.index[0]
    
    stocks_df = stocks_df[stocks_df.index >= first_date]
    
    
    stocks_mapped_df = pd.DataFrame()
    
    for warrant in selected_tickers:
        # CABCXXX => ABC
        stock_ticker = warrant[1:4]
        stock_df = stocks_df[stock_ticker]
        # rename the column to warrant
        stock_df.name = warrant
        
        stocks_mapped_df = pd.concat([stocks_mapped_df, stock_df], axis=1)
        
    plot_multi_line(stocks_mapped_df, title="Stocks Close Price")
    
    
    price_ratio_df = stocks_mapped_df / closes_df
    
    plot_multi_line(price_ratio_df, title="Price Ratio", x_title="Date", y_title="Price Ratio", legend_title="Warrants")
    
    price_ratio_change_df = price_ratio_df.pct_change()
    
    plot_multi_bar(price_ratio_change_df, title="Price Ratio Change", x_title="Date", y_title="Price Ratio Change", legend_title="Warrants")
    
    acc_price_ratio_change_df = price_ratio_change_df.cumsum()
    
    plot_multi_line(acc_price_ratio_change_df, title="Accumulated Price Ratio Change", x_title="Date", y_title="Accumulated Price Ratio Change", legend_title="Warrants")