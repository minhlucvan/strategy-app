import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from time import sleep
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pytz import UTC  # Import the UTC time zone
from utils.plot_utils import plot_multi_bar, plot_multi_line, plot_multi_scatter
from utils.processing import get_stocks
import utils.stock_utils as stock_utils

def run(symbol_benchmark, symbolsDate_dict):
    timeframes = st.selectbox('Select timeframes', ['1Y', '1M', '1W', '1D'])
    
    prices_df = get_stocks(symbolsDate_dict, 'close')
    data = stock_utils.get_intraday_cash_flow_list_all(tickers=symbolsDate_dict['symbols'], timeFrame=timeframes)
    df = stock_utils.load_intraday_cash_flow_latest_to_dataframe(data)
    
    first_date = df.index[0]
    last_date = df.index[-1]
    
    # st.write(prices_df)
    # st.stop()
    
    # filter prices_df by first_date
    prices_df = prices_df[prices_df.index >= first_date]
    prices_df = prices_df[prices_df.index <= last_date]
    
    # st.write(df)
    
    # "bp": 0.418,                 // buyup percent
    # "op": 0.500,                 // over percent
    # "p": 23268.00,               // Price
    # "pp": 0.025,                 // price percent
    # "vol": 4913057,              // Volume
    # "val": 135520044300,         // Value
    # "mc": 107588194916600.00,    // Market Capitalization
    # "nstp": 4.200,               // Performance indicator (e.g., Net Profit Margin)
    # "peravgVolume5d": 0.86,      // Average volume over 5 days
    # "rsi14": 56.04,              // RSI (14)
    # "rs3d": 57.00,               // Relative Strength Index over 3 days
    # "avgrs": 56.00,              // Average relative strength
    # "pPerMa5": 0.84,             // Price/MA(5)
    # "pPerMa20": 0.86,            // Price/MA(20)
    # "isignal": 0.000,            // Signal indicator
    # "fnetVol": -55653.00,        // Foreign net volume
    # "t": "10/05/24",             // Date
    # "seq": 1715299200            // Sequence number or timestamp
    
    prices_forward_df = prices_df.shift(-3)
    prices_change_df = (prices_forward_df - prices_df) / prices_df
    

    df['price_change'] = np.nan
    
    for i, row in df.iterrows():
        if i in prices_change_df.index:
            # st.write(df.loc[i, 'price_change'])
            # st.stop()
            if row['ticker'] not in prices_change_df.columns:
                continue
            val = prices_change_df.loc[i][row['ticker']]
            
            if isinstance(val, pd.Series):
                val = val.values[0]
            
            value = val
            # st.write(value)
            # st.stop()
            df.loc[i, 'price_change'] = value
    
    corr_df = df.copy()
    corr_df = corr_df.drop(columns=['ticker', 'seq', 'comp_name', 'ind_name', 'ind_code', 'p', 't'])
    corr = corr_df.corr()
    fig = px.imshow(corr)
    st.plotly_chart(fig, use_container_width=True)
    
    # select best 5 features
    default_dims = ['bp', 'op', 'pp', 'nstp', 'rsi14']
    selected_dims = st.multiselect('Select dimensions', df.columns, default=default_dims)
    
    fig = px.scatter_matrix(df,
        dimensions=selected_dims,
        # color by ticker
        color='ticker',
    )
    st.plotly_chart(fig, use_container_width=True)
    
    dims_df = pd.DataFrame()
    
    # stacked by ticker
    dims_dfs = {}
    
    for symbol in symbolsDate_dict['symbols']:
        symbol_df = df[df['ticker'] == symbol]
                
        for dim in symbol_df.columns:
            if dim in ['ticker', 't']:
                continue
            
            if dim not in dims_dfs:
                dims_dfs[dim] = pd.DataFrame()
                
            dims_dfs[dim][symbol] = symbol_df[dim]

    # st.write(dims_dfs)
    dims_df = pd.concat(dims_dfs, axis=1)
    

    plot_multi_line(prices_df, title='Prices', x_title='Date', y_title='Price', legend_title='Ticker')

    for dim in selected_dims:
        # st.write(dims_df[dim])
        plot_multi_scatter(dims_df[dim], title=dim, x_title='Date', y_title=dim, legend_title='Ticker')
    
