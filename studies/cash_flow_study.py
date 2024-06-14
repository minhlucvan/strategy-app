import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
from utils.plot_utils import plot_multi_bar, plot_multi_line, plot_multi_scatter
from utils.processing import get_stocks
import utils.stock_utils as stock_utils


def load_data(symbolsDate_dict, timeframes):
    prices_df = get_stocks(symbolsDate_dict, 'close')
    data = stock_utils.get_intraday_cash_flow_list_all(tickers=symbolsDate_dict['symbols'], timeFrame=timeframes)
    df = stock_utils.load_intraday_cash_flow_latest_to_dataframe(data)
    return prices_df, df


def filter_prices(prices_df, df):
    first_date = df.index[0]
    last_date = df.index[-1]
    prices_df = prices_df[(prices_df.index >= first_date) & (prices_df.index <= last_date)]
    return prices_df


def calculate_price_changes(prices_df):
    price_changes = {
        'price_change_1': (prices_df.shift(-1) - prices_df) / prices_df,
        'price_change_3': (prices_df.shift(-3) - prices_df) / prices_df,
        'price_change_5': (prices_df.shift(-5) - prices_df) / prices_df,
        'price_change_10': (prices_df.shift(-10) - prices_df) / prices_df,
        'price_change_15': (prices_df.shift(-15) - prices_df) / prices_df
    }
    return price_changes


def add_price_changes_to_df(df, price_changes):
    df['fnetVolSum'] = df['fnetVol'].cumsum()
    for i, row in df.iterrows():
        if i in price_changes['price_change_1'].index:
            ticker = row['ticker']
            if ticker in price_changes['price_change_1'].columns:
                for key, change in price_changes.items():
                    df.loc[i, key] = change.loc[i, ticker]
    return df


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


def plot_correlation_matrix(df):
    corr_df = df.drop(columns=['ticker', 'seq', 'comp_name', 'ind_name', 'ind_code', 'p', 't'])
    corr = corr_df.corr()
    fig = px.imshow(corr)
    st.plotly_chart(fig, use_container_width=True)


def plot_scatter_matrix(df, selected_dims):
    fig = px.scatter_matrix(df, dimensions=selected_dims, color='ticker')
    st.plotly_chart(fig, use_container_width=True)


def prepare_dims_df(df, symbols):
    dims_dfs = {}
    for symbol in symbols:
        symbol_df = df[df['ticker'] == symbol]
        for dim in symbol_df.columns:
            if dim not in dims_dfs:
                dims_dfs[dim] = pd.DataFrame()
                
            dims_dfs[dim][symbol] = symbol_df[dim]
    dims_df = pd.concat(dims_dfs, axis=1)
    return dims_df


def run(symbol_benchmark, symbolsDate_dict):
    timeframes = st.selectbox('Select timeframes', ['1Y', '1M', '1W', '1D'])
    prices_df, df = load_data(symbolsDate_dict, timeframes)
    prices_df = filter_prices(prices_df, df)

    price_changes = calculate_price_changes(prices_df)
    df = add_price_changes_to_df(df, price_changes)
    
    display_description()
    plot_correlation_matrix(df)

    default_dims = ['bp', 'op', 'pp', 'nstp', 'fnetVol']
    selected_dims = st.multiselect('Select dimensions', df.columns, default=default_dims)
    plot_scatter_matrix(df, selected_dims)
    
    dims_df = prepare_dims_df(df, symbolsDate_dict['symbols'])
    plot_multi_line(prices_df, title='Prices', x_title='Date', y_title='Price', legend_title='Ticker')

    for dim in selected_dims:
        plot_multi_scatter(dims_df[dim], title=dim, x_title='Date', y_title=dim, legend_title='Ticker')

        dim_changes_df = dims_df[dim].pct_change()
        
        plot_multi_bar(dim_changes_df, title=f'{dim} Change', x_title='Date', y_title=f'{dim} Change', legend_title='Ticker')