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

from utils.processing import get_stocks
from studies.rrg import plot_RRG, rs_ratio, RRG_Strategy
from utils.vbt import plot_pf
from vbt_strategy.MOM_D import get_MomDInd
import utils.stock_utils as su
from studies.market_wide import MarketWide_Strategy
import utils.db as db

def run(symbol_benchmark, symbolsDate_dict):    
    tickers_df = db.load_symbols('VN')
    
    selected_exchanges = st.multiselect('Select exchanges', tickers_df['exchange'].unique())
    
    if len(selected_exchanges) > 0:
        tickers_df = tickers_df[tickers_df['exchange'].isin(selected_exchanges)]
    
    tickers = tickers_df['symbol'].tolist()
    df = su.get_intraday_snapshots_all(tickers)
    
    vol_df = df['volume']
    price_df = df['price']
    
    value_df = vol_df * price_df
    
    total_value = value_df.sum(axis=0)
    
    # set column names
    total_value.name = 'Total Value'
    total_value.index.name = 'symbol'

    
    threshold = st.slider('Threshold', 0, 50_000_000_000, 1000_000_000)
    total_value = total_value[total_value > threshold]
    
    st.dataframe(total_value, use_container_width=True)
    
    count = len(total_value)
    
    st.write(f'Total number of stocks with total value > {threshold}: {count}')

        
    total_value_df = pd.DataFrame(total_value, columns=['Total Value'])
    total_value_df.index.name = 'symbol'
    
    # st.write(total_value_df)
    
    # plot volume histogram color by symbol
    fig = px.histogram(total_value_df, x='Total Value', color=total_value_df.index)
    fig.update_layout(bargap=0.1)
    st.plotly_chart(fig, use_container_width=True)
    
    # Create the treemap
    fig = px.treemap(
        total_value_df,
        path=[total_value_df.index],
        values='Total Value',
        color=total_value_df.index
    )

    # Display the treemap in Streamlit
    st.plotly_chart(fig, use_container_width=True)
    
    file_path = st.text_input('File path', 'data/intraday_total_value.csv')
    if st.button('Save to CSV'):
        total_value.to_csv(file_path)
        st.write(f'Saved to {file_path}')