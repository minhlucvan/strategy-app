import pandas as pd
import numpy as np
import json

import streamlit as st
import vectorbt as vbt

import plotly.express as px
import plotly.graph_objs as go

from utils.trader_tcbs import TraderTCBS

def plot_assets_allocation(assets, value_col='value', symbol_col='symbol', title='Assets Allocation'):
    fig = px.pie(assets, values=value_col, names=symbol_col, title=title)
    st.plotly_chart(fig)
    
def plot_assets_return(assets):
    fig = px.bar(assets, x='symbol', y='return', title='Assets Return')
    st.plotly_chart(fig)

def run(symbol_benchmark, symbolsDate_dict):
    trader = TraderTCBS()
        
    # st.write(trader.agent.get_config('fullName'))
    
    assets = trader.agent.get_assets_info()
    cash = trader.agent.get_total_cash()
    st.write(f"Total cash: {cash}")
    
    stocks_df = trader.agent.get_total_stocks()
    
    assets_df = trader.agent.get_assets_allocation()
    plot_assets_allocation(assets_df, title='Assets Allocation')
    
    plot_assets_allocation(stocks_df, title='Stocks Allocation')
    plot_assets_return(stocks_df)