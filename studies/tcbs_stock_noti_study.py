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
    symbol = symbolsDate_dict['symbols'][0]
    trader = TraderTCBS()
     
    notifications = trader.agent.get_stock_noti_all(ticker=symbol)
     
    st.write(notifications)