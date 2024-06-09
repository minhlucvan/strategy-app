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
     
    calender = trader.agent.get_market_calendar()
     
    event_types = calender['defType'].unique()
    
    selected_event_type = st.selectbox('Select event type', event_types)
    
    selected_events = calender[calender['defType'] == selected_event_type]
    
    st.write(selected_events)