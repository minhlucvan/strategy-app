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

from utils.processing import get_stocks, get_stocks_valuation
from studies.rrg import plot_RRG, rs_ratio, RRG_Strategy
from utils.vbt import plot_pf
from vbt_strategy.MOM_D import get_MomDInd

from studies.market_wide import MarketWide_Strategy

def plot_evaluation_pe(price_df, evaluation_df):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02)
    fig.add_trace(go.Scatter(x=price_df.index, y=price_df.iloc[:,0], mode='lines', name='Price'), row=1, col=1)
    fig.add_trace(go.Scatter(x=evaluation_df.index, y=evaluation_df.iloc[:,0], mode='lines', name='PE'), row=2, col=1)
    st.plotly_chart(fig)
    
def plot_evaluation_pb(price_df, evaluation_df):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02)
    fig.add_trace(go.Scatter(x=price_df.index, y=price_df.iloc[:,0], mode='lines', name='Price'), row=1, col=1)
    fig.add_trace(go.Scatter(x=evaluation_df.index, y=evaluation_df.iloc[:,1], mode='lines', name='PB'), row=2, col=1)
    st.plotly_chart(fig)

def run(symbol_benchmark, symbolsDate_dict):
    
    with st.expander("Multi Factor Model"):
        st.markdown("""Multi Factor Model is a model that combines multiple factors to predict stock returns.
                    
Value: Metrics like P/E, P/B, Dividend Yield.
Size: Market capitalization.
Momentum: Past returns over various periods.
Quality: Metrics like ROE, Debt-to-Equity ratio, Earnings Stability.
Volatility: Historical price volatility or Beta.""")
    