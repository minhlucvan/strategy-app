import pandas as pd
import numpy as np
import streamlit as st
import vectorbt as vbt
import utils.plot_utils as pu
from utils.processing import get_stocks
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from vbt_strategy.MOM_D import get_MomDInd


def run(symbol_benchmark, symbolsDate_dict):
    if not symbolsDate_dict.get('symbols'):
        st.info("Please select symbols.")
        return

    # Fetch stock & benchmark data
    stocks_df = get_stocks(symbolsDate_dict, 'close')
    if stocks_df.empty:
        st.warning("No valid stock data retrieved")
        return
        
    benchmark_df = get_stocks(symbolsDate_dict, 'close', benchmark=True)
        
    if benchmark_df.empty:
        st.warning("No valid benchmark data retrieved")
        return

    # User-defined lookback period
    lookback_period = st.slider('Select Lookback Period (days)', 5, 200, 60)
      
    stock_returns = stocks_df.pct_change(periods=lookback_period)
    

    # Symbol selection for detailed view
    selected_symbol = st.selectbox('Select symbols for detailed analysis', symbolsDate_dict['symbols'])
    if selected_symbol:
        selected_stock_df = stocks_df[selected_symbol]
        selected_stock_returns = stock_returns[selected_symbol]
        
        pu.plot_single_line(
            selected_stock_returns,
            title='Selected Symbols Returns',
            x_title='Date',
            y_title='Returns'
        )
        
        # return distribution histogram
        fig = make_subplots(rows=1, cols=1)
        fig.add_trace(go.Histogram(x=selected_stock_returns, nbinsx=50, name='Returns Distribution'))
        st.plotly_chart(fig)
        
        # stats
        st.write("### Returns Statistics")
        st.write(selected_stock_returns.describe())