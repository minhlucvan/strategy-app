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

from utils.plot_utils import plot_events
from utils.processing import get_stocks, get_stocks_events, get_stocks_valuation
from studies.rrg import plot_RRG, rs_ratio, RRG_Strategy
from utils.vbt import plot_pf
from vbt_strategy.MOM_D import get_MomDInd

from studies.market_wide import MarketWide_Strategy
    
def run(symbol_benchmark, symbolsDate_dict):
    
    if len(symbolsDate_dict['symbols']) < 1:
        st.info("Please select symbols.")
        st.stop()
    
    # copy the symbolsDate_dict
    benchmark_dict = symbolsDate_dict.copy()
    benchmark_dict['symbols'] = [symbol_benchmark]
    
    benchmark_df = get_stocks(benchmark_dict,'close')
    stocks_df = get_stocks(symbolsDate_dict,'close')
   
    events_df = get_stocks_events(symbolsDate_dict, 'label')
    
    symbols = stocks_df.columns.values.tolist()
    event_types = events_df.stack().unique()
    
    selected_events = st.multiselect('Select events', event_types, event_types)
    selected_symbols = st.selectbox('Select symbols', symbols)
    
    stock_df = stocks_df[selected_symbols]
    event_df = events_df[selected_symbols].apply(lambda x: x if x in selected_events else np.nan)
    
    for stock in stocks_df.columns:
        try:
            stock_df = stocks_df[stock]
            event_df = events_df[stock].apply(lambda x: x if x in selected_events else np.nan)
            
            # check if there is any event
            if event_df.isna().all():
                continue
            st.write(f"Plotting {stock}")
            plot_events(stock_df, event_df)
        except:
            st.warning(f"Error plotting {stock}")