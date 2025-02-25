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

from utils.processing import get_stocks, get_stocks_events, get_stocks_valuation
from utils.vbt import plot_pf
from vbt_strategy.MOM_D import get_MomDInd

from studies.stock_custom_event_study import run as run_custom_event_study

def run(symbol_benchmark, symbolsDate_dict):
    
    if len(symbolsDate_dict['symbols']) < 1:
        st.info("Please select symbols.")
        st.stop()
    
    # copy the symbolsDate_dict
    benchmark_dict = symbolsDate_dict.copy()
    benchmark_dict['symbols'] = [symbol_benchmark]
    
    benchmark_df = get_stocks(benchmark_dict,'close')
    stocks_df = get_stocks(symbolsDate_dict,'close')
   
    events_df = get_stocks_events(symbolsDate_dict, 'cashDividend')
    st.write(events_df)

    dividend_threshold = st.number_input('Dividend Threshold', min_value=0, max_value=6000, value=1000)
    
    events_df = events_df[events_df > dividend_threshold]
    
    run_custom_event_study(symbol_benchmark,
        symbolsDate_dict,
        benchmark_df=benchmark_df,
        stocks_df=stocks_df,
        events_df=events_df,
        def_days_before=6,
        def_days_after=0
    )
    