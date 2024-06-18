import pandas as pd
import numpy as np

import streamlit as st

import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots # creating subplots

from utils.processing import get_stocks, get_stocks_events
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
   
    events_df = get_stocks_events(symbolsDate_dict, 'label')

    run_custom_event_study(symbol_benchmark, symbolsDate_dict, benchmark_df, stocks_df, events_df, 10, 90)
    
    