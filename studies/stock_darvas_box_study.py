import streamlit as st
import pandas as pd
import numpy as np
import json

from utils.plot_utils import plot_events
from utils.processing import get_stocks
import numpy as np
import vectorbt as vbt
import talib as ta

from utils.vbt import plot_pf
from .stock_custom_event_study import run as run_custom_event_study

def run(symbol_benchmark, symbolsDate_dict):
    
    
    if len(symbolsDate_dict['symbols']) < 1:
        st.info("Please select symbols.")
        st.stop()
            
    prices_df = get_stocks(symbolsDate_dict, stack=True, stack_level='factor')
    benchmark_df = get_stocks(symbolsDate_dict, 'close', benchmark=True)
    