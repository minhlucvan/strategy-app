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
from utils.vbt import plot_pf

def run(symbol_benchmark, symbolsDate_dict):
    if len(symbolsDate_dict['symbols']) < 1:
        st.info("Please select symbols.")
        st.stop()
        
    symbolsDate_dict['symbols'] =  [symbol_benchmark] + symbolsDate_dict['symbols']

    
    stocks_df = get_stocks(symbolsDate_dict,'close')
    
    # drop na
    stocks_df = stocks_df.dropna(axis=1, how='any')
    
    
    st.write(stocks_df)

    pf = RRG_Strategy(symbol_benchmark, stocks_df, output_bool=True)
    plot_pf(pf, name= 'RRG Strategy', bm_symbol=symbol_benchmark, bm_price=stocks_df[symbol_benchmark], select=True)
