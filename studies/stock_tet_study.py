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
from utils.vbt import plot_pf
from vbt_strategy.MOM_D import get_MomDInd
import lunardate as ld

from studies.stock_custom_event_study import run as run_custom_event_study

def calculate_lunar_tet(stocks_df):
    tet_df = pd.DataFrame(index=stocks_df.index, columns=stocks_df.columns)
    
    for index, row in stocks_df.iterrows():
        lunna_date = ld.LunarDate.fromSolarDate(index.year, index.month, index.day)
        
        is_tet = lunna_date.day == 27 and lunna_date.month == 12
        tet_df.loc[index] = is_tet

    return tet_df

def run(symbol_benchmark, symbolsDate_dict):
    
    if len(symbolsDate_dict['symbols']) < 1:
        st.info("Please select symbols.")
        st.stop()
    
    # copy the symbolsDate_dict
    benchmark_dict = symbolsDate_dict.copy()
    benchmark_dict['symbols'] = [symbol_benchmark]
    
    benchmark_df = get_stocks(benchmark_dict,'close')
    stocks_df = get_stocks(symbolsDate_dict,'close')
    
    events_df = calculate_lunar_tet(stocks_df)
    
    events_df = events_df[events_df == True]
    
    st.write(events_df)
    
    # plot_events(benchmark_df[symbol_benchmark], events_df[events_df.columns[0]], 'Tet')
   
    run_custom_event_study(symbol_benchmark,
        symbolsDate_dict,
        benchmark_df=benchmark_df,
        stocks_df=stocks_df,
        events_df=events_df,
        def_days_before=6,
        def_days_after=12
    )
    