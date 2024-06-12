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
from studies.rrg import plot_RRG, rs_ratio, RRG_Strategy
from utils.vbt import plot_pf
from vbt_strategy.MOM_D import get_MomDInd

from studies.stock_custom_event_study import run as run_custom_event_study

import utils.vnstock as vnstock
import utils.google as google

def run(symbol_benchmark, symbolsDate_dict):
    
    if len(symbolsDate_dict['symbols']) < 1:
        st.info("Please select symbols.")
        st.stop()
    
    # copy the symbolsDate_dict
    benchmark_dict = symbolsDate_dict.copy()
    benchmark_dict['symbols'] = [symbol_benchmark]
    
    benchmark_df = get_stocks(benchmark_dict,'close')
    stocks_df = get_stocks(symbolsDate_dict,'close')
    value_change_weighted_df = get_stocks(symbolsDate_dict, 'price_change')
    
    symbol = symbolsDate_dict['symbols'][0]
    
    d = vnstock.get_all_stocks()
    
    st.write(d)
    
    x = google.lookup_table()
    st.write(x)
    
    df = vnstock.get_income_statement()
    
    
    df['last_word'] = df['Đơn vị'].str.split().str[-1]
    
    # join the data where organ_name = Đơn vị
    df = pd.merge(df, d, how='left', left_on='Đơn vị', right_on='organ_name')
    
    # join the data where last_word = 'ticker'
    df = pd.merge(df, d, how='left', left_on='last_word', right_on='ticker')
    
    st.write(df)
   
    # events_df = get_stocks_events(symbolsDate_dict, 'label')
    
    # # set nan to none
    # # events_df = events_df.fillna('')
    
    # # set type to str
    # # events_df = events_df.astype(str)
    
    # # set value = man when the input is F
    # for col in events_df.columns:
    #     for index in events_df.index:
    #         value_change_weighted = value_change_weighted_df.loc[index, col]
    #         events_df.loc[index, col] = value_change_weighted if value_change_weighted > 0 else np.nan
            
    # # st.write(events_df)

    # # events_df = events_df[events_df > dividend_threshold]
    
    # run_custom_event_study(symbol_benchmark,
    #     symbolsDate_dict,
    #     benchmark_df=benchmark_df,
    #     stocks_df=stocks_df,
    #     events_df=events_df,
    #     def_days_before=0,
    #     def_days_after=6)
   