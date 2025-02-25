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
    
    with st.expander("Market Pricing Study"):
        st.markdown("""Market Pricing Study is a study that compares the market price of a bunch of stocks with their valuation. The valuation is calculated by the average of the PE and PB of the stocks. The market price is the close price of the stocks. The study is useful to identify the overvalued and undervalued stocks in the market
                    """)
        
    symbolsDate_dict['symbols'] =  symbolsDate_dict['symbols']
    # copy the symbolsDate_dict
    benchmark_dict = symbolsDate_dict.copy()
    benchmark_dict['symbols'] = [symbol_benchmark]
    
    benchmark_df = get_stocks(benchmark_dict,'close')
    stocks_df = get_stocks(symbolsDate_dict,'close')
   
    pe_df = get_stocks_valuation(symbolsDate_dict, indicator='pe')
    pb_df = get_stocks_valuation(symbolsDate_dict, indicator='pb')
    
    
    market_pe_df = pe_df.mean(axis=1)
    market_pb_df = pb_df.mean(axis=1)
    
    market_valuation_df = pd.DataFrame(index=stocks_df.index)
    market_valuation_df['pe'] = market_pe_df
    market_valuation_df['pb'] = market_pb_df
    
    plot_evaluation_pe(benchmark_df, market_valuation_df)

    plot_evaluation_pb(benchmark_df, market_valuation_df)
    