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

from utils.plot_utils import plot_multi_line
from utils.processing import get_stocks, get_stocks_events, get_stocks_financial, get_stocks_valuation
from studies.rrg import plot_RRG, rs_ratio, RRG_Strategy
from utils.vbt import plot_pf
from vbt_strategy.MOM_D import get_MomDInd

from studies.market_wide import MarketWide_Strategy

def plot_events(price_series, events_series):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=price_series.index, y=price_series, mode='lines', name='Price'))
    max_price = price_series.max()
    min_price = price_series.min()
    # add horizontal line for events, annotation_text = event
    for index in events_series.index:
        event = events_series[index]
        if not pd.isna(event):
            # add horizontal line, x = index, y = max_price
            fig.add_shape(type="line",
                x0=index, y0=max_price, x1=index, y1=min_price,
                line=dict(color="RoyalBlue",width=1))
            # add annotation
            fig.add_annotation(x=index, y=max_price, text=event, showarrow=False, yshift=10)                 
            
    st.plotly_chart(fig)
    
def run(symbol_benchmark, symbolsDate_dict):
    
    if len(symbolsDate_dict['symbols']) < 1:
        st.info("Please select symbols.")
        st.stop()
    
    # copy the symbolsDate_dict
    benchmark_dict = symbolsDate_dict.copy()
    benchmark_dict['symbols'] = [symbol_benchmark]
    
    stocks_df = get_stocks(symbolsDate_dict,'close')
    stocks_df.index = pd.to_datetime(stocks_df.index).tz_localize(None)
    
   
    financials_dfs = get_stocks_financial(symbolsDate_dict, raw=True)
    
    eps_df = pd.DataFrame()
    pe_df = pd.DataFrame()
    pb_df = pd.DataFrame()
    real_pe_df = pd.DataFrame()
    real_pb_df = pd.DataFrame()
    
    # Ensure both indices are timezone-naive
    for symbol in financials_dfs:
        financials_df = financials_dfs[symbol].copy()
        financials_df.index = pd.to_datetime(financials_df.index).tz_localize(None)
        
        # filter financials_df > start_date
        financials_df = financials_df[financials_df.index >= stocks_df.index[0]]
        
        stock_df = stocks_df[symbol]
        
        # reindex stock_df to financials_df, fill nearest value
        union_df = financials_df.reindex(stock_df.index, method='ffill')
    
        union_df['close'] = stock_df
        union_df['realPriceToEarning'] = union_df['close'] / union_df['earningPerShare']
        union_df['realPriceToBook'] = union_df['close'] / union_df['bookValuePerShare']
                
        eps_df = pd.concat([eps_df, union_df['earningPerShare']], axis=1)
        pe_df = pd.concat([pe_df, union_df['priceToEarning']], axis=1)
        pb_df = pd.concat([pb_df, union_df['priceToBook']], axis=1)
        real_pe_df = pd.concat([real_pe_df, union_df['realPriceToEarning']], axis=1)
        real_pb_df = pd.concat([real_pb_df, union_df['realPriceToBook']], axis=1)
        
    
    plot_multi_line(eps_df, title='Earning Per Share', x_title='Date', y_title='Earning Per Share', legend_title='Stocks')
    
    plot_multi_line(pe_df, title='Price to Earning', x_title='Date', y_title='Price to Earning', legend_title='Stocks')
    plot_multi_line(real_pe_df, title='Real Price to Earning', x_title='Date', y_title='Real Price to Earning', legend_title='Stocks')
    
    plot_multi_line(pb_df, title='Price to Book', x_title='Date', y_title='Price to Book', legend_title='Stocks')
    plot_multi_line(real_pb_df, title='Real Price to Book', x_title='Date', y_title='Real Price to Book', legend_title='Stocks')