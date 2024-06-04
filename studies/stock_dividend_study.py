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
    
    benchmark_df = get_stocks(benchmark_dict,'close')
    stocks_df = get_stocks(symbolsDate_dict,'close')
   
    events_df = get_stocks_events(symbolsDate_dict, 'label')
    
    symbols = stocks_df.columns.values.tolist()
    event_types = events_df.stack().unique()      
    
    selected_events = ['D']
    
    event_affection = {}
    
    for stock in stocks_df.columns:
        stock_df = stocks_df[stock]
        event_df = events_df[stock].apply(lambda x: x if x in selected_events else np.nan)
        event_df = event_df.dropna()
        
        event_affection_df = pd.DataFrame(index=stock_df.index, columns=['event_price_change'])
        
        # check if there is any event
        if event_df.empty:
            continue
        
        event_df = pd.DataFrame(event_df)
        
        # find 6 days before the event
        event_df['event_date'] = event_df.index
        
        event_df['event_date'] = pd.to_datetime(event_df['event_date'])
        
        event_df['event_before_date'] = event_df['event_date'].apply(lambda x: x - pd.DateOffset(days=6))
        
        # convert to pd.datetime
        event_df['event_before_date'] = pd.to_datetime(event_df['event_before_date'])
        event_df['event_date'] = pd.to_datetime(event_df['event_date'])
        
        for index, row in event_df.iterrows():
            event_before_date = row['event_before_date']
            event_date = row['event_date']
            # find first index >= event_before_date
            event_before_price = stock_df[stock_df.index >= event_before_date].iloc[0]
            event_price = stock_df[stock_df.index >= event_date].iloc[0]
            
            event_price_change = (event_price - event_before_price) / event_before_price
            
            event_affection_df.loc[event_date, 'event_price_change'] = event_price_change
            
        event_affection[stock] = event_affection_df['event_price_change']

    
        
    events_affection_df = pd.DataFrame(event_affection)
    
    events_affection_unstack_df = events_affection_df.unstack().reset_index()
    events_affection_unstack_df.columns = ['Stock', 'Date', 'Price Change']
    
    # drop na
    events_affection_unstack_df = events_affection_unstack_df.dropna()
        
    # set index to Date
    events_affection_unstack_df.index = events_affection_unstack_df['Date']
    
    # plot the price change distribution
    st.write("Price Change Distribution")
    fig = px.histogram(events_affection_unstack_df, x="Price Change", color="Stock", marginal="box", nbins=50)
    st.plotly_chart(fig)
    
    # plot the price change scatter by date color by stock
    st.write("Price Change Scatter")
    fig = px.scatter(events_affection_unstack_df, x="Date", y="Price Change", color="Stock")
    st.plotly_chart(fig)
    
    # group by index
    events_affection_unstack_daily_df = events_affection_unstack_df.groupby(events_affection_unstack_df.index).agg({'Price Change': 'mean'})
        
    # plot the price change mean by date
    st.write("Price Change Mean by Date")
    fig = px.line(events_affection_unstack_daily_df, x=events_affection_unstack_daily_df.index, y="Price Change")
    st.plotly_chart(fig)
    
    events_affection_unstack_daily_cumsum_df = events_affection_unstack_daily_df.cumsum()
    
    # benchmark return
    benchmark_return = benchmark_df.pct_change()
    benchmark_return_cumsum = benchmark_return.cumsum()
    
    # align benchmark_return_cumsun to the same date range
    benchmark_return_cumsum = benchmark_return_cumsum[benchmark_return_cumsum.index >= events_affection_unstack_daily_cumsum_df.index[0]]
    
    # plot the price change cumsum by date
    st.write("Price Change Cumsum by Date")
    fig = px.line(events_affection_unstack_daily_cumsum_df, x=events_affection_unstack_daily_cumsum_df.index, y="Price Change")
    fig.add_trace(go.Scatter(x=benchmark_return_cumsum.index, y=benchmark_return_cumsum[symbol_benchmark], mode='lines', name='Benchmark'))
    st.plotly_chart(fig)
    