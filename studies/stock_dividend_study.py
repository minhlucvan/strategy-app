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
   
    events_df = get_stocks_events(symbolsDate_dict, 'cashDividend')
    
    
    days_before = st.number_input('Days before event', min_value=1, max_value=10, value=6)
    days_after = st.number_input('Days after event', min_value=1, max_value=10, value=0)
    
    event_affection = {}
    
    for stock in stocks_df.columns:
        stock_df = stocks_df[stock]
        event_df = events_df[stock]
        event_df = event_df.dropna()
        
        event_affection_df = pd.DataFrame(index=stock_df.index, columns=['event_price_change'])
        
        # check if there is any event
        if event_df.empty:
            continue
        
        # remove event < 1000
        event_df = event_df[event_df > 1000]
        
        event_df = pd.DataFrame(event_df)
        
        
        # find 6 days before the event
        event_df['event_date'] = event_df.index
        
        event_df['event_date'] = pd.to_datetime(event_df['event_date'])
        
        event_df['event_before_date'] = event_df['event_date'].apply(lambda x: x - pd.DateOffset(days=days_before))
        event_df['event_after_date'] = event_df['event_date'].apply(lambda x: x + pd.DateOffset(days=days_after))
        
        # convert to pd.datetime
        event_df['event_before_date'] = pd.to_datetime(event_df['event_before_date'])
        event_df['event_date'] = pd.to_datetime(event_df['event_date'])
        event_df['event_after_date'] = pd.to_datetime(event_df['event_after_date'])
        
        
        for index, row in event_df.iterrows():
            event_before_date = row['event_before_date']
            event_date = row['event_date']
            # find first index >= event_before_date
            event_before_price = stock_df[stock_df.index >= event_before_date].iloc[0]
            event_price = stock_df[stock_df.index >= event_date].iloc[0]
            event_after_price = stock_df[stock_df.index >= row['event_after_date']].iloc[0]
            
            event_price_change = (event_after_price - event_before_price) / event_before_price
            
            event_affection_df.loc[event_date, 'event_price_change'] = event_price_change
            
        event_affection[stock] = event_affection_df['event_price_change']

    event_unstack_df = events_df.unstack().reset_index()
    event_unstack_df.columns = ['Stock', 'Date', 'Cash Dividend']
    
    # drop na
    event_unstack_df = event_unstack_df.dropna()
    
    # set index to Date
    event_unstack_df.index = event_unstack_df['Date']
    
    # scatter plot the cash dividend
    st.write("Cash Dividend Scatter")
    fig = px.scatter(event_unstack_df, x="Date", y="Cash Dividend", color="Stock")
    st.plotly_chart(fig)
    
    # plot the cash dividend distribution
    st.write("Cash Dividend Distribution")
    fig = px.histogram(event_unstack_df, x="Cash Dividend", color="Stock", marginal="box", nbins=50)
    st.plotly_chart(fig)    
    
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
        
    # plot the price change mean by date bars
    st.write("Price Change Mean by Date")
    fig = px.bar(events_affection_unstack_daily_df, x=events_affection_unstack_daily_df.index, y="Price Change")
    # color green if positive, red if negative
    fig.update_traces(marker_color=['green' if x >= 0 else 'red' for x in events_affection_unstack_daily_df['Price Change']])
    # hide legend
    fig.update_layout(showlegend=False)
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
    