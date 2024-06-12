import pandas as pd
import numpy as np

import streamlit as st

import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots # creating subplots

from utils.processing import get_stocks, get_stocks_events
    
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
    
    days_before = st.number_input('Days before event', min_value=1, max_value=10, value=6)
    days_after = st.number_input('Days after event', min_value=0, max_value=10, value=0)
    
    event_affection = {}
    
    for stock in stocks_df.columns:
        
        if stock not in events_df.columns:
            continue
        
        stock_df = stocks_df[stock]
        event_df = events_df[stock]
                
        # filter event with label I, A
        event_df = event_df[event_df.isin(['I', 'A'])]
        
        event_df = event_df.dropna()
        
        event_affection_df = pd.DataFrame(index=stock_df.index, columns=['event_price_change'])
        
        # check if there is any event
        if event_df.empty:
            continue
        
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
            event_after_date = row['event_after_date']
            # find first index >= event_before_date
            event_before_price = stock_df[stock_df.index >= event_before_date].iloc[0]
            event_price = stock_df[stock_df.index >= event_date].iloc[0]
            event_after_price = stock_df[stock_df.index >= event_after_date].iloc[0]
            
            if pd.isna(event_before_price) or pd.isna(event_price) or pd.isna(event_after_price):
                continue
            
            event_price_change = (event_after_price - event_before_price) / event_before_price
            
            event_affection_df.loc[event_date, 'event_price_change'] = event_price_change
            
        event_affection[stock] = event_affection_df['event_price_change']

    event_unstack_df = events_df.unstack().reset_index()
    event_unstack_df.columns = ['Stock', 'Date', 'Label']
    
    # drop na
    event_unstack_df = event_unstack_df.dropna()
    
    # set index to Date
    event_unstack_df.index = event_unstack_df['Date'] 
    
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
    events_affection_unstack_daily_df = events_affection_unstack_df.groupby(events_affection_unstack_df.index).agg({'Price Change': 'mean', 'Date': 'count'})

    events_affection_unstack_daily_df.columns = ['Price Change', 'total']
        
    # plot total trade by date
    st.write("Total Trade by Date")
    fig = px.bar(events_affection_unstack_daily_df, x=events_affection_unstack_daily_df.index, y="total")
    st.plotly_chart(fig)

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
    
    # calculate the max drawdown
    max_drawdown = events_affection_unstack_daily_cumsum_df['Price Change'].min()
    st.write(f"Max Drawdown: {max_drawdown}")
    
    anualized_return = events_affection_unstack_daily_cumsum_df['Price Change'].iloc[-1] / len(events_affection_unstack_daily_cumsum_df) * 252
    st.write(f"Anualized Return: {anualized_return}")
    
    # calculate the benchmark anualized return
    benchmark_anualized_return = benchmark_return_cumsum[symbol_benchmark].iloc[-1] / len(benchmark_return_cumsum) * 252
    
    st.write(f"Benchmark Anualized Return: {benchmark_anualized_return}")
    
    # calculate the sharpe ratio
    daily_return = events_affection_unstack_daily_df['Price Change']
    daily_return = daily_return.dropna()
    daily_return = daily_return[daily_return != 0]
    daily_return = daily_return.dropna()
    
    sharpe_ratio = daily_return.mean() / daily_return.std() * np.sqrt(252)
    st.write(f"Sharpe Ratio: {sharpe_ratio}")
    