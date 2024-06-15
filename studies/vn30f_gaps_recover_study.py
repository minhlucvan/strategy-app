
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from studies.stock_news_momentum_study import calculate_price_changes
from studies.stock_overnight_study import plot_scatter_2_sources
from utils.plot_utils import plot_multi_bar, plot_multi_line, plot_single_bar, plot_single_line, plot_multi_scatter, plot_events
from utils.processing import get_stocks, get_stocks_foregin_flow


    
def run(symbol_benchmark, symbolsDate_dict):
    
    # copy the symbolsDate_dict
    # benchmark_dict = symbolsDate_dict.copy()
    symbolsDate_dict['symbols'] = ['VN30F1M']
    
    stock_df = get_stocks(symbolsDate_dict, stack=False, timeframe='D')
    st.write(stock_df)
    
    stock_df['gap'] = stock_df['open'] - stock_df['close'].shift(1)
    stock_df['gap_pct'] = stock_df['gap'] / stock_df['close'].shift(1)
    
    gap_pct = st.slider('Gap Percentage', min_value=0.0, max_value=0.04, value=0.01, step=0.001, format="%.3f")
    
    gap_down_df = stock_df[stock_df['gap_pct'] < -gap_pct]
    gap_up_df = stock_df[stock_df['gap_pct'] > gap_pct]
    
    
    # plot candlestick
    fig = go.Figure(data=[go.Candlestick(x=stock_df.index,
                open=stock_df['open'],
                high=stock_df['high'],
                low=stock_df['low'],
                close=stock_df['close'])])
    # plot gaps down as vertical lines
    fig.add_trace(go.Scatter(x=gap_down_df.index, y=gap_down_df['high'], mode='markers', marker=dict(color='red'), name='Gap Down'))
    # plot gaps up as vertical lines
    fig.add_trace(go.Scatter(x=gap_up_df.index, y=gap_up_df['low'], mode='markers', marker=dict(color='green'), name='Gap Up'))
    st.plotly_chart(fig, use_container_width=True)
    
    # number of gaps
    st.write(f"Number of gaps down: {gap_down_df.shape[0]}")
    st.write(f"Number of gaps up: {gap_up_df.shape[0]}")
     
    stock_df['up'] = stock_df['high'] - stock_df['open']
    stock_df['down'] = stock_df['open'] - stock_df['low']
    
    # calculate the gap fill
    gap_down_df['fill_up'] = (stock_df['high'] - stock_df['open'])
    gap_up_df['fill_down'] = (stock_df['low'] - stock_df['open'])
    
    # fill up - gap down
    gap_down_df['cover_up'] = gap_down_df['fill_up'] + gap_down_df['gap']
    
 
    # cover up > 0
    gap_down_filled_df = gap_down_df[gap_down_df['cover_up'] > 0]
    
    # plot the gap fill
    # plot_single_bar(gap_up_filled_df['fill_down'], title='Gap Up Fill', x_title='Date', y_title='Price', legend_title='Stocks')
    plot_single_bar(gap_down_filled_df['gap'], title='Gap Down Fill', x_title='Date', y_title='Price', legend_title='Stocks')
    
    total_gap_down_filled = len(gap_down_filled_df)
    
    st.write(f"Number of gap down filled: {total_gap_down_filled}")
    

    accuracy = total_gap_down_filled / gap_down_df.shape[0]
    
    st.write(f"Accuracy: {accuracy}")