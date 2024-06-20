import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objs as go

from indicators.AnySign import get_AnySignInd
from utils.component import input_SymbolsDate
from utils.plot_utils import plot_multi_bar, plot_multi_line, plot_single_bar, plot_single_line
from utils.processing import get_stocks
import vectorbt as vbt

from utils.vbt import plot_pf

# Constants
WINDOW_SIZE = 10
CLIP_VALUE = 0.02
HEATMAP_COLORSCALE = 'RdYlGn'
HEATMAP_WIDTH = 1000
HEATMAP_HEIGHT = 400


def process_stock_changes(stocks_df):
    stocks_change_df = stocks_df.pct_change().fillna(0)
    # fill > 0 values with 1, < 0 values with -1
    stocks_change_df = stocks_change_df.applymap(lambda x: 1 if x > 0 else -1)
    # stocks_change_df = stocks_change_df.clip(-CLIP_VALUE, CLIP_VALUE)
    return stocks_change_df

def create_heatmap_df(stocks_change_df):
    stocks_change_unamed_df = pd.DataFrame(index=stocks_change_df.index)
    for i in stocks_change_df.index:
        sorted_values = sorted(stocks_change_df.loc[i].values)
        for j, value in enumerate(sorted_values):
            stocks_change_unamed_df.loc[i, f'{j}'] = value
    return stocks_change_unamed_df

def plot_heatmap(stocks_change_unamed_df):
    fig = go.Figure(data=go.Heatmap(
        z=stocks_change_unamed_df.T,
        x=stocks_change_unamed_df.index,
        y=stocks_change_unamed_df.columns,
        colorscale=HEATMAP_COLORSCALE,
        zmid=0
    ))
    fig.update_layout(
        title='Stocks Change',
        xaxis_title='Date',
        yaxis_title='Stocks',
        width=HEATMAP_WIDTH,
        height=HEATMAP_HEIGHT
    )
    st.plotly_chart(fig, use_container_width=True)

def calculate_mean_change(stocks_change_df):
    mean_change = stocks_change_df.sum(axis=1)
    return mean_change

def run(symbol_benchmark, symbolsDate_dict):
    stocks_df = get_stocks(symbolsDate_dict, 'close')
    value_df = get_stocks(symbolsDate_dict, 'value')
    symbolsDate_dict['symbols'] = [symbol_benchmark]
    benchmark_df = get_stocks(symbolsDate_dict, 'close')
    
    st.write(f"Total stocks: {len(value_df.columns)}")
    
    plot_multi_line(stocks_df, title='Stocks Prices', x_title='Date', y_title='Price', legend_title='Stocks')
    
    value_df = value_df.rolling(window=WINDOW_SIZE).mean()
    
    # plot stacked area chart for stocks
    fig = go.Figure()
    for val in value_df.columns:
        fig.add_trace(go.Scatter(
            x=value_df.index,
            y=value_df[val],
            mode='lines',
            name=val,
            stackgroup='one'  # setting stackgroup for stacked area chart
        ))

    st.plotly_chart(fig)
    
    value_total_df = value_df.sum(axis=1)
    value_contrib_df = value_df.div(value_total_df, axis=0)
    
    # plot stacked area chart for stocks
    fig = go.Figure()
    for val in value_contrib_df.columns:
        fig.add_trace(go.Scatter(
            x=value_contrib_df.index,
            y=value_contrib_df[val],
            mode='lines',
            name=val,
            stackgroup='one'  # setting stackgroup for stacked area chart
        ))
    
    st.plotly_chart(fig)
    
    value_contrib_change_df = value_contrib_df.pct_change().fillna(0).rolling(window=100).sum()
    
    plot_multi_bar(value_contrib_change_df, title='Stocks Contribution Change', x_title='Date', y_title='Price Change', legend_title='Stocks')