import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objs as go

from indicators.AnySign import get_AnySignInd
from utils.component import input_SymbolsDate
from utils.plot_utils import plot_multi_line, plot_single_bar
from utils.processing import get_stocks
import vectorbt as vbt

from utils.vbt import plot_pf

# Constants
WINDOW_SIZE = 50
CLIP_VALUE = 0.07
HEATMAP_COLORSCALE = 'RdYlGn'
HEATMAP_WIDTH = 1000
HEATMAP_HEIGHT = 400

def load_data(symbol_benchmark, symbolsDate_dict):
    stocks_df = get_stocks(symbolsDate_dict, 'close')
    symbolsDate_dict['symbols'] = [symbol_benchmark]
    benchmark_df = get_stocks(symbolsDate_dict, 'close')
    return stocks_df, benchmark_df

def process_stock_changes(stocks_df):
    stocks_change_df = stocks_df.pct_change().fillna(0).clip(-CLIP_VALUE, CLIP_VALUE)
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
    mean_change = stocks_change_df.sum(axis=1).rolling(window=WINDOW_SIZE).mean()
    return mean_change

def run(symbol_benchmark, symbolsDate_dict):
    stocks_df, benchmark_df = load_data(symbol_benchmark, symbolsDate_dict)
    stocks_change_df = process_stock_changes(stocks_df)
    stocks_change_unamed_df = create_heatmap_df(stocks_change_df)
    
    plot_multi_line(benchmark_df, title='Stock Prices', x_title='Date', y_title='Price', legend_title='Stocks')
    plot_heatmap(stocks_change_unamed_df)
    
    mean_change = calculate_mean_change(stocks_change_df)
    plot_single_bar(mean_change, title='Mean Change', x_title='Date', y_title='Price Change', legend_title='Mean', price_df=benchmark_df[symbol_benchmark])

    price = benchmark_df[symbol_benchmark]
    
    # reindex price to mean_change
    price = price.reindex(mean_change.index)
    
    entry_threshold = [0.0, 0.01]
    exit_threshold = [0.0, -0.01]

    mkw_indicator = get_AnySignInd().run(
        price,
        mean_change,
        entry_threshold=entry_threshold,
        exit_threshold=exit_threshold,
        param_product=True
    )

    # entries = True if mean_change > 0 else False
    entries = mkw_indicator.entry_signal.vbt.signals.fshift()
    exits = mkw_indicator.exit_signal.vbt.signals.fshift()
    
    shorts_entries = exits.copy()
    shorts_exits = entries.copy()
        
    pf = vbt.Portfolio.from_signals(
        close=price,
        entries=entries,
        exits=exits,
        # short_entries=shorts_entries,
        # short_exits=shorts_exits,
        # direction='shortonly',
        sl_stop=0.1,
        fees=0.001,
        slippage=0.001,
        freq='1D'
    )

    max_sharpe_id = pf.sharpe_ratio().idxmax()
        
    st.write(f'Max Sharpe Ratio at {max_sharpe_id}')
    st.write(f'Sharpe Ratio: {pf.sharpe_ratio().max()}')
    
    pf = pf[max_sharpe_id]
    
    plot_pf(pf)
    