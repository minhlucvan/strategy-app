
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from studies.stock_news_momentum_study import calculate_price_changes
from studies.stock_overnight_study import plot_scatter_2_sources
from utils.plot_utils import plot_multi_bar, plot_multi_line, plot_single_bar, plot_single_line, plot_multi_scatter, plot_events
from utils.processing import get_stocks, get_stocks_foregin_flow
import vectorbt as vbt

from studies.stock_custom_event_study import calculate_event_affection, get_event_affection, run as run_custom_event_study

    
def run(symbol_benchmark, symbolsDate_dict):
    
    # copy the symbolsDate_dict
    # benchmark_dict = symbolsDate_dict.copy()
    # symbolsDate_dict['symbols'] = ['VN30F1M']
    
    close_df = get_stocks(symbolsDate_dict, 'close')
    open_df = get_stocks(symbolsDate_dict, 'open')
    
    pricing_method = st.selectbox('Pricing Method', ['close', 'open', 'hl2', 'hlc3', 'ohlc4'], index=4)
    
    price_df = get_stocks(symbolsDate_dict, pricing_method)
    
    benchmark_df = get_stocks(symbolsDate_dict, 'close', benchmark=True)
    
    ma_period = st.slider('MA Period', min_value=5, max_value=200, value=5, step=1)
    
    ma_ind = vbt.MA.run(close_df, ma_period)
    
    ma = ma_ind.ma[ma_period]
        
    gap_df = open_df - close_df.shift(1)
    gap_pct_df = gap_df / close_df.shift(1)
    
    gap_threshold = st.slider('Gap Percentage', min_value=0.0, max_value=0.15, value=0.06, step=0.01, format="%.3f")
    
    # keep gap pct > 0.02 or < -0.02
    gap_pct_df = gap_pct_df[(gap_pct_df < -gap_threshold)]
    
    # keep price < ma
    gap_pct_df = gap_pct_df[close_df < ma]
    
    # event_affection_df = calculate_event_affection(close_df, gap_pct_df, 0, 3)
    
    # event_affection_positive_df = event_affection_df[event_affection_df > 0]
    
    # for col in event_affection_df.columns:
    #     plot_single_bar(event_affection_df[col], title=f'Event Affection {col}', x_title='Date', y_title='Affection', legend_title='Stocks', price_df=close_df[col])
    
    run_custom_event_study(
        symbol_benchmark,
        symbolsDate_dict,
        benchmark_df=benchmark_df,
        stocks_df=price_df,
        events_df=gap_pct_df,
        def_days_before=0,
        def_days_after=3)
    