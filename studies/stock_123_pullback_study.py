import streamlit as st
import pandas as pd
import numpy as np
import json

from utils.plot_utils import plot_events
from utils.processing import get_stocks
import numpy as np
import vectorbt as vbt
import talib as ta

from utils.vbt import plot_pf
from .stock_custom_event_study import run as run_custom_event_study

# calculate consecutive lows
# consecutive_lows = today's low < n days ago low
def calculate_consecutive_lows(df, n=3):
    lows = df['low']
    return (lows.shift(1) > lows).rolling(n).sum()

def run(symbol_benchmark, symbolsDate_dict):
    
    with st.expander("1 2 3 Pullback Strategy"):
        st.markdown(    """B. 1–2–3 Pullbacks (very effective strategy)
Buy:

- 14 day ADX less than 30
- +DI greater than -DI
- Last 3 days should have 3 consecutive lower lows

    """)    

    
    if len(symbolsDate_dict['symbols']) < 1:
        st.info("Please select symbols.")
        st.stop()
            
    prices_df = get_stocks(symbolsDate_dict, stack=True, stack_level='factor')
    benchmark_df = get_stocks(symbolsDate_dict, 'close', benchmark=True)
    
    adx = vbt.talib('ADX').run(prices_df['high'], prices_df['low'], prices_df['close'], timeperiod=14, short_name='adx')
    plus_di = vbt.talib('PLUS_DI').run(prices_df['high'], prices_df['low'], prices_df['close'], timeperiod=14, short_name='plus')
    minus_di = vbt.talib('MINUS_DI').run(prices_df['high'], prices_df['low'], prices_df['close'], timeperiod=14, short_name='minus')

    adx_under = adx.real_below(30)
    cross_plus = plus_di.real_above(minus_di.real)
    
    cross_plus = cross_plus[(14, 14)]
    adx_under = adx_under[14]
    
    consecutive_lows = calculate_consecutive_lows(prices_df, n=5)
    
    entries = (cross_plus & adx_under & (consecutive_lows == 5))
    
    events_df = entries[entries == True]

    run_custom_event_study(symbol_benchmark,
        symbolsDate_dict,
        benchmark_df=benchmark_df,
        stocks_df=prices_df['close'],
        events_df=events_df,
        def_days_before=0,
        def_days_after=3)
