import pandas as pd
import numpy as np
import json

import streamlit as st
import vectorbt as vbt

import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots

from studies.vn30_volality_study import calculate_vix_index
from utils.component import check_password, input_dates, input_SymbolsDate
import matplotlib.pyplot as plt

from utils.processing import get_stocks
from utils.vbt import plot_pf
from vbt_strategy.MOM_D import get_MomDInd
from studies.stock_factor_base_study import run as run_magic_fomula
import talib as ta
from utils.plot_utils import plot_multi_line, plot_single_bar, plot_single_line, plot_snapshot

import numpy as np
import pandas as pd
import vectorbt as vbt

# The formula for the Accelerator Oscillator is:
# AC = SMA(5) of (Median Price) – SMA(34) of (Median Price)
# where:
# AC = Acceleration Oscillator SMA = Simple Moving Average Median Price = (High + Low) / 2
def calculate_accelerator_oscillator(high, low, close):
    # median_price = (high + low) / 2
    median_price = (high + low + close) / 3
    sma5 = median_price.rolling(window=5).mean()
    sma34 = median_price.rolling(window=34).mean()
    return sma5 - sma34

def run(symbol_benchmark, symbolsDate_dict):
    st.write("## Volality Factor Study")
    
    stocks_df = get_stocks(symbolsDate_dict, 'close')
    
    high = get_stocks(symbolsDate_dict, 'high')
    low = get_stocks(symbolsDate_dict, 'low')
    
    plot_multi_line(stocks_df, title='Stocks', x_title='Date', y_title='Price', legend_title='Stocks')

    vix_index = calculate_vix_index(stocks_df)
    
   
    ao = calculate_accelerator_oscillator(high, low, stocks_df)
    
    for col in ao.columns:
        st.write(f"### Accelerator Oscillator for {col}")
        plot_single_bar(ao[col], title=f"Accelerator Oscillator for {col}", x_title='Date', y_title='Price', price_df=stocks_df[col])