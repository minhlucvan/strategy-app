import pandas as pd
import numpy as np
import json

import streamlit as st
import vectorbt as vbt

import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots

from utils.component import check_password, input_dates, input_SymbolsDate
import matplotlib.pyplot as plt

from utils.processing import get_stocks
from studies.rrg import plot_RRG, rs_ratio, RRG_Strategy
from utils.vbt import plot_pf
from vbt_strategy.MOM_D import get_MomDInd
from studies.stock_factor_base_study import run as run_magic_fomula
import talib as ta
from utils.plot_utils import plot_multi_line, plot_single_line, plot_snapshot

import numpy as np
import pandas as pd

def magic_formula(metrics):

    return 0

def run(symbol_benchmark, symbolsDate_dict):
    st.write("## Momentum Factor Study")
    
    stocks_df = get_stocks(symbolsDate_dict, 'close')
    
    high = get_stocks(symbolsDate_dict, 'high')
    low = get_stocks(symbolsDate_dict, 'low')
    
    plot_multi_line(stocks_df, title='Stocks', x_title='Date', y_title='Price', legend_title='Stocks')
    
    # Calculate momentum rsi
    stocks_macd = vbt.MACD.run(stocks_df)
    
    stocks_macd_df = stocks_macd.macd  
        
    plot_multi_line(stocks_macd_df, title='RSI', x_title='Date', y_title='RSI', legend_title='Stocks')

    benchmark_macd = stocks_macd_df.mean(axis=1)
    
    plot_single_line(benchmark_macd, title='Benchmark RSI', x_title='Date', y_title='RSI', legend_title='Benchmark RSI')
    