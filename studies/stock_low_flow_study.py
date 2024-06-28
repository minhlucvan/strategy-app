import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
from utils.plot_utils import plot_multi_bar, plot_multi_line, plot_multi_scatter
from utils.processing import get_stocks
import utils.stock_utils as stock_utils


def run(symbol_benchmark, symbolsDate_dict):
    
    # limit list to 30
    symbolsDate_dict['symbols'] = symbolsDate_dict['symbols'][:50]
    
    prices_df = get_stocks(symbolsDate_dict, 'close', value_filter=False)
   
    plot_multi_line(prices_df, title='Stock Prices', x_title='Date', y_title='Price', legend_title='Stocks')
    
    prices_pct_change = prices_df.pct_change().cumsum()
    
    plot_multi_line(prices_pct_change, title='Stock Prices % Change', x_title='Date', y_title='Price % Change', legend_title='Stocks')
    
    