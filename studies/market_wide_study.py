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

from utils.processing import get_stocks
from studies.rrg import plot_RRG, rs_ratio, RRG_Strategy
from utils.vbt import plot_pf
from vbt_strategy.MOM_D import get_MomDInd

from studies.market_wide import MarketWide_Strategy

def run(symbol_benchmark, symbolsDate_dict):
    symbolsDate_dict['symbols'] =  [symbol_benchmark] + symbolsDate_dict['symbols']
    stocks_df = get_stocks(symbolsDate_dict,'close')
    # pf = RRG_Strategy(symbol_benchmark, stocks_df)
    # st.write(pf.stats())


    pf = MarketWide_Strategy(symbol_benchmark, stocks_df, RARM_obj= 'sharpe_ratio', output_bool=True, short=False)

    plot_pf(pf, name="XXX", bm_symbol=symbol_benchmark, bm_price=stocks_df[symbol_benchmark], select=True)
