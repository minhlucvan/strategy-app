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
from utils.vbt import init_vbtsetting, plot_CSCV, plot_pf

def rs_momentum_Strategy(symbol_benchmark, stocks_df, RARM_obj= 'sharpe_ratio', output_bool=False):
    stocks_df[stocks_df<0] = np.nan
    symbols_target = []
    
    for s in stocks_df.columns:
        if s != symbol_benchmark:
            symbols_target.append(s)
    sSel = symbols_target

    
    # Build param grid
    windows = [60, 100, 150, 200, 225, 250, 275, 300]

    param_product = vbt.utils.params.create_param_product([windows])
    param_tuples = list(zip(*param_product))
    param_columns = pd.MultiIndex.from_tuples(param_tuples, names=['rs_window'])
    RRG_indicator = get_RRGInd().run(prices=stocks_df[sSel], bm_price=stocks_df[symbol_benchmark], ratio=rs_ratio_mins, momentum=rs_momentum_mins, window=windows, param_product=True)
    sizes = RRG_indicator.size.shift(periods=1)
    
    init_vbtsetting()
    pf_kwargs = dict(fees=0.001, slippage=0.001, freq='1D')
    pf = vbt.Portfolio.from_orders(
                stocks_df[sSel].vbt.tile(len(param_columns), keys=param_columns), 
                sizes, 
                size_type='targetpercent', 
                group_by=param_columns.names,  # group of two columns
                cash_sharing=True,  # share capital between columns
                **pf_kwargs,
            )
    if not isinstance(pf.total_return(), np.float64):
        RARMs = eval(f"pf.{RARM_obj}()")
        idxmax = RARMs[RARMs != np.inf].idxmax()
        # idxmax = pf.total_return().idxmax()
        st.write(f"The Max {RARM_obj} is {param_columns.names}:{idxmax}")
        if output_bool:
           plot_CSCV(pf, idxmax, RARM_obj)

        pf = pf[idxmax]

        if output_bool:
            pass
    return pf


def run(symbol_benchmark, symbolsDate_dict):
    symbolsDate_dict['symbols'] =  [symbol_benchmark] + symbolsDate_dict['symbols']
    stocks_df = get_stocks(symbolsDate_dict,'close')
    # pf = RRG_Strategy(symbol_benchmark, stocks_df)
    # st.write(pf.stats())

    pf = rs_momentum_Strategy(symbol_benchmark, stocks_df, output_bool=True)
    # plot_pf(pf, name= 'RRG Strategy', bm_symbol=symbol_benchmark, bm_price=stocks_df[symbol_benchmark], select=True)
