# @vbt.cached_method
import streamlit as st
import numpy as np
import pandas as pd
import vectorbt as vbt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from indicators.AnySign import get_AnySignInd
from utils.vbt import init_vbtsetting, plot_CSCV
from vbt_strategy.MOM_D import get_MomDInd


def plot_momd(benchmark_close, stocks_df, mom_indicator):
    mom_indicator_entry = mom_indicator.entry_signal.vbt.signals.fshift()

    mom_indicator_exit = mom_indicator.exit_signal.vbt.signals.fshift()

    # count the number of signals accross all symbols
    mom_indicator_entry_count = mom_indicator_entry.sum(axis=1)

    mom_indicator_exit_count = mom_indicator_exit.sum(axis=1)

    mom_indicator_sum = mom_indicator_entry_count - mom_indicator_exit_count

    # plot the signals
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02)
    fig.add_trace(go.Scatter(x=stocks_df.index, y=benchmark_close, mode='lines', name='benchmark'), row=1, col=1)
    # plot the signals as bars on the second subplot, color green for positive, red for negative
    fig.add_trace(go.Bar(x=stocks_df.index, y=mom_indicator_sum, name='momd', marker=dict(color=np.where(mom_indicator_sum > 0, 'green', 'red'))), row=2, col=1)
    st.plotly_chart(fig, use_container_width=True)



def MarketWide_Strategy(symbol_benchmark, stocks_df, RARM_obj= 'sharpe_ratio', output_bool=False, short=False):
    stocks_df[stocks_df<0] = np.nan
    symbols_target = []
    for s in stocks_df.columns:
        if s != symbol_benchmark:
            symbols_target.append(s)
    sSel = symbols_target
    benchmark_close = stocks_df[symbol_benchmark]

    windows = 24
    smooth_period = 5
    entry_threshold = 1
    exit_threshold = -1
    
    
    # param_product = vbt.utils.params.create_param_product([windows, smooth_period, entry_threshold, exit_threshold])
    # param_tuples = list(zip(*param_product))
    # param_columns = pd.MultiIndex.from_tuples(param_tuples, names=['rs_ratio', 'rs_momentum', 'rs_window'])

    mom_indicator = get_MomDInd().run(stocks_df, window=windows, smooth_period=smooth_period, entry_threshold=entry_threshold, exit_threshold=exit_threshold, param_product=False)
   
    mom_indicator_entry = mom_indicator.entry_signal.vbt.signals.fshift()

    mom_indicator_exit = mom_indicator.exit_signal.vbt.signals.fshift()

    # count the number of signals accross all symbols
    mom_indicator_entry_count = mom_indicator_entry.sum(axis=1)

    mom_indicator_exit_count = mom_indicator_exit.sum(axis=1)

    mom_indicator_sum = mom_indicator_entry_count - mom_indicator_exit_count
    
    count_entry_threshold = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    count_exit_threshold = [-1, -2, -3, -4, -5, -6, -7, -8, -9, -10]
        
    count_indicator = get_AnySignInd().run(
        benchmark_close,
        mom_indicator_sum,
        entry_threshold=count_entry_threshold,
        exit_threshold=count_exit_threshold,
        param_product=True,
    )
   
    entries = count_indicator.entry_signal.vbt.signals.fshift()
    exits = count_indicator.exit_signal.vbt.signals.fshift()
    
    shorts_entries = exits.copy() if short else None
    shorts_exits = entries.copy() if short else None
    
    init_vbtsetting()
    pf_kwargs = dict(fees=0.001, slippage=0.001, freq='1D')
    pf = vbt.Portfolio.from_signals(
        close=benchmark_close,
        entries=entries,
        exits=exits,
        short_entries=shorts_entries,
        short_exits=shorts_exits,
        cash_sharing=True,
        **pf_kwargs
    )
    
    if not isinstance(pf.total_return(), np.float64):
        RARMs = eval(f"pf.{RARM_obj}()")
        idxmax = RARMs[RARMs != np.inf].idxmax()
        # idxmax = pf.total_return().idxmax()
        if output_bool:
           plot_CSCV(pf, idxmax, RARM_obj)

        pf = pf[idxmax]

    if output_bool:
        plot_momd(benchmark_close, stocks_df, mom_indicator)
    

    return pf
