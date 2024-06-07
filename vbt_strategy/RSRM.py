import numpy as np
import pandas as pd

import streamlit as st
import vectorbt as vbt

from .base import BaseStrategy
from utils.vbt import plot_CSCV

from studies.rrg import rs_ratio

def apply_RSRM(price, window):
    mom_pct = np.full(price.shape, np.nan, dtype=np.float_)
    entry_signal = np.full(price.shape, np.nan, dtype=np.bool_)
    exit_signal = np.full(price.shape, np.nan, dtype=np.bool_)
    for col in range(price.shape[1]):
        for i in range(window, price.shape[0]):
            pct_change = price[i,col]/price[i-window,col] - 1
            mom_pct[i,col] = pct_change
            exit_signal[i,col] = (pct_change < lower)
            entry_signal[i,col] = (pct_change > upper)
            
    return mom_pct, entry_signal, exit_signal

def get_RSRMInd():
    MomInd = vbt.IndicatorFactory(
        class_name = 'RSRM',
        input_names = ['price', 'symbol_benchmark'],
        param_names = ['window'],
        output_names = ['entry_signal', 'exit_signal']
    ).from_apply_func(apply_RSRM)
    
    return MomInd

class RSRMtrategy(BaseStrategy):
    '''Relative Strength Ratio-Mnmentum strategy'''
    _name = "RSRM"
    desc = "The Relative Rotation Graph (RRG) is used to depict the cycles of excess returns for various industries. If the average of the RS Ratio and the RS Ratio Momentum ((RS Ratio + RS Ratio Momentum) / 2) is greater than 100, hold the position; otherwise, sell."
    param_def = [
            {
            "name": "window",
            "type": "int",
            "min":  1,
            "max":  20,
            "step": 1   
            },
        ]

    @vbt.cached_method
    def run(self, calledby='add')->bool:
        #1. initialize the variables
        close_price = self.stock_dfs[0][1].close
        open_price = self.stock_dfs[0][1].open

        #2. calculate the indicators
        if calledby == 'add' or self.param_dict['WFO']!='None':
            window = self.param_dict['window']
            fast_ma, slow_ma = vbt.MA.run_combs(close_price, window=window, r=2, short_names=['fast', 'slow'])
        else:
            fast_windows = self.param_dict['fast_window']
            slow_windows = self.param_dict['slow_window']
            fast_ma = vbt.MA.run(close_price, fast_windows)
            slow_ma = vbt.MA.run(close_price, slow_windows)

        #3. remove all the name in param_def from param_dict
        for param in self.param_def:
            del self.param_dict[param['name']]

        #4. generate the vbt signal
        entries = fast_ma.ma_above(slow_ma)
        exits = fast_ma.ma_below(slow_ma)
        
        #Don't look into the future
        entries = entries.vbt.signals.fshift()
        exits = exits.vbt.signals.fshift()

        #5. Build portfolios
        if self.param_dict['WFO']!='None':
            entries, exits = self.maxRARM_WFO(close_price, entries, exits, calledby)
            pf = vbt.Portfolio.from_signals(close=close_price, open=open_price, entries=entries, exits=exits, **self.pf_kwargs)
        else:
            pf = vbt.Portfolio.from_signals(close=close_price, open=open_price, entries=entries, exits=exits, **self.pf_kwargs)
            if calledby == 'add':
                RARMs = eval(f"pf.{self.param_dict['RARM']}()")
                idxmax = RARMs[RARMs != np.inf].idxmax()
                if self.output_bool:
                    plot_CSCV(pf, idxmax, self.param_dict['RARM'])
                pf = pf[idxmax]

                self.param_dict['fast_window'] = int(idxmax[0])
                self.param_dict['slow_window'] =  int(idxmax[1])

        self.pf = pf
        return True

    