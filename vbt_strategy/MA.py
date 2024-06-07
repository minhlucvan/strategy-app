import numpy as np
import pandas as pd

import streamlit as st
import vectorbt as vbt

from .base import BaseStrategy
from utils.vbt import plot_CSCV

class MAStrategy(BaseStrategy):
    '''MA strategy'''
    _name = "MA"
    desc = """This is a trend-following strategy that requires two moving averages: a fast line and a slow line.
     When the fast line crosses above the slow line from below, it forms a golden cross,
    signaling an entry to go long (buy). Conversely, when the fast line crosses below the slow line from above,
    it forms a death cross, signaling an entry to go short (sell).
    **Parameters**:
    - `window`: The window size for the moving average calculation. `fast_window` and `slow_window` are calculated based on this window range.
    """
    param_def = [
            {
            "name": "window",
            "type": "int",
            "min":  1,
            "max":  80,
            "step": 2,
            "default": (10, 20)  
            },
        ]

    @vbt.cached_method
    def run(self, calledby='add')->bool:
        #1. initialize the variables
        close_price = self.stock_dfs[0][1].close
        open_price = self.stock_dfs[0][1].open
        
        ma_src = self.get_ma_src()
        
        #2. calculate the indicators
        if calledby == 'add' or self.param_dict['WFO']!='None':
            window = self.param_dict['window']
            fast_ma, slow_ma = vbt.MA.run_combs(ma_src, window=window, r=2, short_names=['fast', 'slow'])
        else:
            fast_windows = self.param_dict['fast_window']
            slow_windows = self.param_dict['slow_window']
            fast_ma = vbt.MA.run(ma_src, fast_windows)
            slow_ma = vbt.MA.run(ma_src, slow_windows)

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

    def get_ma_src(self):
        return  self.stock_dfs[0][1].close