import numpy as np

from numba import njit
import streamlit as st
import vectorbt as vbt
import pandas as pd

from .base import BaseStrategy
from utils.vbt import plot_CSCV

@njit
def apply_mom_nb(price, window):
    mom_pct = np.full(price.shape, np.nan, dtype=np.float_)
    entry_signal = np.full(price.shape, np.nan, dtype=np.bool_)
    exit_signal = np.full(price.shape, np.nan, dtype=np.bool_)
    
    # print(price.shape) -> (1591, 1)
    
    # for each level in window_levels
    # caculate the price - price shifted by level
    # if the price is greater than 0, then entry_signal is True
    # if the price is less than 0, then exit_signal is True
    for col in range(price.shape[1]):
        for i in range(window, price.shape[0]):
            price_change = 0
            for level in range(1, window+1):
                price_change += price[i,col] - price[i-level,col]
                
            price_change_pct = price_change / price[i,col]
            price_change_pct_mean = price_change_pct / window
            
            mom_pct[i,col] = price_change_pct_mean
            exit_signal[i,col] = (price_change_pct_mean < 0)
            entry_signal[i,col] = (price_change_pct_mean > 0)
            
    return mom_pct, entry_signal, exit_signal

def get_MomDInd():
    MomInd = vbt.IndicatorFactory(
        class_name = 'MomD',
        input_names = ['price'],
        param_names = ['window'],
        output_names = ['mom_pct','entry_signal', 'exit_signal']
    ).from_apply_func(apply_mom_nb)
    
    return MomInd

class MOM_DStrategy(BaseStrategy):
    '''Mom strategy'''
    _name = "MOMD"
    desc = "..."
    param_dict = {}
    param_def = [
        {
        "name": "window",
        "type": "int",
        "min":  2,
        "max":  30,
        "step": 1  
        }
    ]

    @vbt.cached_method
    def run(self, calledby='add'):
        #1. initialize the variables
        window = self.param_dict['window']
        close_price = self.stock_dfs[0][1].close
        open_price = self.stock_dfs[0][1].open
        
        #2. calculate the indicators
        mom_indicator = get_MomDInd().run(close_price, window=window, param_product=True)

        #3. remove all the name in param_def from param_dict
        for param in self.param_def:
            del self.param_dict[param['name']]

        #4. generate the vbt signal
        entries = mom_indicator.entry_signal.vbt.signals.fshift()
        exits = mom_indicator.exit_signal.vbt.signals.fshift()

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

                self.param_dict['window'] = int(idxmax)

        self.pf = pf
        return True