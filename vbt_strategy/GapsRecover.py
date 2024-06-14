import numpy as np
import pandas as pd
import talib
from itertools import combinations

import streamlit as st
import vectorbt as vbt

from indicators.Gap import get_GapInd
from utils.plot_utils import plot_multi_bar, plot_multi_line
from utils.processing import get_stocks

from .base import BaseStrategy
from utils.vbt import plot_CSCV

from numba import njit

class GapsRecoverStrategy(BaseStrategy):
    '''GapsRecover strategy'''
    
    _name = "GapsRecover"
    desc = "Find Gap down and capture the recover"
    stacked_bool = True
    value_filter = True
    # include_bm = True
    param_def = [
            {
            "name": "gap",
            "type": "float",
            "min":  0.02,
            "max":  0.1,
            "step": 0.01  
            },
            {
                "name": "exit_bars",
                "type": "int",
                "min": 3,
                "max": 15,
                "step": 1
            }
        ]

    def run(self, calledby='add')->bool:
        #1. initialize the variables
        close_price = self.stocks_df
        open_price = get_stocks(self.symbolsDate_dict, 'open')
        gap = self.param_dict['gap']
        exit_bars = self.param_dict['exit_bars']
                
        indicator = get_GapInd().run(
            open_price,
            close_price,
            gap_percent=gap,
            exit_bars=exit_bars,
            param_product=True
        )
        
        #3. remove all the name in param_def from param_dict
        for param in self.param_def:
            del self.param_dict[param['name']]
        
        # Don't look into the future
        # Look at the present becase we are are in the beginning of the bar
        entries = indicator.entries 
        exits = indicator.exits
        gaps = indicator.gaps
        
        # st.write("GapsRecover Strategy")
        # st.write(entries)
        # st.write("Exits")
        # st.write(exits)
        # st.write("Gaps")
        # st.write(gaps)
        
        num_symbol = len(self.stocks_df.columns)
        num_group = int(len(entries.columns) / num_symbol)
        group_list = []
        for n in range(num_group):
            group_list.extend([n]*num_symbol)
        group_by = pd.Index(group_list, name='group')
        
        #5. Build portfolios
        if 'WFO' in  self.param_dict['WFO'] and  self.param_dict['WFO']!='None':
            entries, exits = self.maxRARM_WFO(close_price, entries, exits, calledby)
            pf = vbt.Portfolio.from_signals(
                # close=close_price,
                open_price,
                entries=entries,
                exits=exits,
                group_by=group_by,
                **self.pf_kwargs)
            
        else:
            pf = vbt.Portfolio.from_signals(
                # close=close_price,
                open_price,
                entries=entries,
                exits=exits,
                # size=1,
                # size_type='percent',
                group_by=group_by,
                **self.pf_kwargs)
            
            if calledby == 'add':
                RARMs = eval(f"pf.{self.param_dict['RARM']}()")
                if isinstance(RARMs, pd.Series):
                    idxmax = RARMs[RARMs != np.inf].idxmax()
                    
                    if self.output_bool:
                        plot_CSCV(pf, idxmax, self.param_dict['RARM'])
                        
                    pf = pf[idxmax]
                
                    params_value = entries.columns[idxmax*num_symbol]
                    self.param_dict.update(dict(zip(['gap', 'exit_bars'], [float(params_value[0]), int(params_value[1])])))
                else:
                    idxmax = (gap[0], exit_bars[0])
                    self.param_dict(dict(zip(['gap', 'exit_bars'], [gap[0], exit_bars[0]])))
        
        self.pf = pf
        return True

