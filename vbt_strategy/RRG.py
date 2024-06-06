import numpy as np
import pandas as pd
from datetime import timezone

import streamlit as st
import vectorbt as vbt

from indicators.EventArb import generate_arbitrage_signal, get_EventArbInd
from studies.rrg import RRG_Strategy
from utils.processing import get_stocks_events

from .base import BaseStrategy
from utils.vbt import plot_CSCV

class RRGStrategy(BaseStrategy):
    '''Relative Rotation Graph strategy'''
    _name = "RRG"
    desc = "Relative Rotation Graph strategy aiming to capture the price difference between the benchmark and the stocks. the strategy shown that for a bunch of stocks, the price of the stock will be raised before the benchmark price and drop after the benchmark price, the strategy aims to capture the price difference between the two dates"
    stacked_bool = True
    output_bool = False
    bm_symbol = 'VN30'
    inlude_bm = True

    @vbt.cached_method
    def run(self, calledby='add'):
        stocks_df = self.stocks_df
        self.bm_price = stocks_df[self.bm_symbol]
           
        # 5. Build portfolios
        if self.param_dict['WFO'] != 'None':
            raise NotImplementedError('WFO not implemented')
        else:
            pf, params_dict = RRG_Strategy(self.bm_symbol, stocks_df, output_bool=True, ret_param=True, RARM_obj=self.param_dict['RARM'])
            if calledby == 'add':
                RARMs = eval(f"pf.{self.param_dict['RARM']}()")
                if isinstance(RARMs, pd.Series):
                    idxmax = RARMs[RARMs != np.inf].idxmax()
                    if self.output_bool:
                        plot_CSCV(pf, idxmax, self.param_dict['RARM'])
                    pf = pf[idxmax]
                
                    self.param_dict.update(params_dict)
                else:
                    self.param_dict.update(params_dict)
        self.pf = pf
        return True
