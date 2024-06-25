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
    include_bm = True
    bm_symbol = 'VN30'
    

    @vbt.cached_method
    def run(self, calledby='add'):
        stocks_df = self.stocks_df
        bm_df = self.bm_price
                
        if stocks_df.empty:
            st.warning("No data available.")
            st.stop()
         
        rs_ratio = [98, 99, 100, 101, 102]
        rs_momentum = [98, 98.5, 99, 99.5, 100, 100.5, 101, 101.5, 102]
        rs_window = [60, 100, 150, 200, 225, 250, 275, 300]
        
        if calledby == 'update':
            rs_ratio = self.param_dict['rs_ratio']
            rs_momentum = self.param_dict['rs_momentum']
            rs_window = self.param_dict['rs_window']
    

        # 5. Build portfolios
        if self.param_dict['WFO'] != 'None':
            raise NotImplementedError('WFO not implemented')
        else:
            pf, params_dict = RRG_Strategy(
                self.bm_symbol,
                stocks_df,
                output_bool=self.output_bool,
                ret_param=True,
                RARM_obj=self.param_dict['RARM'],
                bm_df=bm_df,
                rs_momentum_mins=rs_momentum,
                rs_ratio_mins=rs_ratio,
                windows=rs_window,
            )
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
