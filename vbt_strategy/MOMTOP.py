import numpy as np
import pandas as pd
from datetime import timezone

import streamlit as st
import vectorbt as vbt

from indicators.EventArb import generate_arbitrage_signal, get_EventArbInd
from studies.momentum_top_study import compute_sizes
from utils.processing import get_stocks_events

from .base import BaseStrategy
from utils.vbt import init_vbtsetting, plot_CSCV


class MOMTOPStrategy(BaseStrategy):
    '''Momentum Top strategy'''
    _name = "MOMTOP"
    desc = "This strategy aims to capture top performing stocks in the market by using a multi-level momentum filter. It identifies stocks with strong momentum and selects the top performers based on a specified number of top stocks."
    stacked_bool = True
    timeframe = 'W'

    @vbt.cached_method
    def run(self, calledby='add'):
        stocks_df = self.stocks_df.dropna(axis=1, how='any')
        
        filters = [
            [50, 32],
            [30, 16],
            [20, 8],
            [10, 4],
            [2, 2]
        ]
        
        sizes = compute_sizes(stocks_df, filters)

        # reindex size to the same index as stocks_df
        sizes = sizes.reindex(stocks_df.index)
                        
        init_vbtsetting()
        self.pf_kwargs = dict(fees=0.001, slippage=0.001, freq='1D')

        # 5. Build portfolios
        if self.param_dict['WFO'] != 'None':
            raise NotImplementedError('WFO not implemented')
        else:
            pf = vbt.Portfolio.from_orders(
                    stocks_df, 
                    sizes, 
                    size_type='targetpercent', 
                    group_by=True,
                    cash_sharing=True,  # share capital between columns
                    **self.pf_kwargs,
                )
            if calledby == 'add':
                RARMs = eval(f"pf.{self.param_dict['RARM']}()")
                if isinstance(RARMs, pd.Series):
                    idxmax = RARMs[RARMs != np.inf].idxmax()
                    if self.output_bool:
                        plot_CSCV(pf, idxmax, self.param_dict['RARM'])
                    pf = pf[idxmax]

                    # self.param_dict.update(params_dict)
                else:
                    # self.param_dict.update(params_dict)
                    pass
                    
        self.pf = pf
        return True
