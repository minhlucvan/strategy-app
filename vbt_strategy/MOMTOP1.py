import numpy as np
import pandas as pd
from datetime import timezone

import streamlit as st
import vectorbt as vbt

from indicators.EventArb import generate_arbitrage_signal, get_EventArbInd
from utils.processing import get_stocks_events

from .base import BaseStrategy
from utils.vbt import init_vbtsetting, plot_CSCV

def compute_sizes(df):
    """
    Compute sizes for MOMTOP1: Rolling buy the best return asset from the previous bar and hold for one bar.
    """
    sizes_df = pd.DataFrame(index=df.index, columns=df.columns, data=0.0)
    
    # Calculate daily returns
    returns = df.pct_change()
    
    for i in range(1, len(df)):
        prev_returns = returns.iloc[i - 1]
        if not prev_returns.empty:
            best_asset = prev_returns.idxmax()
            st.write(prev_returns, best_asset)
            st.stop()
            sizes_df.iloc[i][best_asset] = 1.0  # Allocate fully to the best performing asset from the previous bar
    
    return sizes_df


class MOMTOP1Strategy(BaseStrategy):
    '''Momentum Top 1 strategy'''
    _name = "MOMTOP1"
    desc = "This strategy selects the best-performing stock based on previous bar returns and holds for one bar."
    stacked_bool = True

    @vbt.cached_method
    def run(self, calledby='add'):
        stocks_df = self.stocks_df.dropna(axis=1, how='any')
        stocks_df = stocks_df.fillna(method='ffill')
        
        
        sizes = compute_sizes(stocks_df)
        st.write(sizes)
        
        init_vbtsetting()
        self.pf_kwargs = dict(fees=0.001, slippage=0.001, freq='1D')

        # Build portfolio
        if self.param_dict['WFO'] != 'None':
            raise NotImplementedError('WFO not implemented')
        else:
            pf = vbt.Portfolio.from_orders(
                stocks_df,
                sizes,
                size_type='targetpercent',
                group_by=True,
                cash_sharing=True,
                **self.pf_kwargs,
            )
            
            if calledby == 'add':
                RARMs = eval(f"pf.{self.param_dict['RARM']}()")
                if isinstance(RARMs, pd.Series):
                    idxmax = RARMs[RARMs != np.inf].idxmax()
                    if self.output_bool:
                        plot_CSCV(pf, idxmax, self.param_dict['RARM'])
                    pf = pf[idxmax]
            
        self.pf = pf
        return True
