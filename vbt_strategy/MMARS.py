import numpy as np
import pandas as pd

import streamlit as st
import vectorbt as vbt

from .base import BaseStrategy
from utils.vbt import plot_CSCV


class MMARSStrategy(BaseStrategy):
    '''MMARS strategy'''
    _name = "MMARS"
    desc = """This is a Multi-Moving-Average RS strategy that requires two moving averages: a fast line and a slow line."""
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
    stacked_bool = True
    use_rsc = True
    bm_symbol = 'VN30'
    include_bm = True

    # for multi symbols
    @vbt.cached_method
    def run(self, output_bool=False, calledby='add') -> bool:
        stocks_df = self.stocks_df

        if calledby == 'add' or self.param_dict['WFO']:
            window = self.param_dict['window']
            fast_ma, slow_ma = vbt.MA.run_combs(
                stocks_df, window=window, r=2, short_names=['fast', 'slow'])
        else:
            fast_windows = self.param_dict['fast_window']
            slow_windows = self.param_dict['slow_window']
            fast_ma = vbt.MA.run(stocks_df, fast_windows)
            slow_ma = vbt.MA.run(stocks_df, slow_windows)

        entries = fast_ma.ma_above(slow_ma, level_name='entry')
        exits = fast_ma.ma_below(slow_ma, level_name='exit')
        
        # Don't look into the future
        entries = entries.vbt.signals.fshift()
        exits = exits.vbt.signals.fshift()
        
        num_symbol = len(stocks_df.columns)
        num_group = int(len(entries.columns) / num_symbol)
        
        group_list = []
        for n in range(num_group):
            group_list.extend([n]*num_symbol)
        group_by = pd.Index(group_list, name='group')

        if self.param_dict['WFO'] != 'None':
            entries, exits = self.maxRARM_WFO(
                stocks_df, entries, exits, 'y', group_by)
            pf = vbt.Portfolio.from_signals(stocks_df,
                                            entries=entries, exits=exits,
                                            cash_sharing=True,
                                            **self.pf_kwargs)
            self.param_dict = {'WFO': self.param_dict['WFO']}
        else:
            pf = vbt.Portfolio.from_signals(stocks_df,
                                            entries=entries,
                                            exits=exits,
                                            cash_sharing=True,
                                            group_by=group_by,
                                            **self.pf_kwargs)
            if calledby == 'add':
                SRs = pf.sharpe_ratio()
                idxmax = SRs[SRs != np.inf].idxmax()
                if output_bool:
                    plot_CSCV(pf, idxmax)
                pf = pf[idxmax]
                params_value = entries.columns[idxmax*num_symbol]
                self.param_dict = dict(zip(['fast_window', 'slow_window'], [
                                       int(params_value[0]), int(params_value[1])]))
        self.pf = pf
        return True
