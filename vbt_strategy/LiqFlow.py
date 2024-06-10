import numpy as np

from numba import njit
import streamlit as st
import vectorbt as vbt
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import talib as ta

from indicators.AnySign import get_AnySignInd
from studies.stock_liqudity_flow_study import plot_liquidity_bars
from utils.processing import get_stocks

from .base import BaseStrategy
from utils.vbt import plot_CSCV


class LiqFlowStrategy(BaseStrategy):
    '''Liquidity Flow Strategy'''
    _name = "LiqFlow"
    desc = """Liquidity Flow Strategy is a strategy that uses the liquidity flow indicator to generate buy and sell signals."""
    param_dict = {}
    param_def = [
        {
            "name": "window",
            "type": "int",
            "min":  2,
            "max":  100,
            "step": 2
        },
    ]
    stacked_bool = True

    @vbt.cached_method
    def run(self, calledby='add'):
        # 1. initialize the variables
        window = 21
        # smonth_period = self.param_dict['window']
        # symbol = self.symbolsDate_dict['symbols'][0]
        close_price = self.stocks_df

        liquidity_change_flow_df = get_stocks(
            self.symbolsDate_dict, 'value_change_weighted')

        liquidity_change_flow_df = liquidity_change_flow_df.rolling(
            window=window).sum()
        
        # fill na
        liquidity_change_flow_df.fillna(0, inplace=True)
    
        # 2. calculate the indicators
        indicator = get_AnySignInd().run(
            close_price,
            liquidity_change_flow_df,
            entry_threshold=0,
            exit_threshold=0,
            param_product=False,
        )

        # 3. remove all the name in param_def from param_dict
        for param in self.param_def:
            del self.param_dict[param['name']]

        # 4. generate the vbt signal
        entries = indicator.entry_signal.vbt.signals.fshift()
        exits = indicator.exit_signal.vbt.signals.fshift()

        # 5. Build portfolios
        if self.param_dict['WFO'] != 'None':
            entries, exits = self.maxRARM_WFO(
                close_price, entries, exits, calledby)
            pf = vbt.Portfolio.from_signals(
                close=close_price,
                entries=entries,
                exits=exits,
                **self.pf_kwargs)
        else:
            pf = vbt.Portfolio.from_signals(
                close=close_price,
                entries=entries,
                exits=exits,
                **self.pf_kwargs)
            if calledby == 'add':
                RARMs = eval(f"pf.{self.param_dict['RARM']}()")
                idxmax = RARMs[RARMs != np.inf].idxmax()
                if self.output_bool:
                    # plot_liquidity_bars(liquidity_change_flow_df[symbol], title='Stocks Foregin Flow Change', x_title='Date', y_title='Value change', legend_title='Stocks')
                    plot_CSCV(pf, idxmax, self.param_dict['RARM'])
                pf = pf[idxmax]

        self.pf = pf
        return True
