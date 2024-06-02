import numpy as np

from numba import njit
import streamlit as st
import vectorbt as vbt
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import talib as ta

from .base import BaseStrategy
from utils.vbt import plot_CSCV


def plot_momd(close_prices, mom_indicator, idxmax, RARM):
    indicator_values = mom_indicator.mom_pct[idxmax]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.02)
    fig.add_trace(go.Scatter(x=close_prices.index, y=close_prices,
                  mode='lines', name='price'), row=1, col=1)
    # add bar chart for indicator_values, color green for positive, red for negative
    fig.add_trace(go.Bar(x=close_prices.index, y=indicator_values, name='mom_pct', marker=dict(
        color=np.where(indicator_values > 0, 'green', 'red'))), row=2, col=1)
    fig.update_layout(height=600, width=800, title_text="MOMD Strategy")
    st.plotly_chart(fig, use_container_width=True)


@njit
def apply_mom_nb(price, window, smooth_period, entry_threshold, exit_threshold):
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
            price_changes = np.full(window, np.nan, dtype=np.float_)
            for level in range(1, window+1):
                price_change = (price[i, col] - price[i-level, col])
                price_changes[level-1] = price_change

            price_change_pct = np.sum(price_changes, axis=0)
            price_change_pct_mean = price_change_pct / window

            mom_pct[i, col] = price_change_pct_mean

    # smooth the signal by using the rolling mean 5 (numba)
    mom_pct_sum = np.full(price.shape, np.nan, dtype=np.float_)
    for col in range(price.shape[1]):
        for i in range(window, price.shape[0]):
            mom_pct_sum[i, col] = np.sum(mom_pct[i-smooth_period+1:i+1, col])

    mom_pct = mom_pct_sum / smooth_period

    entry_signal = mom_pct > entry_threshold
    exit_signal = mom_pct < exit_threshold

    return mom_pct, entry_signal, exit_signal


def get_MomDInd():
    MomInd = vbt.IndicatorFactory(
        class_name='MomD',
        input_names=['price'],
        param_names=['window', 'smooth_period',
                     'entry_threshold', 'exit_threshold'],
        output_names=['mom_pct', 'entry_signal', 'exit_signal']
    ).from_apply_func(apply_mom_nb)

    return MomInd


class MOM_DStrategy(BaseStrategy):
    '''Mom strategy'''
    _name = "MOMD"
    desc = """The Momentum dynamic strategy is used to depict the momentum of the stock.
    If the average of the momentum is greater than 0, hold the position; otherwise, sell.
    The momentum is calculated by the difference between the current price and the price of the previous days (mean).
    """
    param_dict = {}
    param_def = [
        {
            "name": "window",
            "type": "int",
            "min":  2,
            "max":  30,
            "step": 1
        },
        {
            "name": "smooth_period",
            "type": "int",
            "min":  1,
            "max":  20,
            "step": 1
        },
        {
            "name": "entry_threshold",
            "type": "int",
            "min":  0,
            "max":  10,
            "step": 1
        },
        {
            "name": "exit_threshold",
            "type": "int",
            "min": -10,
            "max":  0,
            "step": 1
        },
    ]

    @vbt.cached_method
    def run(self, calledby='add'):
        # 1. initialize the variables
        window = self.param_dict['window']
        smooth_period = self.param_dict['smooth_period']
        entry_threshold = self.param_dict['entry_threshold']
        exit_threshold = self.param_dict['exit_threshold']

        close_price = self.stock_dfs[0][1].close
        open_price = self.stock_dfs[0][1].open

        # 2. calculate the indicators
        mom_indicator = get_MomDInd().run(close_price, window=window, smooth_period=smooth_period,
                                          entry_threshold=entry_threshold, exit_threshold=exit_threshold, param_product=True)

        # 3. remove all the name in param_def from param_dict
        for param in self.param_def:
            del self.param_dict[param['name']]

        # 4. generate the vbt signal
        entries = mom_indicator.entry_signal.vbt.signals.fshift()
        exits = mom_indicator.exit_signal.vbt.signals.fshift()

        # 5. Build portfolios
        if self.param_dict['WFO'] != 'None':
            entries, exits = self.maxRARM_WFO(
                close_price, entries, exits, calledby)
            pf = vbt.Portfolio.from_signals(
                close=close_price, open=open_price, entries=entries, exits=exits, **self.pf_kwargs)
        else:
            pf = vbt.Portfolio.from_signals(
                close=close_price, open=open_price, entries=entries, exits=exits, **self.pf_kwargs)
            if calledby == 'add':
                RARMs = eval(f"pf.{self.param_dict['RARM']}()")
                idxmax = RARMs[RARMs != np.inf].idxmax()
                if self.output_bool:
                    plot_momd(close_price, mom_indicator,
                              idxmax, self.param_dict['RARM'])
                    plot_CSCV(pf, idxmax, self.param_dict['RARM'])
                pf = pf[idxmax]

                self.param_dict.update(
                    dict(
                        zip(
                            ['window',
                             'smooth_period',
                             'entry_threshold',
                             'exit_threshold'],
                            [int(idxmax[0]), int(idxmax[1]), int(idxmax[2]), int(idxmax[3])]))
                )
        self.pf = pf
        return True
