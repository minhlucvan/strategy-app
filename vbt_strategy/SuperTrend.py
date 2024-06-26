import numpy as np

from numba import njit
import vectorbt as vbt

import talib

from .base import BaseStrategy
from utils.vbt import plot_CSCV


@njit
def get_final_bands_nb(close, upper, lower):
    trend = np.full(close.shape, np.nan)
    dir_ = np.full(close.shape, 1)
    long = np.full(close.shape, np.nan)
    short = np.full(close.shape, np.nan)

    for i in range(1, close.shape[0]):
        if close[i] > upper[i - 1]:
            dir_[i] = 1
        elif close[i] < lower[i - 1]:
            dir_[i] = -1
        else:
            dir_[i] = dir_[i - 1]
            if dir_[i] > 0 and lower[i] < lower[i - 1]:
                lower[i] = lower[i - 1]
            if dir_[i] < 0 and upper[i] > upper[i - 1]:
                upper[i] = upper[i - 1]

        if dir_[i] > 0:
            trend[i] = long[i] = lower[i]
        else:
            trend[i] = short[i] = upper[i]
            
    return trend, dir_, long, short

@njit
def get_basic_bands(med_price, atr, multiplier):
    matr = multiplier * atr
    upper = med_price + matr
    lower = med_price - matr
    return upper, lower

def faster_supertrend_talib(high, low, close, window, multiplier):
    avg_price = np.full(close.shape, np.nan)
    atr = np.full(close.shape, np.nan)
    upper = np.full(close.shape, np.nan)
    lower = np.full(close.shape, np.nan)
    for col in range(close.shape[1]):
        avg_price[:,col] = talib.MEDPRICE(high[:,col], low[:,col])
        atr[:,col] = talib.ATR(high[:,col], low[:,col], close[:,col], window)
        upper[:,col], lower[:,col] = get_basic_bands(avg_price[:,col], atr[:,col], multiplier)
    return get_final_bands_nb(close, upper, lower)       

class SuperTrendStrategy(BaseStrategy):
    '''SuperTrend strategy'''
    _name = "SuperTrend"
    desc = """The SuperTrend is a trend-following indicator that uses the Average True Range (ATR) and the median price to define a set of upper and lower bands. The concept is quite simple: when the closing price crosses above the upper band, the asset is considered to be in an uptrend, which is a buy signal. Conversely, when the closing price falls below the lower band, the asset is considered to have exited the uptrend, which is a sell signal.

The calculation formula for the SuperTrend Indicator is as follows:
When going long:
\[ \text{SuperTrend Indicator} = \frac{\text{Highest Price} + \text{Lowest Price}}{2} - N \times \text{ATR}(M) \]

Reference:
[Superfast SuperTrend](https://medium.datadriveninvestor.com/superfast-supertrend-6269a3af0c2)"""
    param_dict = {}
    param_def = [
            {
            "name": "window",
            "type": "int",
            "min":  1,
            "max":  30,
            "step": 2   
            },
            {
            "name": "multiplier",
            "type": "float",
            "min":  2.0,
            "max":  4.1,
            "step": 0.2   
            },
    ]

    @vbt.cached_method
    def run(self, calledby='add')->bool:
        #1. initialize the variables
        windows = self.param_dict['window']
        multipliers = self.param_dict['multiplier']
        high_price = self.stock_dfs[0][1].high
        low_price = self.stock_dfs[0][1].low
        close_price = self.stock_dfs[0][1].close
        open_price = self.stock_dfs[0][1].open

        #2. calculate the indicators
        SuperTrend = vbt.IndicatorFactory(
                class_name='SuperTrend',
                input_names=['high', 'low', 'close'],
                param_names=['window', 'multiplier'],
                output_names=['supert', 'superd', 'superl', 'supers']
            ).from_apply_func(
                faster_supertrend_talib
            )

        st_indicator = SuperTrend.run(
                high_price, low_price, close_price, 
                window = windows, 
                multiplier = multipliers,
                param_product=True,
            )

        #3. remove all the name in param_def from param_dict
        for param in self.param_def:
            del self.param_dict[param['name']]

        #4. generate the vbt signal
        entries = (~st_indicator.superl.isnull()).vbt.signals.fshift()
        exits = (~st_indicator.supers.isnull()).vbt.signals.fshift()

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

                self.param_dict.update(dict(zip(['window', 'multiplier'], [int(idxmax[0]), round(idxmax[1], 1)])))
        
        self.pf =pf
        return True
