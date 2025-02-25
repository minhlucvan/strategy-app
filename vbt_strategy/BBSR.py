import numpy as np
import pandas as pd
import vectorbt as vbt
import talib
from .base import BaseStrategy

from talib import MA_Type
import numpy as np
import pandas as pd
import vectorbt as vbt
from utils.vbt import plot_CSCV


def BBSRDef(close, length=20, mult=2.0, lengthRSI=14, upperlimit=70, lowerlimit=30):
    # Handle close as either Pandas Series or NumPy array
    if isinstance(close, pd.Series):
        index = close.index
        close = close.values  # Convert to NumPy array for TA-Lib
    else:
        close = np.asarray(close)
        index = pd.RangeIndex(len(close))  # Default index if none provided
    
    # Bollinger Bands calculation with T3
    upper, middle, lower = talib.BBANDS(close, timeperiod=length, 
                                      nbdevup=mult, nbdevdn=mult, 
                                      matype=MA_Type.T3)
    
    # RSI calculation
    rsi = talib.RSI(close, lengthRSI)
    
    # Stochastic Calculation
    # stoch = talib.STOCH(close, close, close, fastk_period=5, slowk_period=3, slowd_period=3)
    
    # Convert to Pandas Series for signal conditions
    close_series = pd.Series(close, index=index)
    rsi = pd.Series(rsi, index=index)
    
    # Signal conditions
    close_prev = close_series.shift(1)
    upper_prev = pd.Series(upper, index=index).shift(1)
    lower_prev = pd.Series(lower, index=index).shift(1)
    rsi_prev = rsi.shift(1)
    
    bear_entries = (close_prev > upper_prev) & (close_series < upper) & \
                   (rsi_prev > upperlimit)
    bull_entries = (close_prev < lower_prev) & (close_series > lower) & \
                   (rsi_prev < lowerlimit)
    
    return bear_entries, bull_entries

BBSR = vbt.IndicatorFactory(
    class_name="BBSR",
    short_name="BBSR",
    input_names=["close"],
    param_names=["length", "mult", "lengthRSI", "upperlimit", "lowerlimit"],
    output_names=["bear_entries", "bull_entries"]
).from_apply_func(
    BBSRDef,
    length=20,
    mult=2.0,
    lengthRSI=14,
    upperlimit=70,  # Changed to typical RSI overbought level
    lowerlimit=30,  # Changed to typical RSI oversold level
    to_2d=False
)

class BBSRStrategy(BaseStrategy):
    '''Bollinger Bands Stochastic RSI Extreme Strategy'''
    _name = "BBSR"
    desc = "BBSR is a strategy combining Bollinger Bands and Stochastic RSI to identify extreme price conditions. Generates bearish signals when price crosses below upper BB with overbought Stochastic RSI, and bullish signals when price crosses above lower BB with oversold Stochastic RSI."
    
    param_def = [
        {"name": "length", "type": "int", "min": 10, "max": 20, "step": 5},  # all int
        {"name": "mult", "type": "float", "min": 1.0, "max": 2.0, "step": 0.5},  # all float
        # {"name": "smoothK", "type": "int", "min": 1, "max": 3, "step": 1},  # all int
        # {"name": "smoothD", "type": "int", "min": 1, "max": 3, "step": 1},  # all int
        {"name": "lengthRSI", "type": "int", "min": 5, "max": 14, "step": 3},  # all int
        # {"name": "lengthStoch", "type": "int", "min": 5, "max": 14, "step": 3},  # all int
        {"name": "upperlimit", "type": "float", "min": 70.0, "max": 90.0, "step": 10.0},  # all float
        {"name": "lowerlimit", "type": "float", "min": 10.0, "max": 30.0, "step": 10.0},  # all float
    ]

    @vbt.cached_method
    def run(self, calledby='add') -> bool:
        # Initialize variables
        close_price = self.stock_dfs[0][1].close
        open_price = self.stock_dfs[0][1].open
        
        # Get parameters
        params = {param['name']: self.param_dict[param['name']] 
                 for param in self.param_def}

        # Calculate indicators
        ind = BBSR.run(close_price, **params, param_product=True)

        # Clean up param_dict
        for param in self.param_def:
            del self.param_dict[param['name']]

        # Generate signals with no look-ahead bias
        bear_entries = ind.bear_entries.vbt.signals.fshift()
        bull_entries = ind.bull_entries.vbt.signals.fshift()
        
        # For this strategy, we'll use bear signals as exits for long positions
        # and bull signals as entries
        entries = bull_entries
        exits = bear_entries

        # Build portfolio
        if self.param_dict['WFO'] != 'None':
            entries, exits = self.maxRARM_WFO(close_price, entries, exits, calledby)
            pf = vbt.Portfolio.from_signals(
                close=close_price, 
                open=open_price, 
                entries=entries, 
                exits=exits, 
                **self.pf_kwargs
            )
        else:
            pf = vbt.Portfolio.from_signals(
                close=close_price, 
                open=open_price, 
                entries=entries, 
                exits=exits, 
                **self.pf_kwargs
            )
            if calledby == 'add':
                RARMs = eval(f"pf.{self.param_dict['RARM']}()")
                idxmax = RARMs[RARMs != np.inf].idxmax()
                if self.output_bool:
                    plot_CSCV(pf, idxmax, self.param_dict['RARM'])
                pf = pf[idxmax]
                
                self.param_dict.update(dict(zip(
                    [p['name'] for p in self.param_def],
                    [float(x) if p['type'] == 'float' else int(x) 
                     for x, p in zip(idxmax, self.param_def)]
                )))
        
        self.pf = pf
        return True
