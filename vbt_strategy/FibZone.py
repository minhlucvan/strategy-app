import numpy as np
import pandas as pd
import vectorbt as vbt
from .base import BaseStrategy

def FibZoneDef(close, high, low, per=21):
    # Handle inputs as either Pandas Series or NumPy arrays
    if isinstance(close, pd.Series):
        index = close.index
        close = close.values
        high = high.values
        low = low.values
    else:
        close = np.asarray(close)
        high = np.asarray(high)
        low = np.asarray(low)
        index = pd.RangeIndex(len(close))
    
    # Calculate rolling highest high and lowest low
    hl = vbt.nb.rolling_max_nb(high, per)
    ll = vbt.nb.rolling_min_nb(low, per)
    dist = hl - ll  # Range of the channel
    
    # Calculate Fibonacci levels
    hf = hl - dist * 0.236    # Highest Fibonacci line (23.6%)
    cfh = hl - dist * 0.382   # Center High Fibonacci line (38.2%)
    cfl = hl - dist * 0.618   # Center Low Fibonacci line (61.8%)
    lf = hl - dist * 0.764    # Lowest Fibonacci line (76.4%)
    
    # Convert to Pandas Series for signal conditions
    close_series = pd.Series(close, index=index)
    
    # Generate signals based on price position relative to zones
    # Bullish entry: price crosses above lowest fib line from below
    # Bearish entry: price crosses below highest fib line from above
    close_prev = close_series.shift(1)
    hf_series = pd.Series(hf, index=index)
    lf_series = pd.Series(lf, index=index)
    
    bull_entries = (close_prev < lf_series) & (close_series > lf_series)
    bear_entries = (close_prev > hf_series) & (close_series < hf_series)
    
    return bull_entries, bear_entries, hl, ll, hf, cfh, cfl, lf

FibZone = vbt.IndicatorFactory(
    class_name="FibZone",
    short_name="FZ",
    input_names=["close", "high", "low"],
    param_names=["per"],
    output_names=["bull_entries", "bear_entries", "hl", "ll", "hf", "cfh", "cfl", "lf"]
).from_apply_func(
    FibZoneDef,
    per=21,
    to_2d=False
)

class FibZoneStrategy(BaseStrategy):
    '''Fibonacci Zone Strategy'''
    _name = "FibZone"
    desc = "FibZone is a strategy that uses Fibonacci retracement levels to identify potential trend reversals. Generates bullish signals when price crosses above the 76.4% Fibonacci level and bearish signals when price crosses below the 23.6% Fibonacci level."
    
    param_def = [
        {"name": "per", "type": "int", "min": 10, "max": 30, "step": 5},  # Period for lookback
    ]

    @vbt.cached_method
    def run(self, calledby='add') -> bool:
        # Initialize variables
        close_price = self.stock_dfs[0][1].close
        high_price = self.stock_dfs[0][1].high
        low_price = self.stock_dfs[0][1].low
        open_price = self.stock_dfs[0][1].open
        
        # Get parameters
        params = {param['name']: self.param_dict[param['name']] 
                 for param in self.param_def}

        # Calculate indicators
        ind = FibZone.run(close_price, high_price, low_price, **params, param_product=True)

        # Clean up param_dict
        for param in self.param_def:
            del self.param_dict[param['name']]

        # Generate signals with no look-ahead bias
        bull_entries = ind.bull_entries.vbt.signals.fshift()
        bear_entries = ind.bear_entries.vbt.signals.fshift()
        
        # Strategy logic: go long on bull entries, exit on bear entries
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
                    [int(x) for x in idxmax]  # All params are int in this case
                )))
        
        self.pf = pf
        return True