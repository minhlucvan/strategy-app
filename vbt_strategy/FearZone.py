import streamlit as st
import vectorbt as vbt
import numpy as np
from numba import njit
import pandas as pd

# Numba function with only input arguments
@njit
def apply_fear_greed_nb(open, close, high, low, fz1, fz1_limit, fz2, fz2_limit, gz1, gz1_limit):
    entries_long = np.full(close.shape, False)  # Fearzone (buy)
    exits_long = np.full(close.shape, False)
    entries_short = np.full(close.shape, False)  # Greedzone (sell)
    exits_short = np.full(close.shape, False)
    
    for col in range(close.shape[1]):
        for row in range(close.shape[0]):
            # Calculate True Range (ta.tr equivalent)
            if row > 0:
                tr = max(high[row, col] - low[row, col], 
                         abs(high[row, col] - close[row - 1, col]), 
                         abs(low[row, col] - close[row - 1, col]))
            else:
                tr = high[row, col] - low[row, col]
            
            # Fearzone Condition (Long Entry)
            if fz1[row, col] > fz1_limit[row, col] and fz2[row, col] < fz2_limit[row, col] and (row == 0 or not entries_long[row - 1, col]):
                entries_long[row, col] = True
                if row + 1 < close.shape[0]:
                    exits_long[row + 1, col] = True  # Exit next bar
            
            # Greedzone Condition (Short Entry)
            if gz1[row, col] > gz1_limit[row, col] and (row == 0 or not entries_short[row - 1, col]):
                entries_short[row, col] = True
                if row + 1 < close.shape[0]:
                    exits_short[row + 1, col] = True  # Exit next bar
    
    return entries_long, exits_long, entries_short, exits_short

# Indicator Factory with parameters for sweeping but not passed to Numba
def get_FearGreedInd():
    FearGreedInd = vbt.IndicatorFactory(
        class_name='FearGreed',
        input_names=['open', 'close', 'high', 'low', 'fz1', 'fz1_limit', 'fz2', 'fz2_limit', 'gz1', 'gz1_limit'],
        param_names=['high_period', 'stdev_period'],  # Used for sweeping, not passed to Numba
        output_names=['entries_long', 'exits_long', 'entries_short', 'exits_short']
    ).from_apply_func(apply_fear_greed_nb)
    return FearGreedInd

# Strategy Class
from .base import BaseStrategy
from utils.vbt import plot_CSCV
from utils.processing import get_stocks

class FearGreedStrategy(BaseStrategy):
    '''Fearzone & Greedzone Contrarian Strategy'''
    
    _name = "FearGreed"
    desc = "Contrarian strategy based on Fearzone (oversold) and Greedzone (overbought) conditions"
    stacked_bool = True
    value_filter = True
    param_def = [
        {
            "name": "high_period",
            "type": "int",
            "min": 10,
            "max": 50,
            "step": 5
        },
        {
            "name": "stdev_period",
            "type": "int",
            "min": 20,
            "max": 100,
            "step": 10
        }
    ]

    def run(self, calledby='add') -> bool:
        # 1. Initialize variables
        close_price = self.stocks_df
        open_price = get_stocks(self.symbolsDate_dict, 'open')
        high_price = get_stocks(self.symbolsDate_dict, 'high')
        low_price = get_stocks(self.symbolsDate_dict, 'low')
        
        # Handle parameter arrays/Series
        high_period = np.array(self.param_dict['high_period'])
        stdev_period = np.array(self.param_dict['stdev_period'])
        
        # Ensure input data is clean and in DataFrame format
        for df in [close_price, open_price, high_price, low_price]:
            if not isinstance(df, (pd.DataFrame, pd.Series)):
                raise ValueError(f"Input data must be a Pandas DataFrame or Series, got {type(df)}")
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.fillna(method='ffill', inplace=True)
        
        close_price = close_price if isinstance(close_price, pd.DataFrame) else close_price.to_frame()
        open_price = open_price if isinstance(open_price, pd.DataFrame) else open_price.to_frame()
        high_price = high_price if isinstance(high_price, pd.DataFrame) else high_price.to_frame()
        low_price = low_price if isinstance(low_price, pd.DataFrame) else low_price.to_frame()

        # 2. Precompute Fear and Greed metrics for each parameter combination
        source = (open_price + high_price + low_price + close_price) / 4
        
        fz1_list, fz1_limit_list, fz2_list, fz2_limit_list, gz1_list, gz1_limit_list = [], [], [], [], [], []
        
        for hp in high_period:
            for sp in stdev_period:
                # Fearzone FZ1
                highest_high = high_price.vbt.rolling_max(window=int(hp), minp=1)
                fz1 = (highest_high - source) / highest_high
                avg1 = fz1.vbt.rolling_mean(window=int(sp), minp=1)
                stdev1 = fz1.vbt.rolling_std(window=int(sp), minp=1)
                fz1_limit = avg1 + stdev1
                
                # Fearzone FZ2
                fz2 = source.vbt.rolling_mean(window=int(hp), minp=1)
                avg2 = fz2.vbt.rolling_mean(window=int(sp), minp=1)
                stdev2 = fz2.vbt.rolling_std(window=int(sp), minp=1)
                fz2_limit = avg2 - stdev2
                
                # Greedzone GZ1
                lowest_low = low_price.vbt.rolling_min(window=int(hp), minp=1)
                gz1 = (source - lowest_low) / (highest_high - lowest_low)
                avg_gz1 = gz1.vbt.rolling_mean(window=int(sp), minp=1)
                stdev_gz1 = gz1.vbt.rolling_std(window=int(sp), minp=1)
                gz1_limit = avg_gz1 + stdev_gz1
                
                # Append computed metrics
                fz1_list.append(fz1)
                fz1_limit_list.append(fz1_limit)
                fz2_list.append(fz2)
                fz2_limit_list.append(fz2_limit)
                gz1_list.append(gz1)
                gz1_limit_list.append(gz1_limit)

        # Stack the metrics for VectorBT parameter broadcasting
        fz1 = pd.concat(fz1_list, axis=1, keys=pd.MultiIndex.from_product([high_period, stdev_period], names=['high_period', 'stdev_period']))
        fz1_limit = pd.concat(fz1_limit_list, axis=1, keys=pd.MultiIndex.from_product([high_period, stdev_period]))
        fz2 = pd.concat(fz2_list, axis=1, keys=pd.MultiIndex.from_product([high_period, stdev_period]))
        fz2_limit = pd.concat(fz2_limit_list, axis=1, keys=pd.MultiIndex.from_product([high_period, stdev_period]))
        gz1 = pd.concat(gz1_list, axis=1, keys=pd.MultiIndex.from_product([high_period, stdev_period]))
        gz1_limit = pd.concat(gz1_limit_list, axis=1, keys=pd.MultiIndex.from_product([high_period, stdev_period]))

        # 3. Run the FearGreed Indicator (only pass inputs, not parameters)
        indicator = get_FearGreedInd().run(
            open_price,
            close_price,
            high_price,
            low_price,
            high_period=high_period,
            stdev_period=stdev_period,
            fz1=fz1,
            fz1_limit=fz1_limit,
            fz2=fz2,
            fz2_limit=fz2_limit,
            gz1=gz1,
            gz1_limit=gz1_limit,
            param_product=True  # Parameters are handled by IndicatorFactory
        )

        # 4. Remove parameters from param_dict
        for param in self.param_def:
            del self.param_dict[param['name']]

        # 5. Extract signals
        entries_long = indicator.entries_long
        exits_long = indicator.exits_long
        entries_short = indicator.entries_short
        exits_short = indicator.exits_short

        # 6. Group by symbols for portfolio
        num_symbol = len(self.stocks_df.columns)
        num_group = int(len(entries_long.columns) / num_symbol)
        group_list = []
        for n in range(num_group):
            group_list.extend([n] * num_symbol)
        group_by = pd.Index(group_list, name='group')

        # 7. Build portfolio
        if 'WFO' in self.param_dict and self.param_dict['WFO'] != 'None':
            entries_long, exits_long = self.maxRARM_WFO(close_price, entries_long, exits_long, calledby)
            pf_long = vbt.Portfolio.from_signals(
                open_price,
                entries=entries_long,
                exits=exits_long,
                group_by=group_by,
                **self.pf_kwargs
            )
            entries_short, exits_short = self.maxRARM_WFO(close_price, entries_short, exits_short, calledby)
            pf_short = vbt.Portfolio.from_signals(
                open_price,
                entries=entries_short,
                exits=exits_short,
                group_by=group_by,
                direction='shortonly',
                **self.pf_kwargs
            )
            pf = pf_long + pf_short
        else:
            pf_long = vbt.Portfolio.from_signals(
                open_price,
                entries=entries_long,
                exits=exits_long,
                group_by=group_by,
                **self.pf_kwargs
            )
            pf_short = vbt.Portfolio.from_signals(
                open_price,
                entries=entries_short,
                exits=exits_short,
                group_by=group_by,
                direction='shortonly',
                **self.pf_kwargs
            )
            pf = pf_long + pf_short

            if calledby == 'add':
                RARMs = eval(f"pf.{self.param_dict['RARM']}()")
                if isinstance(RARMs, pd.Series):
                    idxmax = RARMs[RARMs != np.inf].idxmax()
                    if self.output_bool:
                        plot_CSCV(pf, idxmax, self.param_dict['RARM'])
                    pf = pf[idxmax]
                    params_value = entries_long.columns[idxmax * num_symbol]
                    self.param_dict.update(dict(zip(['high_period', 'stdev_period'], 
                                                  [int(params_value[0]), int(params_value[1])])))
                else:
                    idxmax = (high_period[0], stdev_period[0])
                    self.param_dict.update(dict(zip(['high_period', 'stdev_period'], 
                                                  [int(high_period[0]), int(stdev_period[0])])))

        self.pf = pf
        return True