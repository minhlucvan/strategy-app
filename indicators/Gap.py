import streamlit as st
import vectorbt as vbt
import numpy as np
from numba import njit


@njit
def apply_gap_nb(open, close, gap_percent, exit_bars):
    gaps = np.full(close.shape, np.nan)
    entries = np.full(close.shape, False)
    exits = np.full(close.shape, False)
    
    for col in range(close.shape[1]):
        holding_days = -1
        for row in range(close.shape[0]):
            gap = (open[row, col] - close[row - 1, col]) / close[row - 1, col]          
            if row < 3:
                gap = np.nan
                  
            gaps[row, col] = gap
            

            if gap <= -gap_percent and holding_days == -1:
                entries[row, col] = True
                holding_days = 0
            
            if holding_days >= 0:
                holding_days += 1
                if holding_days >= exit_bars:
                    exits[row, col] = True
                    holding_days = -1
    
        
    return gaps, entries, exits

# Relative Strength Comparison Indicator
# the indicator is used to compare the relative strength of two stocks
# RS  = Base Security / Comparative Security
def get_GapInd():
    GapInd = vbt.IndicatorFactory(
        class_name='Gap',
        input_names=['open', 'close'], 
        param_names=['gap_percent', 'exit_bars'],
        output_names=['gaps', 'entries', 'exits']
    ).from_apply_func(apply_gap_nb)

    return GapInd

