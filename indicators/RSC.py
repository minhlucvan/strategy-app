import streamlit as st
import vectorbt as vbt
import numpy as np
from numba import njit


@njit
def apply_rs_nb(price, comparative_price):
    rs = np.full(price.shape, np.nan)
    comparative_price_float = comparative_price / 1.0
    
    for i in range(price.shape[0]):
        # divide the price (float)
        rs[i] = np.divide(price[i], comparative_price_float[i])
        
    return rs

# Relative Strength Comparison Indicator
# the indicator is used to compare the relative strength of two stocks
# RS  = Base Security / Comparative Security
def get_RSCInd():
    RSCInd = vbt.IndicatorFactory(
        class_name='RS',
        input_names=['price', 'comparative_price'],
        param_names=[],
        output_names=['rs']
    ).from_apply_func(apply_rs_nb)

    return RSCInd

