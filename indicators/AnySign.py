import streamlit as st
import vectorbt as vbt
import numpy as np
from numba import njit


@njit
def apply_signal_nb(pirce, signal, entry_threshold, exit_threshold):
    entry_signal = np.full(signal.shape, False)
    exit_signal = np.full(signal.shape, False)
    
    for i in range(signal.shape[0]):
        if signal[i] > entry_threshold:
            entry_signal[i] = True
        elif signal[i] < exit_threshold:
            exit_signal[i] = True
            
    return entry_signal, exit_signal


def get_AnySignInd():
    AnySignInd = vbt.IndicatorFactory(
        class_name='AnySign',
        input_names=['price', 'signal'],
        param_names=['entry_threshold', 'exit_threshold'],
        output_names=['entry_signal', 'exit_signal']
    ).from_apply_func(apply_signal_nb)

    return AnySignInd

