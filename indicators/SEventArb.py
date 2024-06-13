from numba import njit
import numpy as np
import vectorbt as vbt
import pandas as pd

import streamlit as st

@njit
def apply_SEventArb_nb(close, days_to_event, signals_strength, days_before_threshold, days_after_threshold, signal_threshold):
    sizes = np.full_like(close, 0.0, dtype=np.float_)

    for i in range(close.shape[0]):
        total_pos = 0.0
        
        for j in range(close.shape[1]):
            day_to_event = days_to_event[i, j]
            signal_strength = signals_strength[i, j]
            if day_to_event >= -days_before_threshold and day_to_event <= days_after_threshold:
                if day_to_event == -days_before_threshold and signal_strength < signal_threshold:
                    continue
                
                total_pos += 1
                sizes[i, j] = 1.0
                
        ## fill all sizes with the minimum size
        if total_pos > 0:
            for j in range(close.shape[1]):
                if sizes[i, j] == 1.0:
                    sizes[i, j] = 1.0 / total_pos

    return sizes

def get_SEventArbInd():
    EventArb = vbt.IndicatorFactory(
        class_name="SEventArb",
        input_names=["close", "days_to_event", "signal_strength"],
        param_names=["days_before_threshold", "days_after_threshold", "signal_threshold"],
        output_names=["sizes"]
    ).from_apply_func(apply_SEventArb_nb)

    return EventArb

