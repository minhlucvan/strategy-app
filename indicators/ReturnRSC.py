import streamlit as st
import vectorbt as vbt
import numpy as np
from numba import njit

import numpy as np
from numba import njit

import numpy as np
from numba import njit

@njit
def apply_rs_nb(close, comparative_close):
    """
    Calculate the Relative Return Strength (RRS) between two sets of closing prices.

    Parameters:
    close (ndarray): Closing prices of the target asset.
    comparative_close (ndarray): Closing prices of the comparative asset.

    Returns:
    ndarray: RRS values.
    """
    # Ensure the input arrays are 2D and have the same shape
    if close.shape != comparative_close.shape:
        raise ValueError("Input arrays must have the same shape")
    
    # Calculate returns manually since np.diff with axis is not supported in njit
    close_return = (close[1:] - close[:-1]) / close[:-1]
    comparative_close_return = (comparative_close[1:] - comparative_close[:-1]) / comparative_close[:-1]
    
    # fill comparative_close_return with 0 where comparative_close_return < 0.001
    comparative_close_return = np.where(comparative_close_return < 0.001, 0, comparative_close_return)
    
    # Ensure the returns have the same shape
    if close_return.shape != comparative_close_return.shape:
        raise ValueError("Return arrays must have the same shape")
    
    # Calculate absolute returns and signs
    close_return_abs = np.abs(close_return)
    comparative_close_return_abs = np.abs(comparative_close_return)
    close_return_sign = np.sign(close_return)
    
    # Compute relative return strength
    rrs = (close_return_abs / comparative_close_return_abs) * close_return_sign
    
    # Pad the first row with NaN to match the shape of the input arrays
    rrs_padded = np.empty_like(close)
    rrs_padded[0, :] = np.nan
    rrs_padded[1:, :] = rrs
    
    return rrs_padded

# Rolling Relative Strength Comparison Indicator
def get_ReturnRSCInd():
    """
    Returns the Rolling Relative Strength Comparison Indicator.

    This indicator compares the relative strength of a target asset against a comparative asset over time.
    It helps identify the performance of one asset relative to another.

    Output:
    - rrs: Relative Return Strength values indicating the relative performance.
    """
    ReturnRSCInd = vbt.IndicatorFactory(
        class_name='RS',
        input_names=['close', 'comparative_close'],
        param_names=[],
        output_names=['rrs']
    ).from_apply_func(apply_rs_nb)

    return ReturnRSCInd

# Streamlit app to display the indicator
def main():
    st.title("Rolling Relative Strength Comparison Indicator")

    # Simulated example data
    np.random.seed(42)
    dates = np.arange('2020-01-01', '2020-12-31', dtype='datetime64[D]')
    close = np.random.rand(len(dates), 1) * 100
    comparative_close = np.random.rand(len(dates), 1) * 100

    # Create the indicator instance
    RSCInd = get_ReturnRSCInd()

    # Run the indicator
    rrs = RSCInd.run(close, comparative_close).rrs

    # Display the results
    st.line_chart(rrs)

if __name__ == "__main__":
    main()
