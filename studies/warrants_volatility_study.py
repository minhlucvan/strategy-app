import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import utils.algotrade as at

from studies.stock_news_momentum_study import calculate_price_changes
from studies.stock_overnight_study import plot_scatter_2_sources
import utils.plot_utils as pu
from utils.processing import get_stocks, get_stocks_foregin_flow, get_stocks_info
import utils.stock_utils as su

import plotly.graph_objects as go
import streamlit as st
import numpy as np

import numpy as np
import pandas as pd

import numpy as np
import pandas as pd

from .warrants_history_study import load_full_warrants_history, reload_warrant_news, reload_warrant_price_history

import numpy as np
from scipy.stats import norm

import numpy as np
from scipy.stats import norm

import numpy as np
from scipy.stats import norm


import numpy as np
from scipy.stats import norm

def calculate_black_scholes(S, K, T, r, ratio, sigma, option_type='put'):
    """
    Calculate the Black-Scholes option pricing model with a conversion ratio.
    
    Parameters:
    S (float): Current stock price
    K (float): Strike price of the option
    T (float): Time to expiration in years
    r (float): Risk-free interest rate (annualized)
    ratio (float): Conversion ratio (e.g., 10 options = 1 stock, ratio = 10)
    sigma (float): Volatility of the underlying asset (annualized)
    option_type (str): Type of option ('call' or 'put')
    
    Returns:
    float: Option price per option contract
    """
    
    # Ensure valid input for option type
    if option_type not in ['call', 'put']:
        raise ValueError("option_type must be either 'call' or 'put'")
    
    # Adjust stock price and strike price based on the conversion ratio
    adjusted_S = S / ratio
    adjusted_K = K / ratio
    
    # Calculate d1 and d2 according to the Black-Scholes formula
    d1 = (np.log(adjusted_S / adjusted_K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    # Compute the option price based on type
    if option_type == 'call':
        option_price = adjusted_S * norm.cdf(d1) - adjusted_K * np.exp(-r * T) * norm.cdf(d2)
    else:  # 'put'
        option_price = adjusted_K * np.exp(-r * T) * norm.cdf(-d2) - adjusted_S * norm.cdf(-d1)
        
    return option_price

def calculate_implied_volatility(S, K, T, r, market_price, ratio, option_type='put', tol=1e-6, max_iter=10000):
    """
    Calculate implied volatility using a binary search approach.
    
    Parameters:
    S (float): Current stock price
    K (float): Strike price of the option
    T (float): Time to expiration in years
    r (float): Risk-free interest rate (annualized)
    market_price (float): Observed closing price of the covered warrant
    ratio (float): Conversion ratio
    option_type (str): Type of option ('call' or 'put')
    tol (float): Tolerance for price difference
    max_iter (int): Maximum number of iterations
    
    Returns:
    float: Implied volatility
    """
    
    sigma_low, sigma_high = 0.01, 1.5  # Set more reasonable bounds for volatility
    for _ in range(max_iter):
        sigma_mid = (sigma_low + sigma_high) / 2
        price = calculate_black_scholes(S, K, T, r, ratio, sigma_mid, option_type)
        
        if abs(price - market_price) < tol:
            return sigma_mid
        elif price < market_price:
            sigma_low = sigma_mid
        else:
            sigma_high = sigma_mid
    
    return sigma_mid  # Return the best estimate


def calculate_volatility(
    df, 
    window=252, 
    column='close', 
    annualize=False, 
    periods_per_year=252
):
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    returns = df[column].pct_change()
    vol = returns.rolling(window=window).std()
    
    if annualize:
        vol *= np.sqrt(periods_per_year)
    
    return vol

def run(symbol_benchmark, symbolsDate_dict):
    if len(symbolsDate_dict['symbols']) < 1:
        st.info("Please select symbols.")
        st.stop()
    
    ticker = symbolsDate_dict['symbols'][0]
    
    # button to reload
    if st.button('Reload Info'):
        reload_warrant_news(ticker)
        
    if st.button('Reload price'):
        reload_warrant_price_history(ticker)
        
        
    full_df = load_full_warrants_history(ticker)
    
    stocks_df = get_stocks(symbolsDate_dict, 'close')
    
    stock_df = stocks_df[[ticker]]
    stock_df.columns = ['close']
    
    # index to datetime
    stock_df.index = pd.to_datetime(stock_df.index)

    # plot stock
    pu.plot_single_line(stock_df['close'], title=f'{ticker} price')
    
    all_tickers = full_df['ticker'].unique()
    select_all = st.checkbox('Select all', value=False)
    default_tickers = all_tickers if select_all else []
    
    selected_tickers = st.multiselect('Select tickers', all_tickers, default=default_tickers)
        
    for cw_ticker in selected_tickers:
        st.write(f'Warrant: {cw_ticker}')
        cw_df = full_df[full_df['ticker'] == cw_ticker]
        
        # index to datetime
        stock_volatility = calculate_volatility(stock_df, window=252, column='close', annualize=True)
        
        # pu.plot_single_line(stock_volatility, title=f'{ticker} volatility 1y')
        
        # pu.plot_single_line(stock_volatility, title=f'{cw_ticker} volatility 1y')
        pu.plot_single_line(cw_df['close_cw'], title=f'{cw_ticker} price')
        
        implied_volatility_df = pd.DataFrame()
        
        # caculate implied volatility history
        for row in cw_df.itertuples():
            date = row.TradingDate
            S = row.close_stock
            K = row.Exercise_Price
            T = row.days_to_expired / 365
            r = 0.045
            market_price = row.close_cw
            option_type = 'put'
            ratio  = row.Exercise_Ratio
            
            # st.write(f'Date: {date}, S: {S}, K: {K}, T: {T}, r: {r}, market_price: {market_price}, option_type: {option_type}, ratio: {ratio}')
            
            implied_vol = calculate_implied_volatility(S, K, T, r, ratio, market_price, option_type)
            implied_volatility_df = pd.concat([implied_volatility_df, pd.DataFrame({'date': [date], 'implied_vol': [implied_vol]})])
        
        st.dataframe(cw_df[['TradingDate', 'close_stock', 'Exercise_Price', 'days_to_expired', 'Exercise_Ratio', 'close_cw']])
        
        implied_volatility_df.set_index('date', inplace=True)
        
        stock_volatility_aligned_df = stock_volatility.reindex(implied_volatility_df.index).dropna()

        vol_df = pd.concat([implied_volatility_df, stock_volatility_aligned_df], axis=1)
        vol_df.columns = ['implied_vol', 'stock_vol']
        
        pu.plot_multi_line(vol_df, title='Prices', x_title='Date', y_title='Price', legend_title='Ticker')
        
        
        pu.plot_single_line(vol_df['stock_vol'], title=f'{cw_ticker} implied volatility')