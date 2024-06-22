import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import utils.algotrade as at

from studies.stock_news_momentum_study import calculate_price_changes
from studies.stock_overnight_study import plot_scatter_2_sources
from utils.plot_utils import plot_double_side_bars, plot_multi_bar, plot_multi_line, plot_single_bar, plot_single_line, plot_multi_scatter, plot_events
from utils.processing import get_stocks, get_stocks_foregin_flow, get_stocks_info
import utils.stock_utils as su

import plotly.graph_objects as go
import streamlit as st
import numpy as np
from studies.stock_gaps_recover_study import run as stock_gaps_recover_study

import numpy as np
import pandas as pd

import numpy as np
import pandas as pd

import pandas as pd
from optionlab import run_strategy

def calculate_volatility(stock_prices):
    log_returns = np.log(stock_prices / stock_prices.shift(1))
    volatility = log_returns.rolling(window=252).std() * np.sqrt(252)
    return volatility

@st.cache_data
def calculate_historical_pop(closes_df, stocks_mapped_df, exercise_price_df, maturity_date_df, exercise_ratios_df, selected_tickers):
    pop_df = pd.DataFrame(index=closes_df.index, columns=closes_df.columns)
    
    for selected_ticker in selected_tickers:
        stock_mapped_df = stocks_mapped_df[selected_ticker]
        close_df = closes_df[selected_ticker]
        
        for idx in closes_df.index:
            stock_mapped_recent_df = stock_mapped_df.loc[:idx].iloc[-252:]
            
            exercise_price = exercise_price_df[selected_ticker].loc[idx]
            maturity_date = maturity_date_df[selected_ticker].loc[idx]
            
            maturity_date_str = maturity_date.strftime("%Y-%m-%d")
            today_date_str = idx.strftime("%Y-%m-%d")
            
            last_stock_price = stock_mapped_df.loc[idx]
            last_close_price = close_df.loc[idx]
            
            exercise_ratio = exercise_ratios_df[selected_ticker].loc[idx]
            
            premium_price = exercise_ratio * last_close_price
            
            volatilities = calculate_volatility(stock_mapped_df.loc[:idx])
            volatility = volatilities.loc[idx]
            days_to_maturity = (maturity_date - idx).days
            
            min_stock = stock_mapped_recent_df.min()
            max_stock = stock_mapped_recent_df.max()
            
            inputs_data = {
                "stock_price": last_stock_price,
                "start_date": today_date_str,
                "target_date": maturity_date_str,
                "volatility": volatility,
                "interest_rate": 0.0002,
                "min_stock": min_stock,
                "max_stock": max_stock,
                "strategy": [
                    {
                        "type": "call",
                        "expiration": days_to_maturity,
                        "strike": exercise_price,
                        "premium": last_close_price,
                        "n": 100,
                        "action": "buy"
                    }
                ],
            }
            
            try:
                out = run_strategy(inputs_data)
                probability_of_profit = out.probability_of_profit
                
                pop_df.at[idx, selected_ticker] = probability_of_profit
            except Exception as e:
                print(e)
                pop_df.at[idx, selected_ticker] = np.nan
    
    return pop_df

def calculate_premium(stock_price, exercise_price, premium):
    return stock_price - exercise_price - premium

@st.cache_data
def fetch_warrants_data():
    data =  su.get_warrants_data()

    data_df = pd.DataFrame(data)

    # keep the first word of the period
    data_df['period'] = data_df['period'].str.split().str[0]

    # convert to date
    data_df['listedDate'] = pd.to_datetime(data_df['listedDate'])
    data_df['issuedDate'] = pd.to_datetime(data_df['issuedDate'])
    data_df['expiredDate'] = pd.to_datetime(data_df['expiredDate'])

    return data_df

@st.cache_data
def fetch_data():
    warrants_df = fetch_warrants_data()
    
    warrants_df['days_to_expire'] = (warrants_df['expiredDate'] - pd.Timestamp.today()).dt.days
    
    # filter out the expired warrants > 60
    
    warrants_df = warrants_df[warrants_df['days_to_expire'] > 3]
    
    tickers = warrants_df['cw'].unique()
    
    warrants_intraday_df = su.get_last_trading_history(tickers=tickers)
    
    return warrants_df, warrants_intraday_df

def run(symbol_benchmark, symbolsDate_dict):
    
    # copy the symbolsDate_dict
    # benchmark_dict = symbolsDate_dict.copy()
    warrants_df, warrants_intraday_df = fetch_data()
    
    stock_tickers = warrants_df['underlyingStock'].unique()
    
    selected_stocks = st.multiselect('Select Stocks', stock_tickers, default=stock_tickers)
    
    if len(selected_stocks) > 0:
        warrants_df = warrants_df[warrants_df['underlyingStock'].isin(selected_stocks)]
        warrants_intraday_df = warrants_intraday_df[warrants_intraday_df.index.get_level_values(0).isin(warrants_df['cw'])]

    warrants_intraday_df['value'] = warrants_intraday_df['volume'] * warrants_intraday_df['close']
    
    value_filter = st.slider('Value Filter', min_value=0, max_value=1_000_000_000, value=100_000_000, step=1_000, format="%d")
    
    warrants_intraday_value_df = warrants_intraday_df[warrants_intraday_df['value'] > value_filter]
    
    tickers = warrants_intraday_value_df.index.get_level_values(0).unique().values.tolist()
        
    st.write(f"Number of Warrants: {len(tickers)}")
    
    select_all = st.checkbox('Select All')
    
    selected_tickers = st.multiselect('Select Tickers', tickers, default=tickers if select_all else [])
    
        
    if len(selected_tickers) < 1:
        st.info("Please select symbols.")
        st.stop()
    
    symbolsDate_dict['symbols'] = selected_tickers
    
    
    closes_df = get_stocks(symbolsDate_dict, 'close', stock_type='warrant')
    
    warrants_selected_df = warrants_df[warrants_df['cw'].isin(selected_tickers)]
    
    stocks_tickers = warrants_selected_df['underlyingStock'].unique()
    
    symbolsDate_dict_copy = symbolsDate_dict.copy()
    symbolsDate_dict_copy['symbols'] = stocks_tickers
    
    stocks_df = get_stocks(symbolsDate_dict_copy, 'close')
    
    exercise_prices = get_stocks_info(symbolsDate_dict, 'Exercise_Price')
    
    maturity_date = get_stocks_info(symbolsDate_dict, 'Maturity_Date')
    
    exercise_ratios = get_stocks_info(symbolsDate_dict, 'Exercise_Ratio')
    
    exercise_ratios_df = pd.DataFrame(index=closes_df.index, columns=closes_df.columns)
    
    for stock in exercise_ratios_df.columns:
        exercise_ratios_df[stock] = exercise_ratios[stock].values[0]    
    
    maturity_date_df = pd.DataFrame(index=closes_df.index, columns=closes_df.columns)
    
    for stock in maturity_date_df.columns:
        maturity_date_df[stock] = maturity_date[stock].values[0]
    
    exercise_price_df = pd.DataFrame(index=closes_df.index, columns=closes_df.columns)
    
    for stock in exercise_price_df.columns:
        exercise_price_df[stock] = exercise_prices[stock].values[0]
    
    
    stocks_mapped_df = pd.DataFrame()
    
    for warrant in selected_tickers:
        # CABCXXX => ABC
        stock_ticker = warrant[1:4]
        stock_df = stocks_df[stock_ticker]
        # rename the column to warrant
        stock_df.name = warrant
        
        stocks_mapped_df = pd.concat([stocks_mapped_df, stock_df], axis=1)
        
    plot_multi_line(stocks_mapped_df, title="Stocks Close Price")

    plot_multi_line(closes_df, title="Warrants Close Price")    
    
    
    # Usage example:
    pop_df = calculate_historical_pop(closes_df, stocks_mapped_df, exercise_price_df, maturity_date_df, exercise_ratios_df, selected_tickers)

    plot_multi_line(pop_df, title="Historical Probability of Profit")