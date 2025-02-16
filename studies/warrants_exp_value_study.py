import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import utils.algotrade as at

from studies.stock_news_momentum_study import calculate_price_changes
from studies.stock_overnight_study import plot_scatter_2_sources
from utils.plot_utils import plot_double_side_bars, plot_multi_bar, plot_multi_line, plot_single_bar, plot_single_line, plot_multi_scatter, plot_events
from utils.processing import get_stocks, get_stocks_foregin_flow
import utils.stock_utils as su

import plotly.graph_objects as go
import streamlit as st
import numpy as np
from studies.stock_gaps_recover_study import run as stock_gaps_recover_study

import numpy as np
import pandas as pd

import numpy as np
import pandas as pd

def monte_carlo_simulation(df, num_simulations, num_days, lookback_days=252):
    # Compute daily returns for each stock
    last_year_df = df.iloc[-lookback_days:]
    returns = last_year_df.pct_change().dropna()
    
    # Compute statistics for each stock
    returns_mean = returns.mean().values
    returns_std = returns.std().values

    # Get the last close prices for each stock
    last_close = df.iloc[-1].values

    # Initialize the simulation results array for each stock
    simulation_results = np.zeros((num_simulations, num_days, df.shape[1]))

    # Generate random returns for all simulations, days, and stocks at once
    random_returns = np.random.normal(returns_mean, returns_std, (num_simulations, num_days, df.shape[1]))
    
    # return bound +-0.07
    random_returns = np.clip(random_returns, -0.07, 0.07)

    # Calculate the simulated price paths for each stock
    simulation_results[:, 0, :] = last_close
    for t in range(1, num_days):
        simulation_results[:, t, :] = simulation_results[:, t-1, :] * (1 + random_returns[:, t, :])
        
    # Calculate the median price path for all simulations
    sim_median = np.median(simulation_results, axis=0)
    
    sim_min = np.min(simulation_results, axis=0)
    sim_max = np.max(simulation_results, axis=0)
    sim_mean = np.mean(simulation_results, axis=0)
    sim_median = np.median(simulation_results, axis=0)
    sim_std = np.std(simulation_results, axis=0)
    sim_std_high = sim_mean + sim_std
    sim_std_low = sim_mean - sim_std

    result_df = pd.DataFrame({
        'min': sim_min.flatten(),
        'max': sim_max.flatten(),
        'mean': sim_mean.flatten(),
        'median': sim_median.flatten(),
        'std_high': sim_std_high.flatten(),
        'std_low': sim_std_low.flatten()
    })

    result_df.index = pd.date_range(start=last_year_df.index[-1], periods=num_days)

    return result_df

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
    
    warrants_df = warrants_df[warrants_df['days_to_expire'] > 30]
    
    tickers = warrants_df['cw'].unique()
    
    warrants_intraday_df = su.get_last_trading_history(tickers=tickers)
    
    return warrants_df, warrants_intraday_df

def fetch_stock_data(ticker='ACB', stock_type='stock', timeframe='D', count_back=300):
    data_df = su.get_stock_bars_very_long_term_cached(
        ticker=ticker, stock_type=stock_type, count_back=count_back, resolution=timeframe)
    data_df = data_df.reset_index()
    data_df['Ticker'] = ticker
    data_df['Datetime'] = data_df['tradingDate']
    data_df = su.load_price_data_into_yf_dataframe(data_df, set_index=False)
    
    # set index to tradingDate
    data_df.set_index('Datetime', inplace=True)
    
    # keep only the close price
    data_df = data_df[['close']]

    return data_df

def run(symbol_benchmark, symbolsDate_dict):
    
    # copy the symbolsDate_dict
    # benchmark_dict = symbolsDate_dict.copy()
    warrants_df, warrants_intraday_df = fetch_data()
    
    stock_tickers = warrants_df['underlyingStock'].unique()
    fpt_index = stock_tickers.tolist().index('FPT')
    selected_stock = st.selectbox('Select Stocks', stock_tickers, index=fpt_index)
    
    warrants_df = warrants_df[warrants_df['underlyingStock'] == selected_stock]
    warrants_intraday_df = warrants_intraday_df[warrants_intraday_df.index.get_level_values(0).isin(warrants_df['cw'])]

    warrants_intraday_df['value'] = warrants_intraday_df['volume'] * warrants_intraday_df['close']
    
    # value_filter = st.slider('Value Filter', min_value=0, max_value=1_000_000_000, value=100_000_000, step=1_000, format="%d")
    
    # warrants_intraday_value_df = warrants_intraday_df[warrants_intraday_df['value'] > value_filter]
    
    tickers = warrants_intraday_df.index.get_level_values(0).unique().values.tolist()
            
    st.write(f"Number of Warrants: {len(tickers)}")
        
    cw_ticker = st.selectbox('Select Tickers', tickers, index=0, format_func=lambda x: x)
                
    warrants_selected_df = warrants_df[warrants_df['cw'] == cw_ticker]
    
    stock_df = fetch_stock_data(ticker=selected_stock, stock_type='stock', timeframe='D', count_back=2929)
    
    # get the stock info
    cw_info_data = su.get_stock_info_data(tickers=[cw_ticker])

    cw_info_df = su.load_cw_info_to_dataframe(cw_info_data)

    cw_info_df['cw'] = cw_info_df['Stock_Symbol']

    cw_df = pd.merge(cw_info_df, warrants_df, on='cw')
    
    # keep only the columns we need Closing_Price, Exercise_Price, Exercise_Ratio, expiredDate, issuedDate
    cw_df = cw_df[['cw', 'Exercise_Price', 'Exercise_Ratio', 'expiredDate', 'issuedDate', 'listedDate']]
        
    cw_price_df = fetch_stock_data(ticker=cw_ticker, stock_type='coveredWarr', timeframe='D', count_back=2929)
    
    # fake the first row of cw_df index to be the same as cw_price_df 0
    cw_df['Datetime'] = cw_price_df.index[0]
    cw_df = cw_df.set_index('Datetime')
    # fix index of cw_df to be the same as cw_price_df
    cw_df = cw_df.reindex(cw_price_df.index)
    # fill the missing values with the last value
    cw_df = cw_df.fillna(method='ffill')
    
    # caculate warrant price
    cw_df['date'] = cw_df.index
    cw_df['listedDate'] = pd.to_datetime(cw_df['listedDate'])
    # # calculate days to listedDate = Date - listedDate
    cw_df['days_to_listed'] = (cw_df['date'].dt.date - cw_df['listedDate'].dt.date)
    cw_df['days_to_listed'] = cw_df['days_to_listed'].apply(lambda x: x.days)
    # # keep only rows with days_to_listed >= 0
    cw_df = cw_df[cw_df['days_to_listed'] >= 0]
    # calculate days to expiredDate = expiredDate - Date
    cw_df['days_to_expired'] = (cw_df['expiredDate'].dt.date - cw_df['date'].dt.date)
    cw_df['days_to_expired'] = cw_df['days_to_expired'].apply(lambda x: x.days)
            
    # merge cw_price_df with stock_df
    df = pd.merge(stock_df, cw_price_df, on='Datetime', suffixes=["", "_cw"] )
    
    # merge with df with cw_df
    df = pd.merge(df, cw_df, on='Datetime')
        
    # ============ Plotting ============
    
    # first_date = closes_df.index[0]
    
    # stocks_df = stocks_df[stocks_df.index >= first_date]
        
    all_simulations_df = pd.DataFrame()
    

    # Loop through index of df
    for i in range(len(df)):
        sim_date = df.index[i]
        stock_df_copy = stock_df[stock_df.index <= sim_date].copy()
        
        days_to_expired = df['days_to_expired'][i]
        # Perform the Monte Carlo simulation for a fixed period of 100 days
        sim_df = monte_carlo_simulation(stock_df_copy, 10000, days_to_expired, lookback_days=252)
        
        last_sim = sim_df.iloc[-1]
        
        last_sim_values = pd.DataFrame(last_sim).T
        last_sim_values.index = [sim_date]
        
        # st.write(last_sim_values)
                
        all_simulations_df = pd.concat([all_simulations_df, last_sim_values])
        
    
    # plot distribution of the last day
    st.write("Distribution of the last day")
    # histogram
    fig = px.histogram(sim_df, x='mean', title='Distribution of the last day')
    st.plotly_chart(fig)

    # merge sim_df with df
    result_df = pd.merge(df, all_simulations_df, left_index=True, right_index=True, suffixes=["", "_sim"])
    
    # caculate cw_value = cw_close * cw_Exercise_Ratio
    result_df['cw_value'] = result_df['close_cw'] * result_df['Exercise_Ratio']
    
    # expected price = mean
    result_df['expected_price'] = result_df['std_low']
    
    # expected_profit = expected_price - exercise_price
    result_df['expected_profit'] = result_df['expected_price'] - result_df['Exercise_Price']
    
    # expect_warrant_profit = expected_profit / Exercise_Ratio
    result_df['expect_warrant_profit'] = result_df['expected_profit'] / result_df['Exercise_Ratio']
    
    # expected_warrant_return
    result_df['expected_warrant_return'] = result_df['expect_warrant_profit'] / result_df['close_cw']
    
    # expected_warrant_return_daily
    result_df['expected_warrant_return_daily'] = result_df['expected_warrant_return'] / result_df['days_to_expired'] * 100
    
    
    plot_single_line(result_df['close'], title="Stocks Close Price")
    plot_single_line(result_df['expected_price'], title="Warant Expected Price")
    plot_single_line(result_df['close_cw'], title="Warrant Close Price")
    

    st.write("Warrants Expected Value Study")
    st.write("Days to Expire: ", result_df['days_to_expired'].iloc[-1])
    st.write("Expected Warrant Return: ", result_df['expected_warrant_return'].iloc[-1])
    plot_single_bar(result_df['expected_warrant_return_daily'], title="Expected Daily Return(%)")
    
    