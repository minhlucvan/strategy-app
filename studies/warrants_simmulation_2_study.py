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

    # Calculate the simulated price paths for each stock
    simulation_results[:, 0, :] = last_close
    for t in range(1, num_days):
        simulation_results[:, t, :] = simulation_results[:, t-1, :] * (1 + random_returns[:, t, :])
        
    # Calculate the median price path for all simulations
    sim_median = np.median(simulation_results, axis=0)
    
    sim_median_df = pd.DataFrame(sim_median)
    
    sim_median_df.index = pd.date_range(start=last_year_df.index[-1], periods=num_days)
    
    sim_median_df.columns = df.columns
    
    return sim_median_df

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
    
    # first_date = closes_df.index[0]
    
    # stocks_df = stocks_df[stocks_df.index >= first_date]
    
    
    stocks_mapped_df = pd.DataFrame()
    
    for warrant in selected_tickers:
        # CABCXXX => ABC
        stock_ticker = warrant[1:4]
        stock_df = stocks_df[stock_ticker]
        # rename the column to warrant
        stock_df.name = warrant
        
        stocks_mapped_df = pd.concat([stocks_mapped_df, stock_df], axis=1)
        
    plot_multi_line(stocks_mapped_df, title="Stocks Close Price")
    
    # Assuming monte_carlo_simulation and plot_multi_line are defined elsewhere

    historical_sim_df = pd.DataFrame()
    
    sim_period = st.number_input('Simulation Period', min_value=1, max_value=100, value=100, step=1)
    sim_lookback = st.number_input('Simulation Lookback', min_value=1, max_value=100, value=100, step=1)

    # Loop through the full history
    for i in range(len(stocks_mapped_df)):
        days_sim = len(stocks_mapped_df) - i
        start_date = stocks_mapped_df.index[-days_sim]
        stock_df = stocks_mapped_df[stocks_mapped_df.index <= start_date]
        
        # Perform the Monte Carlo simulation for a fixed period of 100 days
        sim_df = monte_carlo_simulation(stock_df, 100, sim_period, sim_lookback)
        
        last_sim = sim_df.iloc[-1]
        end_date = sim_df.index[-1]
        
        last_sim_values = pd.DataFrame(last_sim).T
        last_sim_values.index = [end_date]
                
        historical_sim_df = pd.concat([historical_sim_df, last_sim_values])
        
    
    # rename col to sim_
    historical_sim_cp_df = historical_sim_df.copy()
    historical_sim_df.columns = [f'sim_{col}' for col in historical_sim_df.columns]    

    # Merge the simulated data with the actual data
    historical_sim_df = pd.concat([stocks_mapped_df, historical_sim_df])

    # Plot the historical and simulated data
    plot_multi_line(historical_sim_df, title="Historical Simulation")


    # calculate accuracy
    accuracy_df = (historical_sim_cp_df - stocks_mapped_df) / stocks_mapped_df
    
    plot_multi_bar(accuracy_df, title="Accuracy")
    
    mean_accuracy = accuracy_df.mean()
    median_accuracy = accuracy_df.median()
    
    st.write("Mean Accuracy", mean_accuracy.values[0])
    st.write("Median Accuracy", median_accuracy.values[0])