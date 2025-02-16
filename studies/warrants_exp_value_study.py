import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import utils.algotrade as at

from studies.stock_news_momentum_study import calculate_price_changes
from studies.stock_overnight_study import plot_scatter_2_sources
from utils.plot_utils import plot_cash_and_assets, plot_multi_bar, plot_multi_line, plot_single_bar, plot_single_line, plot_multi_scatter, plot_events
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

def monte_carlo_simulation(df, num_simulations, num_days, lookback_days=252, exercise_price=0):
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
    
    simulation_results_last = simulation_results[:, -1, :]
    
    simulation_results_last_first_stock = simulation_results_last[:, 0]
            
    
    sim_min = np.min(simulation_results_last_first_stock)
    sim_max = np.max(simulation_results_last_first_stock)
    sim_mean = np.mean(simulation_results_last_first_stock)
    sim_median = np.median(simulation_results_last_first_stock)
    sim_std = np.std(simulation_results_last_first_stock)
    sim_std_high = sim_mean + sim_std
    sim_std_low = sim_mean - sim_std
    total_sims = num_simulations * df.shape[1]
    win_rate = np.sum(simulation_results_last_first_stock > exercise_price) / total_sims
    loss_mean = np.mean(simulation_results_last_first_stock[simulation_results_last_first_stock < exercise_price])

    result_df = pd.DataFrame({
        'min': sim_min.flatten(),
        'max': sim_max.flatten(),
        'mean': sim_mean.flatten(),
        'loss_mean': loss_mean,
        'median': sim_median.flatten(),
        'std_high': sim_std_high.flatten(),
        'std_low': sim_std_low.flatten(),
        'win_rate': win_rate
    })

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

def process_simulations_results(df, all_simulations_df):
    # merge sim_df with df
    result_df = pd.merge(df, all_simulations_df, left_index=True, right_index=True, suffixes=["", "_sim"])
    
    # expected price = mean
    result_df['expected_price'] = result_df['mean']
    
    # expected_profit = expected_price - exercise_price
    result_df['expected_profit'] = result_df['expected_price'] - result_df['Exercise_Price']
    
    # expect_warrant_profit = expected_profit / Exercise_Ratio
    result_df['expect_warrant_profit'] = result_df['expected_profit'] / result_df['Exercise_Ratio']
    
    # cw_expected_return
    result_df['cw_expected_return'] = result_df['expect_warrant_profit'] / result_df['close_cw']
    
    # cw_expected_return_daily
    result_df['cw_expected_return_daily'] = result_df['cw_expected_return'] / result_df['days_to_expired']
    
    # expected_loss =  exercise_price - loss_mean
    result_df['expected_loss'] = result_df['Exercise_Price'] - result_df['loss_mean']
    
    # cw_expected_loss = expected_loss / Exercise_Ratio
    result_df['cw_expected_loss'] = result_df['expected_loss'] / result_df['Exercise_Ratio']
    
    # cw_expected_loss_return = cw_expected_loss / close_cw
    result_df['cw_expected_loss_return'] = result_df['cw_expected_loss'] / result_df['close_cw']
    
    # cw_expected_loss_return_daily
    result_df['cw_expected_loss_return_daily'] = result_df['cw_expected_loss_return'] / result_df['days_to_expired']
    
    # expect_value = cw_expected_return * win_rate + cw_expected_loss_return * (1 - win_rate)
    result_df['expect_value'] = result_df['cw_expected_return'] * result_df['win_rate'] - result_df['cw_expected_loss_return'] * (1 - result_df['win_rate'])
    
    # expect_value_daily
    result_df['expect_value_daily'] = result_df['expect_value'] / result_df['days_to_expired'] * 100
                                                                 
    return result_df

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
        sim_df = monte_carlo_simulation(stock_df_copy,30000, days_to_expired, lookback_days=252, exercise_price=df['Exercise_Price'][i])
        
        last_sim = sim_df.iloc[-1]
            
        last_sim_values = pd.DataFrame(last_sim).T
        last_sim_values.index = [sim_date]    
          
        
        # st.write(last_sim_values)
                
        all_simulations_df = pd.concat([all_simulations_df, last_sim_values])
    
    result_df = process_simulations_results(df, all_simulations_df)

    
    plot_single_line(result_df['close'], title="Stocks Close Price")
    # plot_single_line(result_df['expected_price'], title="Warant Expected Price")
    plot_single_line(result_df['close_cw'], title="Warrant Close Price")
    
    st.write("Days to Expire: ", result_df['days_to_expired'].iloc[-1])
    st.write("Expected profit: ", result_df['cw_expected_return'].iloc[-1])
    st.write("Profit probability: ", result_df['win_rate'].iloc[-1])
    st.write("Expected loss: ", result_df['cw_expected_loss_return'].iloc[-1])
    st.write("Expected value: ", result_df['expect_value'].iloc[-1])
    plot_single_bar(result_df['expect_value'], title="Expected Value")
    
    
    # st.write("Expected profit daily: ", result_df['cw_expected_return_daily'].iloc[-1])
    # st.write("Expected loss daily: ", result_df['cw_expected_loss_return_daily'].iloc[-1])
    # st.write("Expected value daily: ", result_df['expect_value_daily'].iloc[-1])
    # plot_single_bar(result_df['expect_value_daily'], title="Expected Value Daily")
    
   # ============ Backtesting ============
   
    def dynamic_position_sizing(portfolio_ratio, win_prob, expected_profit, expected_loss, unrealized_pnl, max_risk=0, fractional_kelly=0.5):
        
        expected_value = win_prob * expected_profit - (1 - win_prob) * expected_loss
        
        if expected_value < 2:
            # exit if expected value is less than 2
            return 0
        
        # Compute Kelly Fraction
        kelly_fraction = (win_prob * expected_profit - (1 - win_prob) * expected_loss) / (expected_profit * expected_loss)
        # Apply fractional Kelly to reduce risk
        kelly_fraction = max(0, min(fractional_kelly * kelly_fraction, 1))  # Keep it between 0-1
        
        pnl_adjustment_factor = 1
        if unrealized_pnl != 0:     
            # Position Size Adjustment (aware of expected vs unrealized P&L)
            pnl_adjustment_factor = (expected_profit - abs(unrealized_pnl)) / (expected_loss + abs(unrealized_pnl))
            pnl_adjustment_factor = max(0.5, min(pnl_adjustment_factor, 1.5))  # Keep adjustments reasonable
                    
        drawdown_factor = 1  # Default to no drawdown factor
        if max_risk > 0:
            # Adjust Position Size based on Max Risk and unrealized P&L
            drawdown_factor = max(0, 1 - unrealized_pnl / max_risk)  # Drawdown Factor
            drawdown_factor = max(0.5, min(drawdown_factor, 1.5))
        
        # Compute New Position Size
        new_position =  portfolio_ratio * kelly_fraction * pnl_adjustment_factor * drawdown_factor
        
        # Ensure Position Size is within 0-100% of Portfolio
        new_position = max(0, min(new_position, portfolio_ratio))
        
        
        return new_position

    # Example Usage
    current_cash = 10_000_000
    portfolio_value = current_cash
    current_value = 0
    current_position = 0
    current_asset = 0
    volume = 0
    current_pnl = 0
    current_return = 1
    min_order_size = 100
    max_risk = 0

    trade_df = pd.DataFrame()
    
    for i in range(len(result_df)):
        date = result_df.index[i]
        win_prob = result_df['win_rate'][i]
        expected_profit = result_df['cw_expected_return'][i]
        expected_loss = result_df['cw_expected_loss_return'][i]
        cw_price = result_df['close_cw'][i]
        new_pnl = 0
        new_return = 0
        
        if i > 0:
            new_value = volume * cw_price
            new_pnl = round(new_value - current_value)
            current_pnl += new_pnl
            new_return = new_pnl / current_value if current_value > 0 else 0
            current_return += new_return
        
        new_position = dynamic_position_sizing(1, win_prob, expected_profit, expected_loss, current_return, max_risk)       
        new_volume = new_position * portfolio_value / cw_price 
        # round to the nearest 100
        new_volume = round(new_volume / min_order_size) * min_order_size
        
        # update cash
        if new_volume > volume:
            # buy
            current_cash -= (new_volume - volume) * cw_price
        elif new_volume < volume:
            # sell
            current_cash += (volume - new_volume) * cw_price
        
        volume = new_volume
        current_value = volume * cw_price
        current_position = new_position
        current_asset = volume * cw_price
        portfolio_value = current_cash + current_asset
        
        new_trade = pd.DataFrame({
            'Date': [date],
            'Position': [current_position],
            'Close': [cw_price],
            'DailyPnL': [new_pnl],
            'PnL': [current_pnl],
            'DailyReturn': [new_return],
            'Return': [current_return],
            'Cash': [current_cash],
            'Asset': [current_asset],
            'Volume': [volume]
        })
        
        trade_df = pd.concat([trade_df, new_trade])
        
    # set index to tradingDate
    trade_df.set_index('Date', inplace=True)
        
    # plot return
    plot_single_line(trade_df['Return'], title="Return")

    # plot volume
    plot_single_bar(trade_df['Volume'], title="Volume")

    # plot cash & asset
    plot_cash_and_assets(trade_df, 'Cash', 'Asset')
    
    show_data = st.checkbox('Show Data', value=False)
    
    if show_data:
        st.dataframe(trade_df, use_container_width=True)
    