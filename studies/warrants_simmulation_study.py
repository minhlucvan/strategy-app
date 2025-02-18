import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from time import sleep
import plotly.express as px
import plotly.graph_objects as go
from pytz import UTC  # Import the UTC time zone
from plotly.subplots import make_subplots

from utils.stock_utils import (
    get_stock_bars_very_long_term_cached,
    get_warrants_data,
    get_stock_info_data,
    load_cw_info_to_dataframe,
    load_price_data_into_yf_dataframe,
    process_warrants_data,
    warrant_break_even_point,
)


@st.cache_data
def monte_carlo_simulation(df, num_simulations, num_days):
    df['return'] = df['close'].pct_change()
    df['retturn_mean'] = df['return'].mean()
    df['return_std'] = df['return'].std()

    returns = df['return'].dropna().values
    mean_return = df['return'].mean()
    std_return = df['return'].std()

    simulation_results = pd.DataFrame()

    for i in range(num_simulations):
        daily_returns = np.random.normal(mean_return, std_return, num_days)
        cumulative_returns = np.cumprod(1 + daily_returns) - 1
        simulation_results[f'Simulation_{i+1}'] = (
            1 + cumulative_returns) * df['close'].iloc[-1]

    return simulation_results


@st.cache_data
def fetch_warrants_data():
    data = get_warrants_data()

    data_df = pd.DataFrame(data)

    # keep the first word of the period
    data_df['period'] = data_df['period'].str.split().str[0]

    # convert to date
    data_df['listedDate'] = pd.to_datetime(data_df['listedDate'])
    data_df['issuedDate'] = pd.to_datetime(data_df['issuedDate'])
    data_df['expiredDate'] = pd.to_datetime(data_df['expiredDate'])

    return data_df

@st.cache_data
def simulate_warrants_data(tickers, data_df, stock_df, df, num_simulations=1000):
    sims_results  = pd.DataFrame()
    
    for ticker in tickers:
        ticker_stock_df = stock_df[stock_df['Ticker'] == ticker]
        stock_price = ticker_stock_df['close'].iloc[-1]

        ticker_data_df = data_df[data_df['underlyingStock'] == ticker]
        cw_tickers = ticker_data_df['cw'].unique().tolist()

        ticker_sims_results = pd.DataFrame()

        for cw_ticker in cw_tickers:
            cw_data_df = ticker_data_df[ticker_data_df['cw'] == cw_ticker]
            if cw_data_df.shape[0] == 0:
                continue

            cw_last_price = cw_data_df['Closing_Price'].values[0]
            cw_exercise_price = cw_data_df['Exercise_Price'].values[0]
            cw_conversion_ratio = cw_data_df['Exercise_Ratio'].values[0]
            cw_expiry_date = cw_data_df['expiredDate'].values[0]

            # convert to date
            cw_expiry_date = pd.to_datetime(cw_expiry_date).date()

            cw_break_even_price = warrant_break_even_point(
                current_warrant_price=cw_last_price,
                execution_price=cw_exercise_price,
                conversion_ratio=cw_conversion_ratio,
            )
            
            end_date = stock_df['date'].iloc[-1]

            cw_days_to_expiry = (cw_expiry_date - end_date).days

            results = monte_carlo_simulation(df, num_simulations, cw_days_to_expiry)

            mc_last_results = results.iloc[-1]

            mc_mean = mc_last_results.mean()

            # price gap
            price_gap = mc_mean - cw_break_even_price
            price_gap_pct = price_gap / cw_break_even_price * 100

            # possibility of price above stock price
            above_stock_price = (mc_last_results >= stock_price).sum()
            above_stock_price_pct = above_stock_price / num_simulations * 100

            # possibility of price above break even price
            above_break_even_price = (mc_last_results >= cw_break_even_price).sum()
            above_break_even_price_pct = above_break_even_price / num_simulations * 100

            # win probs
            win_probs = above_break_even_price_pct

            # expected roi
            expected_roi = price_gap_pct * cw_conversion_ratio / 100

            # expected value
            expected_value = (above_break_even_price_pct + 1) * expected_roi - 1

            # expected value daily
            expected_value_daily = expected_value / cw_days_to_expiry

            sims_result = pd.DataFrame([[ticker, cw_ticker, cw_last_price, cw_exercise_price, cw_conversion_ratio, cw_expiry_date, cw_break_even_price, cw_days_to_expiry, price_gap, price_gap_pct, above_stock_price, above_stock_price_pct, above_break_even_price, above_break_even_price_pct, win_probs, expected_roi, expected_value, expected_value_daily]])
            # set column names
            sims_result.columns = ['ticker', 'cw_ticker', 'cw_last_price', 'cw_exercise_price', 'cw_conversion_ratio', 'cw_expiry_date', 'cw_break_even_price', 'cw_days_to_expiry', 'price_gap', 'price_gap_pct', 'above_stock_price', 'above_stock_price_pct', 'above_break_even_price', 'above_break_even_price_pct', 'win_probs', 'expected_roi', 'expected_value', 'expected_value_daily']

            ticker_sims_results = pd.concat([ticker_sims_results, sims_result])

        sims_results = pd.concat([sims_results, ticker_sims_results])

    return sims_results

def fetch_data(ticker='ACB', stock_type='stock', timeframe='D', count_back=300):
    data_df = get_stock_bars_very_long_term_cached(
        ticker=ticker, stock_type=stock_type, count_back=count_back, resolution=timeframe)
    data_df = data_df.reset_index()
    data_df['Ticker'] = ticker
    data_df['Datetime'] = data_df['tradingDate']
    data_df = load_price_data_into_yf_dataframe(data_df, set_index=False)

    return data_df

@st.cache_data
def fetch_dat_multiple_tickers(tickers, stock_type='stock', timeframe='D'):
    data_df = pd.DataFrame()

    for ticker in tickers:
        df = fetch_data(ticker=ticker, stock_type=stock_type, timeframe=timeframe)
        data_df = pd.concat([data_df, df], ignore_index=True)

    return data_df

def run(symbol_benchmark, symbolsDate_dict):
        
    # Load data
    a_year_ago = datetime.now() - timedelta(days=365)
    start_date = st.date_input("Start date", a_year_ago)

    warrants_df = fetch_warrants_data()

    tickers = warrants_df['underlyingStock'].unique()

    # index of FPT
    index = tickers.tolist().index('FPT')

    selected_ticker = st.selectbox("Ticker", tickers, index=index)

    test_all_tickers = st.checkbox("Test all tickers", value=False)

    df = fetch_data(ticker=selected_ticker)

    # Extract date and hour from the timestamp
    df['date'] = df['Datetime'].dt.date
    df['Hour'] = df['Datetime'].dt.hour

    max_date = df['date'].max()

    end_date = st.date_input("End date", max_date)

    df = df[df['date'] >= start_date]
    df = df[df['date'] <= end_date]

    # plot close price
    fig = px.line(df, x='Datetime', y='close', title='Close price')
    st.plotly_chart(fig, use_container_width=True)
    #---------------------------------------------

    ticker_data_df = warrants_df[warrants_df['underlyingStock'] == selected_ticker] if not test_all_tickers else warrants_df

    cw_tickers = ticker_data_df['cw'].unique().tolist()

    ticker_warrants_tickers = ticker_data_df['cw'].unique().tolist()

    cw_info_data = get_stock_info_data(tickers=ticker_warrants_tickers)

    cw_info_df = load_cw_info_to_dataframe(cw_info_data)

    cw_info_df['cw'] = cw_info_df['Stock_Symbol']

    cw_df = pd.merge(cw_info_df, warrants_df, on='cw')

    st.write("### Warrants Test")
    test_simulations = st.number_input("Test simulations", value=1000, min_value=1, max_value=10000)
    enable_test = st.checkbox("Run test", value=False)

    test_tickers = [selected_ticker]
    test_df = df


    # filter days_to_expired
    min_days_to_expired = st.number_input("Min days to expired", value=30, min_value=0, max_value=1000)

    if test_all_tickers:
        test_tickers = tickers
        test_df = fetch_dat_multiple_tickers(tickers=tickers, stock_type='stock', timeframe='D')


    cw_tickers_results = cw_tickers
    if enable_test:
        results = simulate_warrants_data(test_tickers, cw_df, test_df, df, num_simulations=test_simulations)
        # sort by win_probs
        results = results.sort_values(by=['expected_value_daily'], ascending=False)
        result_display_df = results[['cw_ticker', 'cw_days_to_expiry', 'win_probs', 'expected_roi', 'expected_value', 'expected_value_daily']]
        
        result_display_df = result_display_df[result_display_df['cw_days_to_expiry'] >= min_days_to_expired]
        
        st.dataframe(result_display_df, use_container_width=True)

        cw_tickers_results = result_display_df['cw_ticker'].unique().tolist()

    st.write("### Warrants ticker")

    cw_ticker = st.selectbox("CW Ticker", cw_tickers_results, index=0)

    num_simulations = st.number_input(
        "Number of simulations", value=1000, min_value=1, max_value=10000)

    cw_ticker_df = cw_df[cw_df['cw'] == cw_ticker]

    show_info = st.checkbox("Show info", value=False)
    if show_info:
        st.dataframe(cw_ticker_df.T, use_container_width=True)

    stock_price = df['close'].iloc[-1]
    cw_last_price = cw_ticker_df['Closing_Price'].values[0]
    cw_exercise_price = cw_ticker_df['Exercise_Price'].values[0]
    cw_conversion_ratio = cw_ticker_df['Exercise_Ratio'].values[0]
    cw_expiry_date = cw_ticker_df['expiredDate'].values[0]
    cw_issue_date = cw_ticker_df['issuedDate'].values[0]

    # convert to date
    cw_expiry_date = pd.to_datetime(cw_expiry_date).date()

    cw_break_even_price = warrant_break_even_point(
        current_warrant_price=cw_last_price,
        execution_price=cw_exercise_price,
        conversion_ratio=cw_conversion_ratio,
    )

    cw_days_to_expiry = (cw_expiry_date - end_date).days

    results = monte_carlo_simulation(df, num_simulations, cw_days_to_expiry)

    mc_last_results = results.iloc[-1]

    mc_mean = mc_last_results.mean()
    mc_std = mc_last_results.std()
    mac_lower = mc_mean - mc_std

    # days to expiry
    # caculate price gap
    price_gap = mc_mean - cw_break_even_price
    price_gap_pct = price_gap / cw_break_even_price * 100
    stock_price_gap = mc_mean - stock_price
    stock_price_gap_pct = stock_price_gap / stock_price * 100

    # calculate lower band possibility
    lower_band = (mc_last_results <= mac_lower).sum()
    lower_band_pct = lower_band / num_simulations * 100

    # calculate free risk gap = break even price - lower band
    free_risk_gap = mac_lower - cw_break_even_price
    free_risk_gap_pct = free_risk_gap / cw_break_even_price * 100

    # caculate possibility of price above stock price
    above_stock_price = (mc_last_results >= stock_price).sum()
    above_stock_price_pct = above_stock_price / num_simulations * 100

    # caculate possibility of price above break even price
    above_break_even_price = (mc_last_results >= cw_break_even_price).sum()
    above_break_even_price_pct = above_break_even_price / num_simulations * 100

    # calculate possibility of price above mc_mean
    above_mc_mean = (mc_last_results >= mc_mean).sum()
    above_mc_mean_pct = above_mc_mean / num_simulations * 100

    # caculate stock expected roi
    stock_expected_roi = stock_price_gap_pct / 100

    # caculate stock expected annual roi
    stock_expected_annual_roi = stock_expected_roi / cw_days_to_expiry * 365

    # caculate expected roi
    expected_roi = price_gap_pct * cw_conversion_ratio / 100

    # calculate expected value
    expected_value = (above_break_even_price_pct + 1) * expected_roi - 1

    # calculate expected value daily
    expected_value_daily = expected_value / cw_days_to_expiry

    # caculate expected annual roi
    expected_annual_roi = expected_roi / cw_days_to_expiry * 365

    # caculate annual roi ratio
    annual_roi_ratio = expected_annual_roi / stock_expected_annual_roi

    cw_price_df = fetch_data(ticker=cw_ticker, stock_type='coveredWarr', timeframe='D', count_back=2929)
    cw_price_df['cw'] = cw_price_df['Ticker']

    # convert to date
    df['date'] = pd.to_datetime(df['Datetime'])

    # merge cw_price_df and cw_data_df on Date
    cw_ticker_data_df = pd.merge(df, cw_price_df, on='date', suffixes=["", "_cw"] )

    # merge cw_ticker_data_df and cw_info_df on Ticker_cw = Stock_Symbol
    cw_ticker_data_df = pd.merge(cw_ticker_data_df, cw_ticker_df, on='cw',  suffixes=["", "_info"] )

    cw_ticker_data_df['date'] = pd.to_datetime(cw_ticker_data_df['date'])

    # # convert listedDate to date
    cw_ticker_data_df['listedDate'] = pd.to_datetime(cw_ticker_data_df['listedDate'])
    # # calculate days to listedDate = Date - listedDate
    cw_ticker_data_df['days_to_listed'] = (cw_ticker_data_df['date'].dt.date - cw_ticker_data_df['listedDate'].dt.date)
    cw_ticker_data_df['days_to_listed'] = cw_ticker_data_df['days_to_listed'].apply(lambda x: x.days)
    # # keep only rows with days_to_listed >= 0
    cw_ticker_data_df = cw_ticker_data_df[cw_ticker_data_df['days_to_listed'] >= 0]
    # calculate days to expiredDate = expiredDate - Date
    cw_ticker_data_df['days_to_expired'] = (cw_ticker_data_df['expiredDate'].dt.date - cw_ticker_data_df['date'].dt.date)
    cw_ticker_data_df['days_to_expired'] = cw_ticker_data_df['days_to_expired'].apply(lambda x: x.days)

    # process warrants data
    cw_ticker_data_df = process_warrants_data(cw_ticker_data_df, risk_free_rate=0.07)
    # plot simulation results
    
    st.write("##### Days to expiry {}".format(cw_days_to_expiry))
    st.write("##### Price gap", "{:.2f}%".format(price_gap_pct))
    st.write("##### Risk free ratio", "{:.2f}%".format(free_risk_gap_pct))
    st.write("##### Win probs", "{:.2f}%".format(above_break_even_price_pct))
    st.write("##### Stock expected ROI", "{:.2f}%".format(stock_expected_roi))
    st.write("##### Stock expected annual ROI", "{:.2f}%".format(stock_expected_annual_roi))
    st.write("##### Expected ROI", "{:.2f}%".format(expected_roi))
    st.write("##### Expected value", "{:.2f}%".format(expected_value))
    st.write("##### Expected value daily", "{:.2f}%".format(expected_value_daily))
    st.write("##### Expected annual ROI", "{:.2f}%".format(expected_annual_roi))
    st.write("##### Annual ROI ratio", "{:.2f}".format(annual_roi_ratio))
    cw_price_df['date'] = pd.to_datetime(cw_price_df['Datetime'])


    show_chart = st.checkbox("Show chart", value=False)
    if show_chart:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02)
        # add trace for stock price cw_ticker_data_df['close']
        fig.add_trace(go.Scatter(x=cw_ticker_data_df['date'], y=cw_ticker_data_df['close'], name='stock_price', line=dict(color='blue')), row=1, col=1)
        # add trace for cw price cw_ticker_data_df['cw_close']
        fig.add_trace(go.Scatter(x=cw_ticker_data_df['date'], y=cw_ticker_data_df['close_cw'], name='cw_price', line=dict(color='red')), row=2, col=1)
        # add trace for break even price
        fig.add_trace(go.Scatter(x=cw_ticker_data_df['date'], y=cw_ticker_data_df['break_even_price'], name='break_even_price', line=dict(color='green')), row=1, col=1)
        
        st.plotly_chart(fig, use_container_width=True)

    visualize_df = results.copy()
    last_date = df['date'].iloc[-1]
    first_date = cw_ticker_data_df['date'].iloc[0]
    cw_ticker_break_even_price = cw_ticker_data_df['break_even_price'].iloc[-1]
    # date = pd.date_range(last_date, periods=cw_days_to_expiry, freq='D')
    visualize_df['date'] = pd.date_range(
        last_date, periods=cw_days_to_expiry, freq='D')
    visualize_df = visualize_df.set_index('date')
    visualize_df['mean'] = visualize_df.mean(axis=1)
    visualize_df['min'] = visualize_df.min(axis=1)
    visualize_df['max'] = visualize_df.max(axis=1)
    visualize_df['std'] = visualize_df.std(axis=1)
    visualize_df['std_up'] = visualize_df['mean'] + visualize_df['std']
    visualize_df['std_down'] = visualize_df['mean'] - visualize_df['std']
    visualize_df['median'] = visualize_df.median(axis=1)
    visualize_df['stock_price'] = stock_price
    visualize_df['break_even_price'] = cw_ticker_break_even_price
    visualize_df = visualize_df[['mean', 'min', 'max', 'std', 'stock_price', 'break_even_price', 'std_up', 'std_down', 'median']]
    # concat df close price and simulation results
    visualize_df = pd.concat([df.set_index('date')['close'], visualize_df], axis=1)

    visualize_df['hist_break_even_price'] = None
    break_even_price_df = cw_ticker_data_df[['date', 'break_even_price']]
    break_even_price_df = break_even_price_df.set_index('date')
    visualize_df['hist_break_even_price'] = break_even_price_df


    # plot close price and simulation results

    fig = go.Figure()
    # trace for stock price, color blue
    fig.add_trace(go.Scatter(x=visualize_df.index, y=visualize_df['stock_price'], mode='lines', name='stock_price', line=dict(color='blue')))
    # trace for projected break even price, color red
    fig.add_trace(go.Scatter(x=visualize_df.index, y=visualize_df['break_even_price'], mode='lines', name='break_even_price', line=dict(color='red', dash='dash')))
    # trace for min price, color ora
    fig.add_trace(go.Scatter(x=visualize_df.index, y=visualize_df['min'], mode='lines', name='min', line=dict(color='orange', dash='dash')))
    # trace for max price, color ora
    # fig.add_trace(go.Scatter(x=visualize_df.index, y=visualize_df['max'], mode='lines', name='max', line=dict(color='orange', dash='dash')))
    # trace for historical break even price, color red
    fig.add_trace(go.Scatter(x=visualize_df.index, y=visualize_df['hist_break_even_price'], mode='lines', name='hist_break_even_price', line=dict(color='red')))
    # trace for mean, color green
    fig.add_trace(go.Scatter(x=visualize_df.index, y=visualize_df['mean'], mode='lines', name='mean', line=dict(color='green')))
    # trace for std_up, color black
    fig.add_trace(go.Scatter(x=visualize_df.index, y=visualize_df['std_up'], mode='lines', name='std', line=dict(color='black', dash='dash')))
    # trace for std_down, color black
    fig.add_trace(go.Scatter(x=visualize_df.index, y=visualize_df['std_down'], mode='lines', name='std', line=dict(color='black', dash='dash'), fill='tonexty', fillcolor='rgba(0,100,80,0.2)'))
    # trace for close price, color black
    fig.add_trace(go.Scatter(x=visualize_df.index, y=visualize_df['close'], mode='lines', name='close', line=dict(color='black')))
    # trace for median, color black
    # fig.add_trace(go.Scatter(x=visualize_df.index, y=visualize_df['median'], mode='lines', name='median', line=dict(color='black', dash='dashdot')))
    # add vertical line x = cw_issue_date, y = stock_price
    fig.add_shape(
        type="line",
        x0=cw_issue_date,
        y0=0,
        x1=cw_issue_date,
        y1=stock_price,
        line=dict(
            color="Blue",
            width=3,
            dash="dashdot",
        )
    )
    st.plotly_chart(fig, use_container_width=True)


    # plot histogram of simulation results
    fig = px.histogram(mc_last_results, x=mc_last_results.values)
    # adding horizontal line for break even price
    fig.add_shape(
        type="line",
        x0=cw_break_even_price,
        y0=0,
        x1=cw_break_even_price,
        y1=len(mc_last_results.values)/25,
        name='break_even_price',
        line=dict(
            color="Red",
            width=3,
            dash="dashdot",
        )
    )
    # adding horizontal line for mean price
    fig.add_shape(
        type="line",
        x0=mc_mean,
        y0=0,
        x1=mc_mean,
        y1=len(mc_last_results.values)/25,
        name='mean',
        line=dict(
            color="Green",
            width=3,
            dash="dashdot",
        )
    )
    # ading horizontal line for stock price
    fig.add_shape(
        type="line",
        x0=stock_price,
        y0=0,
        x1=stock_price,
        y1=len(mc_last_results.values)/25,
        name='stock_price',
        line=dict(
            color="Blue",
            width=3,
            dash="dashdot",
        )
    )
    # fake trace for legend
    fig.add_trace(go.Scatter(x=[None], y=[None], name='break_even_price', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=[None], y=[None], name='expected_price', line=dict(color='green')))
    fig.add_trace(go.Scatter(x=[None], y=[None], name='stock_price', line=dict(color='blue')))
    # show legend
    fig.update_layout(
        title=f'{cw_ticker} simulation results',
        xaxis_title='Price',
        yaxis_title='Count',
        legend_title='Stats',
        font=dict(
            family="Courier New, monospace",
            size=18,
            color="RebeccaPurple"
        )
    )
    st.plotly_chart(fig, use_container_width=True)
