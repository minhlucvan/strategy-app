import streamlit as st
import pandas as pd
import json
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
from datetime import datetime
from io import BytesIO
import numpy as np
import copy
import talib as ta

from utils.component import input_SymbolsDate, check_password, form_SavePortfolio
from utils.db import get_SymbolsName
from portfolios import get_portfolio, get_list_of_portfolios
from utils.processing import get_stocks

from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
from pypfopt import EfficientFrontier, risk_models, expected_returns, plotting


def plot_cum_returns(data, title):
    daily_cum_returns = 1 + data.dropna().pct_change()
    daily_cum_returns = daily_cum_returns.cumprod() * 100
    fig = px.line(daily_cum_returns, title=title)
    return fig


def plot_efficient_frontier_and_max_sharpe(mu, S):
    ef = EfficientFrontier(mu, S)
    fig, ax = plt.subplots(figsize=(6, 4))
    ef_max_sharpe = copy.deepcopy(ef)
    plotting.plot_efficient_frontier(ef, ax=ax, show_assets=False)
    ef_max_sharpe.max_sharpe(risk_free_rate=0.02)
    ret_tangent, std_tangent, _ = ef_max_sharpe.portfolio_performance()
    ax.scatter(std_tangent, ret_tangent, marker="*", s=100, c="r", label="Max Sharpe")
    n_samples = 1000
    w = np.random.dirichlet(np.ones(ef.n_assets), n_samples)
    rets = w.dot(ef.expected_returns)
    stds = np.sqrt(np.diag(w @ ef.cov_matrix @ w.T))
    sharpes = rets / stds
    ax.scatter(stds, rets, marker=".", c=sharpes, cmap="viridis_r")
    ax.legend()
    return fig


def check_params(params):
    return True


def load_portfolio(selected_portfolio, symbolsDate_dict):
    portfolio = get_portfolio(selected_portfolio, symbolsDate_dict)
    if not portfolio.is_ready():
        st.error("Portfolio is not ready. Please check the symbols and dates.")
        st.stop()
    return portfolio


def get_stock_data(portfolio):
    return portfolio.get_assets()


def calculate_portfolio_performance(stocks_df):
    mu = expected_returns.mean_historical_return(stocks_df)
    S = risk_models.sample_cov(stocks_df)
    return mu, S


def get_optimized_weights(mu, S):
    ef = EfficientFrontier(mu, S)
    ef.max_sharpe(risk_free_rate=0.02)
    weights = ef.clean_weights()
    expected_annual_return, annual_volatility, sharpe_ratio = ef.portfolio_performance()
    return weights, expected_annual_return, annual_volatility, sharpe_ratio

def plot_prices(stocks_df):
    return px.line(stocks_df, title='Price of Individual Stocks')

def plot_cumulative_returns(stocks_df, title):
    cumulative_returns = stocks_df.pct_change().add(1).cumprod() * 100
    fig = px.line(cumulative_returns, title=title)
    return fig

def plot_correlation(stocks_df):
    corr_df = stocks_df.corr().round(2)
    return px.imshow(corr_df, text_auto=True, title='Correlation between Stocks')

def plot_efficient_frontier(mu, S):
    fig_efficient_frontier = BytesIO()
    fig_ef = plot_efficient_frontier_and_max_sharpe(mu, S)
    fig_ef.savefig(fig_efficient_frontier, format="png")
    return fig_efficient_frontier

def plot_optimized_portfolio_cumulative_returns(stocks_df, weights):
    stocks_df['Optimized Portfolio'] = sum(stocks_df[ticker] * weight for ticker, weight in weights.items())
    return plot_cumulative_returns(stocks_df[['Optimized Portfolio']], 'Cumulative Returns of Optimized Portfolio Starting with $100')

def plot_optimized_portfolio_weights(weights):
    weights_df = pd.DataFrame.from_dict(weights, orient='index', columns=['weights'])
    weights_df = weights_df[weights_df.weights > 0.003]
    return px.pie(weights_df, values='weights', names=weights_df.index, title='Optimized Max Sharpe Portfolio Weights')

def display_results(stocks_df, weights, expected_annual_return, annual_volatility, sharpe_ratio, mu, S, tickers):

    st.subheader(f"Your Portfolio Consists of {', '.join(tickers)} Stocks")
    fig_cum_returns_optimized = plot_optimized_portfolio_cumulative_returns(stocks_df, weights)
    st.plotly_chart(fig_cum_returns_optimized)
    
    
    st.subheader("Optimized Max Sharpe Portfolio Weights")
    fig_pie = plot_optimized_portfolio_weights(weights)
    st.plotly_chart(fig_pie)
    
    st.subheader("Optimized Max Sharpe Portfolio Performance")
    fig_efficient_frontier = plot_efficient_frontier(mu, S)
    st.image(fig_efficient_frontier)
    
    st.subheader(f'Expected annual return: {(expected_annual_return * 100).round(2)}%')
    st.subheader(f'Annual volatility: {(annual_volatility * 100).round(2)}%')
    st.subheader(f'Sharpe Ratio: {sharpe_ratio.round(2)}')
    fig_corr = plot_correlation(stocks_df)
    st.plotly_chart(fig_corr)
    fig_price = plot_prices(stocks_df)
    st.plotly_chart(fig_price)
    fig_cum_returns = plot_cumulative_returns(stocks_df, 'Cumulative Returns of Individual Stocks Starting with $100')
    st.plotly_chart(fig_cum_returns)
    
    # plot momenutm indicator RSI
    st.subheader("Momentum Indicator RSI")
    stocks_df_rsi = stocks_df.copy()
    for ticker in tickers:
        stocks_df_rsi[ticker] = ta.RSI(stocks_df[ticker], timeperiod=14)
    
    fig_rsi = plot_cum_returns(stocks_df_rsi, 'RSI Indicator')
    st.plotly_chart(fig_rsi)
    
    # plot volatility indicator ATR
    st.subheader("Volatility Indicator ATR")
    stocks_df_atr = stocks_df.copy()
    
    for ticker in tickers:
        stocks_df_atr[ticker] = ta.ATR(stocks_df['High'][ticker], stocks_df['Low'][ticker], stocks_df['Close'][ticker], timeperiod=14)
        
    fig_atr = plot_cum_returns(stocks_df_atr, 'ATR Indicator')
    st.plotly_chart(fig_atr)
    
    # calculate the benchmark = sum of all stocks
    st.subheader("Benchmark")
    stocks_benchmark = stocks_df.copy()
    stocks_benchmark['Benchmark'] = stocks_df.sum(axis=1)
    fig_benchmark = plot_cum_returns(stocks_benchmark, 'Benchmark')
    st.plotly_chart(fig_benchmark)
    
    # calculate the rs_ratio = stocks / benchmark
    st.subheader("Relative Strength Ratio")
    stocks_rs_ratio = stocks_df.copy()
    stocks_rs_ratio['Benchmark'] = stocks_df.sum(axis=1)
    for ticker in tickers:
        stocks_rs_ratio[ticker] = stocks_df[ticker] / stocks_df['Benchmark']
    fig_rs_ratio = plot_cum_returns(stocks_rs_ratio, 'Relative Strength Ratio')
    st.plotly_chart(fig_rs_ratio)
    


def main():
    if check_password():
        portfolio_list = get_list_of_portfolios()
        selected_portfolio = st.sidebar.selectbox('Select Portfolio', portfolio_list)
        symbolsDate_dict = input_SymbolsDate(group=True)
        portfolio = load_portfolio(selected_portfolio, symbolsDate_dict)
        
        col1, col2 = st.columns(2)
        tickers = symbolsDate_dict['symbols']

        try:
            stocks_df = get_stock_data(portfolio)
            mu, S = calculate_portfolio_performance(stocks_df)
            weights, expected_annual_return, annual_volatility, sharpe_ratio = get_optimized_weights(mu, S)
            display_results(stocks_df, weights, expected_annual_return, annual_volatility, sharpe_ratio, mu, S, tickers)
        
        except Exception as e:
            st.error(f"Error: {e}")
            st.stop()


if __name__ == "__main__":
    main()
