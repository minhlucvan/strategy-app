import streamlit as st
import pandas as pd
import json
import plotly.graph_objects as go

from utils.component import input_SymbolsDate, check_password, params_selector, form_SavePortfolio
from utils.db import get_SymbolsName

import streamlit as st
import yfinance as yf
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
from pypfopt import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt import plotting
import copy
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from datetime import datetime
from io import BytesIO

from utils.processing import get_stocks

def plot_cum_returns(data, title):    
	daily_cum_returns = 1 + data.dropna().pct_change()
	daily_cum_returns = daily_cum_returns.cumprod()*100
	fig = px.line(daily_cum_returns, title=title)
	return fig
	
def plot_efficient_frontier_and_max_sharpe(mu, S): 
	# Optimize portfolio for max Sharpe ratio and plot it out with efficient frontier curve
	ef = EfficientFrontier(mu, S)
	fig, ax = plt.subplots(figsize=(6,4))
	ef_max_sharpe = copy.deepcopy(ef)
	plotting.plot_efficient_frontier(ef, ax=ax, show_assets=False)
	# Find the max sharpe portfolio
	ef_max_sharpe.max_sharpe(risk_free_rate=0.02)
	ret_tangent, std_tangent, _ = ef_max_sharpe.portfolio_performance()
	ax.scatter(std_tangent, ret_tangent, marker="*", s=100, c="r", label="Max Sharpe")
	# Generate random portfolios
	n_samples = 1000
	w = np.random.dirichlet(np.ones(ef.n_assets), n_samples)
	rets = w.dot(ef.expected_returns)
	stds = np.sqrt(np.diag(w @ ef.cov_matrix @ w.T))
	sharpes = rets / stds
	ax.scatter(stds, rets, marker=".", c=sharpes, cmap="viridis_r")
	# Output
	ax.legend()
	return fig



def check_params(params):
    # for key, value in params.items():
    #     if len(params[key]) < 2:
    #         st.error(f"{key} 's numbers are not enough. ")
    #         return False
    return True

if check_password():
    symbolsDate_dict = input_SymbolsDate(group=True)
    
    if len(symbolsDate_dict['symbols']) == 0:
        st.info("Please select at least one stock.")
        st.stop()
        
    col1, col2 = st.columns(2)

    start_date = symbolsDate_dict['start_date']
    end_date = symbolsDate_dict['end_date']
    tickers = symbolsDate_dict['symbols']

    try:
        # Get Stock Prices using pandas_datareader Library	
        stocks_df = get_stocks(symbolsDate_dict, 'close')
        # Plot Individual Stock Prices
        fig_price = px.line(stocks_df, title='Price of Individual Stocks')
        # Plot Individual Cumulative Returns
        fig_cum_returns = plot_cum_returns(stocks_df, 'Cumulative Returns of Individual Stocks Starting with $100')
        # Calculatge and Plot Correlation Matrix between Stocks
        corr_df = stocks_df.corr().round(2)
        fig_corr = px.imshow(corr_df, text_auto=True, title = 'Correlation between Stocks')
            
        # Calculate expected returns and sample covariance matrix for portfolio optimization later
        mu = expected_returns.mean_historical_return(stocks_df)
        S = risk_models.sample_cov(stocks_df)
        
        # Plot efficient frontier curve
        fig = plot_efficient_frontier_and_max_sharpe(mu, S)
        fig_efficient_frontier = BytesIO()
        fig.savefig(fig_efficient_frontier, format="png")
        
        # Get optimized weights
        ef = EfficientFrontier(mu, S)
        ef.max_sharpe(risk_free_rate=0.02)
        weights = ef.clean_weights()
        expected_annual_return, annual_volatility, sharpe_ratio = ef.portfolio_performance()
        weights_df = pd.DataFrame.from_dict(weights, orient = 'index')
        weights_df.columns = ['weights']
        
        # Calculate returns of portfolio with optimized weights
        stocks_df['Optimized Portfolio'] = 0
        for ticker, weight in weights.items():
            stocks_df['Optimized Portfolio'] += stocks_df[ticker]*weight
        
        # Plot Cumulative Returns of Optimized Portfolio
        fig_cum_returns_optimized = plot_cum_returns(stocks_df['Optimized Portfolio'], 'Cumulative Returns of Optimized Portfolio Starting with $100')
        
        tickers_string = ', '.join(tickers)
        
        # Display everything on Streamlit
        st.subheader("Your Portfolio Consists of {} Stocks".format(tickers_string))	
        st.plotly_chart(fig_cum_returns_optimized)
        
        st.subheader("Optimized Max Sharpe Portfolio Weights")
        st.dataframe(weights_df)
        
        st.subheader("Optimized Max Sharpe Portfolio Performance")
        st.image(fig_efficient_frontier)
        
        st.subheader('Expected annual return: {}%'.format((expected_annual_return*100).round(2)))
        st.subheader('Annual volatility: {}%'.format((annual_volatility*100).round(2)))
        st.subheader('Sharpe Ratio: {}'.format(sharpe_ratio.round(2)))
        
        st.plotly_chart(fig_corr) # fig_corr is not a plotly chart
        st.plotly_chart(fig_price)
        st.plotly_chart(fig_cum_returns)
        

        
    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()
        


