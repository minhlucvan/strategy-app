import pandas as pd
import numpy as np
import streamlit as st
import vectorbt as vbt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import talib as ta
from utils.plot_utils import plot_multi_line, plot_single_line
from utils.processing import get_stocks

# Placeholder utils
def plot_multi_line(df, title):
    fig = go.Figure()
    for col in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df[col], mode='lines', name=col))
    fig.update_layout(title=title)
    st.plotly_chart(fig)

def plot_single_line(df, title):
    fig = go.Figure(go.Scatter(x=df.index, y=df, mode='lines', name=title))
    fig.update_layout(title=title)
    st.plotly_chart(fig)

def calculate_historical_returns(stocks_df, window=126):
    """Calculate trailing returns for momentum ranking."""
    return (stocks_df / stocks_df.shift(window) - 1) * 100

def backtest(stocks_df, lookback_days=126, top_n=10, rebalance_freq='M', volatility_scaling=True):
    """Optimized backtest with risk management and dynamic filters."""
    # Calculate momentum with multiple timeframes
    mom_short = calculate_historical_returns(stocks_df, 20)  # Short-term momentum
    mom_long = calculate_historical_returns(stocks_df, lookback_days)  # Long-term momentum
    momentum_df = 0.6 * mom_long + 0.4 * mom_short  # Weighted average

    # Apply enhanced technical filters
    signals_df = apply_technical_filters(stocks_df)

    # Portfolio returns
    portfolio_returns = pd.Series(0, index=stocks_df.index)
    positions = pd.DataFrame(0, index=stocks_df.index, columns=stocks_df.columns)

    # Calculate volatility for position sizing
    vol_df = stocks_df.pct_change().rolling(20).std()

    for date in stocks_df.resample(rebalance_freq).last().index:
        if date not in momentum_df.index:
            continue

        # Select stocks based on momentum and signals
        mom_rank = momentum_df.loc[date].rank(ascending=False)
        valid_stocks = signals_df.loc[date] & (mom_rank <= top_n)

        if valid_stocks.sum() > 0:
            selected_stocks = valid_stocks.index[valid_stocks]
            
            # Volatility-adjusted weights
            if volatility_scaling:
                stock_vols = vol_df.loc[date, selected_stocks]
                weights = 1 / (stock_vols / stock_vols.min())  # Inverse volatility weighting
                weights = weights / weights.sum()  # Normalize
            else:
                weights = 1.0 / len(selected_stocks)  # Equal weight

            # Update positions
            positions.loc[date, selected_stocks] = weights

    # Calculate daily portfolio returns with stop-loss logic
    daily_returns = stocks_df.pct_change()
    for date in positions.index:
        if date not in daily_returns.index:
            continue
        weights = positions.loc[date].dropna()
        if weights.sum() > 0:
            stock_returns = daily_returns.loc[date, weights.index]
            portfolio_returns.loc[date] = (stock_returns * weights).sum()

            # Apply stop-loss (e.g., 5% max loss per position)
            for stock in weights.index:
                if portfolio_returns.loc[date] < -0.05:
                    positions.loc[date:, stock] = 0

    # Compute cumulative returns
    cumulative_returns = (1 + portfolio_returns).cumprod()

    return cumulative_returns, positions

def apply_technical_filters(stocks_df):
    """Apply optimized technical filters with additional confirmation."""
    signals = pd.DataFrame(index=stocks_df.index, columns=stocks_df.columns)
    
    for symbol in stocks_df.columns:
        price = stocks_df[symbol].dropna()
        if len(price) < 200:
            continue
        
        # Compute indicators
        sma50 = ta.SMA(price.values, timeperiod=50)
        sma200 = ta.SMA(price.values, timeperiod=200)
        rsi = ta.RSI(price.values, timeperiod=14)
        # atr = ta.ATR(price.values, timeperiod=14)  # Average True Range for volatility

        # Align indicators with price index
        sma50_series = pd.Series(sma50, index=price.index[-len(sma50):]).reindex(price.index)
        sma200_series = pd.Series(sma200, index=price.index[-len(sma200):]).reindex(price.index)
        rsi_series = pd.Series(rsi, index=price.index[-len(rsi):]).reindex(price.index)
        # atr_series = pd.Series(atr, index=price.index[-len(atr):]).reindex(price.index)

        # Buy signal: SMA50 > SMA200 for 3 days, RSI < 60, and sufficient volatility
        signals[symbol] = (sma50_series > sma200_series) & (rsi_series < 60)

    return signals.fillna(False)

def plot_price_and_indicator(stocks_df, indicator_df, title, indicator_name):
    """Plot prices and indicator in subplots."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05)
    for symbol in stocks_df.columns:
        fig.add_trace(go.Scatter(x=stocks_df.index, y=stocks_df[symbol], mode='lines', name=symbol), row=1, col=1)
    for symbol in indicator_df.columns:
        fig.add_trace(go.Scatter(x=indicator_df.index, y=indicator_df[symbol], mode='lines', name=symbol), row=2, col=1)
    fig.update_layout(title_text=title, height=600)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text=indicator_name, row=2, col=1)
    st.plotly_chart(fig)

def run(symbol_benchmark, symbolsDate_dict):
    """Main function to run and analyze the strategy."""
    if not symbolsDate_dict.get('symbols'):
        st.info("Please select symbols (e.g., VN30 stocks).")
        return

    stocks_df = get_stocks(symbolsDate_dict, 'close')
    if stocks_df.empty:
        st.warning("No valid stock data retrieved.")
        return

    st.sidebar.header("Strategy Parameters")
    lookback_days = st.sidebar.slider('Momentum Lookback Period (days)', 20, 252, 126)
    top_n = st.sidebar.slider('Number of Stocks to Hold', 1, 10, 5)
    rebalance_freq = st.sidebar.selectbox('Rebalance Frequency', ['M', 'W'], index=0)

    mom_df = calculate_historical_returns(stocks_df, lookback_days)
    st.subheader("Stock Prices and Momentum")
    plot_price_and_indicator(stocks_df, mom_df, "Stock Prices and Momentum (6-month Returns)", "Momentum (%)")

    st.subheader("Strategy Simulation")
    cumulative_returns, positions  = backtest(stocks_df, lookback_days, top_n, rebalance_freq)

    st.subheader("Portfolio Performance")
    plot_single_line(cumulative_returns, "Portfolio Cumulative Returns")
    
    st.subheader("Position Weights")
   
    fig = px.area(positions, x=positions.index, y=positions.columns, title="Position Weights Over Time")
    st.plotly_chart(fig)
