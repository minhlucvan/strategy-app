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

def apply_technical_filters(stocks_df):
    """Apply SMA and RSI filters to each stock using TA-Lib."""
    signals = pd.DataFrame(index=stocks_df.index, columns=stocks_df.columns)
    for symbol in stocks_df.columns:
        price = stocks_df[symbol].dropna()
        if len(price) < 200:  # Ensure enough data for SMA200
            signals[symbol] = False
            continue
        
        # Calculate indicators
        sma50 = ta.SMA(price.values, timeperiod=50)
        sma200 = ta.SMA(price.values, timeperiod=200)
        rsi = ta.RSI(price.values, timeperiod=14)
        
        # Align lengths by padding NaN at the start
        sma50_series = pd.Series(np.concatenate([np.full(49, np.nan), sma50]), index=price.index)
        sma200_series = pd.Series(np.concatenate([np.full(199, np.nan), sma200]), index=price.index)
        rsi_series = pd.Series(np.concatenate([np.full(13, np.nan), rsi]), index=price.index)
        
        # Buy signal: SMA50 > SMA200 and RSI < 70
        signals[symbol] = (sma50_series > sma200_series) & (rsi_series < 70)
    
    return signals.fillna(False)

def simulate_strategy(stocks_df, rebalance_freq='M', lookback_days=126, top_n=5):
    """Simulate the momentum strategy with monthly rebalancing."""
    mom_df = calculate_historical_returns(stocks_df, lookback_days)
    tech_signals = apply_technical_filters(stocks_df)
    
    mom_df = mom_df.loc[tech_signals.index]
    stocks_df = stocks_df.loc[tech_signals.index]
    
    weights = pd.DataFrame(index=stocks_df.index, columns=stocks_df.columns)
    for date in stocks_df.index:
        if date not in mom_df.index or date not in tech_signals.index:
            continue
        mom_rank = mom_df.loc[date].rank(ascending=False)
        valid_stocks = tech_signals.loc[date] & (mom_rank <= top_n)
        if valid_stocks.sum() > 0:
            weights.loc[date, valid_stocks.index] = 1.0 / valid_stocks.sum()
        else:
            weights.loc[date] = 0.0
    
    weights = weights.fillna(0.0)
    
    portfolio = vbt.Portfolio.from_holding(
        close=stocks_df,
        weights=weights,
        rebalance_freq=rebalance_freq,
        cash_sharing=False,
        call_seq='auto'
    )
    return portfolio

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
    portfolio = simulate_strategy(stocks_df, rebalance_freq, lookback_days, top_n)

    total_return = portfolio.total_return() * 100
    annualized_return = portfolio.annualized_return() * 100
    sharpe = portfolio.sharpe_ratio()
    max_drawdown = portfolio.max_drawdown() * 100

    st.write(f"**Total Return**: {total_return:.2f}%")
    st.write(f"**Annualized Return**: {annualized_return:.2f}%")
    st.write(f"**Sharpe Ratio**: {sharpe:.2f}")
    st.write(f"**Max Drawdown**: {max_drawdown:.2f}%")

    equity_curve = portfolio.value()
    st.subheader("Portfolio Equity Curve")
    plot_single_line(equity_curve, "Strategy Equity Curve")

    if symbol_benchmark:
        benchmark_df = get_stocks({'symbols': [symbol_benchmark]}, 'close')
        if not benchmark_df.empty:
            benchmark_returns = benchmark_df.pct_change().add(1).cumprod() * 100
            st.subheader("Strategy vs Benchmark")
            comparison_df = pd.DataFrame({
                "Strategy": equity_curve / equity_curve.iloc[0] * 100,
                "Benchmark": benchmark_returns[symbol_benchmark]
            })
            plot_multi_line(comparison_df, "Strategy vs Benchmark (Normalized to 100)")

    st.subheader("Portfolio Holdings")
    weights = portfolio.weights
    fig = px.area(weights, title="Portfolio Weights Over Time")
    st.plotly_chart(fig)