import pandas as pd
import numpy as np
import streamlit as st
import vectorbt as vbt
import plotly.graph_objs as go
from plotly.subplots import make_subplots

from studies.vn30_volality_study import calculate_vix_index
from utils.component import check_password, input_dates, input_SymbolsDate
from utils.processing import get_stocks
from utils.plot_utils import plot_multi_line, plot_single_line

# --- Factor Calculation Functions ---
def calculate_relative_volatility(stocks_df: pd.DataFrame, market_df: pd.Series, window: int = 21) -> pd.DataFrame:
    """Calculate annualized relative volatility using log returns for robustness."""
    stock_returns = np.log(stocks_df / stocks_df.shift(1))
    market_returns = np.log(market_df / market_df.shift(1))
    stock_vol = stock_returns.rolling(window=window).std() * np.sqrt(252)
    market_vol = market_returns.rolling(window=window).std() * np.sqrt(252)
    return stock_vol.div(market_vol, axis=0)

# --- Data Preparation ---
def prepare_data(symbol_benchmark: str, symbolsDate_dict: dict) -> tuple:
    stocks_df = get_stocks(symbolsDate_dict, 'close', benchmark=True, merge_benchmark=True)
    benchmark_df = stocks_df[symbol_benchmark] * 100_000
    stocks_df = stocks_df.drop(symbol_benchmark, axis=1)
    high = get_stocks(symbolsDate_dict, 'high')
    low = get_stocks(symbolsDate_dict, 'low')
    return stocks_df, benchmark_df, high, low

# --- Backtest Function ---
def run_backtest(prices: pd.DataFrame, rel_vol: pd.DataFrame, threshold: float) -> vbt.Portfolio:
    """Backtest a low volatility strategy."""
    entries = rel_vol < threshold
    exits = rel_vol >= threshold
    entries = entries & (~entries.shift(1).fillna(False))  # Avoid multiple entries
    exits = exits & (~exits.shift(1).fillna(False))
    pf = vbt.Portfolio.from_signals(
        close=prices,
        entries=entries,
        exits=exits,
        freq='1D',
        init_cash=100_000,
        fees=0.001,
        slippage=0.001,
        size=1.0 / len(prices.columns)  # Equal weighting across stocks
    )
    return pf

# --- Visualization Functions ---
def plot_backtest_results(pf: vbt.Portfolio):
    fig = make_subplots(rows=2, cols=1, subplot_titles=('Equity Curve', 'Drawdown'), vertical_spacing=0.1)
    equity = pf.value()
    fig.add_trace(go.Scatter(x=equity.index, y=equity, name='Equity', line=dict(color='blue')), row=1, col=1)
    drawdown = pf.drawdown()
    fig.add_trace(go.Scatter(x=drawdown.index, y=drawdown, name='Drawdown', line=dict(color='red')), row=2, col=1)
    fig.update_layout(height=600, width=800, showlegend=True)
    st.plotly_chart(fig)
    stats = pf.stats()
    st.write("### Backtest Statistics")
    st.write(f"Total Return: {stats['Total Return [%]']:.2f}%")
    st.write(f"Sharpe Ratio: {stats['Sharpe Ratio']:.2f}")
    st.write(f"Max Drawdown: {stats['Max Drawdown [%]']:.2f}%")

# --- Main Study Function ---
def run(symbol_benchmark: str, symbolsDate_dict: dict):
    st.write("## Low Volatility Factor Study")
    
    # Prepare data
    stocks_df, benchmark_df, high, low = prepare_data(symbol_benchmark, symbolsDate_dict)
    plot_multi_line(stocks_df, title='Stock Prices', x_title='Date', y_title='Price', legend_title='Stocks')
    
    # Calculate relative volatility
    hv_period = st.slider('Select Volatility Window', 10, 252, 63)  # 63 days ~ 3 months
    rel_vol = calculate_relative_volatility(stocks_df, benchmark_df, window=hv_period)
    plot_multi_line(rel_vol, title='Relative Volatility (Stock/Market)', x_title='Date', y_title='Ratio', legend_title='Stocks')
    
    # Low volatility filter
    rel_vol_threshold = st.slider('Select Low Volatility Threshold', 0.2, 1.5, 0.8)
    low_rel_vol = rel_vol[rel_vol > rel_vol_threshold]
    plot_multi_line(low_rel_vol, title='Low Relative Volatility Stocks', x_title='Date', y_title='Ratio', legend_title='Stocks')
    
    # Stock selection
    selected_stock = st.selectbox('Select Stock', low_rel_vol.columns)
    plot_multi_line(rel_vol[[selected_stock]], title=f'{selected_stock} Relative Volatility', x_title='Date', y_title='Ratio', legend_title=selected_stock)
    plot_multi_line(stocks_df[[selected_stock]], title=f'{selected_stock} Price', x_title='Date', y_title='Price', legend_title=selected_stock)
    
    # Backtest
    st.write("### Backtest Low Volatility Strategy")
    pf = run_backtest(stocks_df, rel_vol, rel_vol_threshold)
    plot_backtest_results(pf)
