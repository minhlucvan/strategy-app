import pandas as pd
import numpy as np
import streamlit as st
import vectorbt as vbt
import plotly.graph_objs as go
from plotly.subplots import make_subplots

from studies.vn30_volality_study import calculate_vix_index  # Assuming this exists elsewhere
from utils.component import check_password, input_dates, input_SymbolsDate
from utils.processing import get_stocks
from utils.plot_utils import plot_multi_line, plot_single_line

# --- Factor Calculation Functions ---
def calculate_momentum(stocks_df: pd.DataFrame, window: int = 126) -> pd.DataFrame:
    """Calculate momentum as cumulative return over the past window periods (default 6 months ~ 126 trading days)."""
    returns = stocks_df.pct_change().fillna(0)  # Daily returns
    momentum = (1 + returns).rolling(window=window).apply(np.prod, raw=True) - 1  # Cumulative return
    return momentum.shift(1)  # Shift to avoid look-ahead bias

# --- Data Preparation ---
def prepare_data(symbol_benchmark: str, symbolsDate_dict: dict) -> tuple:
    stocks_df = get_stocks(symbolsDate_dict, 'close', benchmark=True, merge_benchmark=True)
    benchmark_df = stocks_df[symbol_benchmark] * 100_000  # Scale benchmark for visualization
    stocks_df = stocks_df.drop(symbol_benchmark, axis=1)
    high = get_stocks(symbolsDate_dict, 'high')
    low = get_stocks(symbolsDate_dict, 'low')
    return stocks_df, benchmark_df, high, low

# --- Backtest Function ---
def run_backtest(prices: pd.DataFrame, momentum: pd.DataFrame, top_n: int) -> vbt.Portfolio:
    """Backtest a momentum strategy by longing top performers."""
    # Rank stocks by momentum each period
    momentum_rank = momentum.rank(axis=1, ascending=False)
    entries = momentum_rank <= top_n  # Long top N stocks
    exits = momentum_rank > top_n     # Exit when no longer in top N
    
    # Avoid multiple entries/exits on the same signal
    entries = entries & (~entries.shift(1).fillna(False))
    exits = exits & (~exits.shift(1).fillna(False))
    
    pf = vbt.Portfolio.from_signals(
        close=prices,
        entries=entries,
        exits=exits,
        freq='1D',
        init_cash=100_000,
        fees=0.001,  # 0.1% transaction cost
        slippage=0.001,
        size=1.0 / top_n  # Equal weighting among selected stocks
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
    st.write("## Momentum Factor Study")
    
    # Prepare data
    stocks_df, benchmark_df, high, low = prepare_data(symbol_benchmark, symbolsDate_dict)
    plot_multi_line(stocks_df, title='Stock Prices', x_title='Date', y_title='Price', legend_title='Stocks')
    
    # Calculate momentum
    mom_period = st.slider('Select Momentum Lookback Period (days)', 21, 252, 126)  # Default ~6 months
    momentum = calculate_momentum(stocks_df, window=mom_period)
    plot_multi_line(momentum, title='Momentum (Cumulative Return)', x_title='Date', y_title='Momentum', legend_title='Stocks')
    
    # Momentum filter
    top_n = st.slider('Select Number of Top Momentum Stocks', 1, len(stocks_df.columns), 5)
    top_momentum = momentum.apply(lambda x: x.nlargest(top_n), axis=1)
    plot_multi_line(top_momentum, title=f'Top {top_n} Momentum Stocks', x_title='Date', y_title='Momentum', legend_title='Stocks')
    
    # Stock selection
    selected_stock = st.selectbox('Select Stock', stocks_df.columns)
    plot_multi_line(momentum[[selected_stock]], title=f'{selected_stock} Momentum', x_title='Date', y_title='Momentum', legend_title=selected_stock)
    plot_multi_line(stocks_df[[selected_stock]], title=f'{selected_stock} Price', x_title='Date', y_title='Price', legend_title=selected_stock)
    
    # Backtest
    st.write("### Backtest Momentum Strategy")
    pf = run_backtest(stocks_df, momentum, top_n)
    plot_backtest_results(pf)

