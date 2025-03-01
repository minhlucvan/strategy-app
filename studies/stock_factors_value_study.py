import pandas as pd
import numpy as np
import streamlit as st
import vectorbt as vbt
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from utils.processing import get_stocks, get_stocks_valuation, get_stocks_financial
import utils.plot_utils as pu
import utils.stock_utils as su

def plot_backtest_results(pf, benchmark_returns):
    """Plot backtest results including portfolio value and key metrics"""
    fig = make_subplots(rows=2, cols=1, 
                       subplot_titles=('Portfolio Value', 'Cumulative Returns'),
                       vertical_spacing=0.1)
    
    # Plot portfolio value
    fig.add_trace(
        go.Scatter(x=pf.value().index, 
                  y=pf.value(), 
                  name='Portfolio Value',
                  line=dict(color='blue')),
        row=1, col=1
    )
    
    # Calculate cumulative returns
    portfolio_cum_returns = (1 + pf.returns()).cumprod()
    benchmark_cum_returns = (1 + benchmark_returns).cumprod()
    
    # Plot cumulative returns
    fig.add_trace(
        go.Scatter(x=portfolio_cum_returns.index, 
                  y=portfolio_cum_returns, 
                  name='Portfolio',
                  line=dict(color='blue')),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=benchmark_cum_returns.index, 
                  y=benchmark_cum_returns, 
                  name='Benchmark',
                  line=dict(color='red')),
        row=2, col=1
    )
    
    fig.update_layout(
        height=800,
        showlegend=True,
        title_text="Value Factor Strategy Backtest Results"
    )
    fig.update_yaxes(title_text="Value ($)", row=1, col=1)
    fig.update_yaxes(title_text="Cumulative Return", row=2, col=1)
    
    # Calculate key metrics
    metrics = {
        'Total Trades': pf.trades.count(),
        'Total Return': pf.total_return(),
        'Sharpe Ratio': pf.sharpe_ratio(),
        'Max Drawdown': pf.max_drawdown(),
        'Annualized Return': pf.annualized_return(),
        'Annualized Volatility': pf.annualized_volatility()
    }
    
    trades_df = pf.trades.records
    
    # map trades col to symbol
    trades_df['symbol'] = trades_df['col'].map(dict(enumerate(pf.wrapper.columns)))
    
    # pick only relevant columns
    trades_df = trades_df[['col', 'symbol', 'direction', 'size', 'entry_price', 'exit_price', 'return', 'pnl']]
    
    # direction: 0 for long, 1 for short
    trades_df['direction'] = trades_df['direction'].map({0: 'Long', 1: 'Short'})
    
    return fig, metrics, trades_df

def backtest_value_factor(stocks_df, pe_ratio_df, benchmark_df, rebalance_period='M'):
    """
    Backtest a value factor strategy using target percentage weights
    
    Parameters:
    - stocks_df: DataFrame of stock prices (close)
    - pe_ratio_df: DataFrame of PE/Industry PE ratios
    - benchmark_df: DataFrame of benchmark prices
    - rebalance_period: Rebalancing frequency ('M' for monthly, 'Q' for quarterly, etc.)
    
    Returns:
    - pf: vectorbt Portfolio object
    - benchmark_returns: Series of benchmark returns
    """
    # Ensure proper index alignment
    stocks_df = stocks_df.dropna(how='all')
    pe_ratio_df = pe_ratio_df.reindex(stocks_df.index).fillna(method='ffill')
    benchmark_df = benchmark_df.reindex(stocks_df.index).fillna(method='ffill')
    
    # Calculate factor scores (already PE/Industry PE ratio)
    factor_scores = pe_ratio_df
    
    # Create portfolio weights
    def create_portfolio_weights(scores, n_stocks=5):
        # Rank stocks (lower PE ratio = better value)
        ranks = scores.rank(axis=1, ascending=True)
        # Select top N stocks
        weights = (ranks <= n_stocks).astype(float)
        # Equal weight among selected stocks
        weights = weights.div(weights.sum(axis=1), axis=0).fillna(0)
        return weights
    
    portfolio_weights = create_portfolio_weights(factor_scores)
    
    # Create size DataFrame for orders
    size = pd.DataFrame.vbt.empty_like(stocks_df, fill_value=np.nan)
    
    # Set rebalance dates
    rebalance_dates = stocks_df.index.to_series().resample(rebalance_period).first()
    
    # Fill size with target weights at rebalance points
    for date in rebalance_dates:
        if date in portfolio_weights.index:
            size.loc[date] = portfolio_weights.loc[date]
    
    # Create portfolio using from_orders
    pf = vbt.Portfolio.from_orders(
        close=stocks_df,
        size=size,
        size_type='targetpercent',
        group_by=True,  # Treat all stocks as one group
        cash_sharing=True,  # Share cash across all assets
        init_cash=1000000,
        freq='1D',
        call_seq='auto'  # Automatically determine order execution sequence
    )
    
    # Calculate benchmark returns
    benchmark_returns = benchmark_df.pct_change()
    
    return pf, benchmark_returns

def backtest_value_factor_v2(stocks_df, pe_ratio_df, benchmark_df, entry_threshold=0.8, exit_threshold=1.2):
    """
    Backtest a value factor strategy that holds until PE ratio reaches sell threshold
    
    Parameters:
    - stocks_df: DataFrame of stock prices (close)
    - pe_ratio_df: DataFrame of PE/Industry PE ratios
    - benchmark_df: DataFrame of benchmark prices
    - entry_threshold: PE ratio below which to buy (relative to industry)
    - exit_threshold: PE ratio above which to sell (relative to industry)
    
    Returns:
    - pf: vectorbt Portfolio object
    - benchmark_returns: Series of benchmark returns
    """
    # Ensure proper index alignment
    stocks_df = stocks_df.dropna(how='all')
    pe_ratio_df = pe_ratio_df.reindex(stocks_df.index).fillna(method='ffill')
    benchmark_df = benchmark_df.reindex(stocks_df.index).fillna(method='ffill')
    
    # Calculate factor scores
    factor_scores = pe_ratio_df
    
    # Create entry and exit signals
    def create_signals(scores, entry_thresh, exit_thresh, n_stocks=5):
        # Entry: when PE ratio is below threshold and among top N lowest
        ranks = scores.rank(axis=1, ascending=True)
        entry_signals = (scores < entry_thresh) & (ranks <= n_stocks)
        
        # Exit: when PE ratio exceeds threshold
        exit_signals = scores > exit_thresh
        
        # Create weights (equal weighting among selected stocks)
        weights = pd.DataFrame(0.0, index=scores.index, columns=scores.columns)
        
        current_holdings = pd.Series(0.0, index=scores.columns)
        
        for date in scores.index:
            # Check exits first
            current_exits = exit_signals.loc[date]
            current_holdings[current_exits] = 0.0
            
            # Check entries only if we have room for new positions
            active_positions = (current_holdings > 0).sum()
            if active_positions < n_stocks:
                new_entries = entry_signals.loc[date] & (current_holdings == 0)
                n_new = min(n_stocks - active_positions, new_entries.sum())
                if n_new > 0:
                    # Select top remaining entries
                    available_entries = scores.loc[date][new_entries].nsmallest(n_new)
                    current_holdings[available_entries.index] = 1.0
            
            # Assign weights to current holdings
            n_holdings = current_holdings.sum()
            if n_holdings > 0:
                weights.loc[date] = current_holdings / n_holdings
        
        return weights
    
    portfolio_weights = create_signals(factor_scores, entry_threshold, exit_threshold)
    
    # Create size DataFrame for orders
    size = pd.DataFrame.vbt.empty_like(stocks_df, fill_value=np.nan)
    
    # Only place orders when weights change
    weight_changes = portfolio_weights.diff()
    size[weight_changes != 0] = portfolio_weights[weight_changes != 0]
    
    # Create portfolio using from_orders
    pf = vbt.Portfolio.from_orders(
        close=stocks_df,
        size=size,
        size_type='targetpercent',
        group_by=True,
        cash_sharing=True,
        init_cash=1000000,
        freq='1D',
        call_seq='auto'
    )
    
    # Calculate benchmark returns
    benchmark_returns = benchmark_df.pct_change()
    
    return pf, benchmark_returns

# Example usage with the updated function:
def run(symbol_benchmark, symbolsDate_dict):
    st.write("## Value Factor Study")
    with st.expander("About this study", expanded=False):
        st.write("""
        This script implements a Value Factor Investing Strategy, which selects stocks based on their Price-to-Earnings (PE) ratio relative to industry averages. The key approach is:

        - **Stock Selection**: Stocks with the lowest PE ratios (indicating potential undervaluation) are ranked, and the top 5 are selected.
        - **Portfolio Weighting**: Selected stocks receive equal weight in the portfolio.
        - **Rebalancing**: The portfolio is rebalanced monthly to ensure holdings reflect the latest valuation metrics.
        """)
    # Get data
    benchmark_df = get_stocks(symbolsDate_dict, 'close', benchmark=True)
    stocks_df = get_stocks(symbolsDate_dict, 'close')
    financials_df = get_stocks_financial(symbolsDate_dict, stack=True)
    eps_df = financials_df['earningPerShare']
    
    
    # reindex eps_df to match stocks_df
    eps_df = eps_df.reindex(stocks_df.index, method='ffill')
    
    all_industries = su.get_all_industries()
    
    selected_industry = st.selectbox("Select Industry", all_industries)
    
    # Filter stocks by industry
    stocks_industry_df = su.filter_stocks_by_industry(stocks_df, selected_industry)
    eps_industry_df = su.filter_stocks_by_industry(eps_df, selected_industry)   
    real_pe_df = stocks_industry_df.div(eps_industry_df)
    
    
    # Calculate real-time Price-to-Book (P/B) ratio
    book_value_df = financials_df['bookValuePerShare']
    
    # Reindex book_value_df to match stocks_df
    book_value_df = book_value_df.reindex(stocks_df.index, method='ffill')
    
    # Filter book value by industry
    book_value_industry_df = su.filter_stocks_by_industry(book_value_df, selected_industry)
    
    
    # Calculate P/B ratio
    real_pb_df = stocks_industry_df.div(book_value_industry_df)
    real_pb_ratio_df = real_pb_df.div(real_pb_df.mean(axis=1), axis=0)
    
    
    pu.plot_multi_line(real_pe_df, title=f"Real PE Ratio in {selected_industry}")
    
    pu.plot_multi_line(real_pb_df, title=f"Real P/B Ratio in {selected_industry}")
    
    selected_stocks = st.multiselect("Select Stocks", stocks_industry_df.columns)
    
    real_pb_ratio_selected_df = real_pb_ratio_df[selected_stocks]
    real_pe_selected_df = real_pe_df[selected_stocks]
    stocks_selected_df = stocks_industry_df[selected_stocks]
    
    pu.plot_multi_line(real_pe_selected_df, title=f"Real PE Ratio Relative to Industry in {selected_industry}")
    
    pu.plot_multi_line(real_pb_ratio_selected_df, title=f"Real P/B Ratio Relative to Industry in {selected_industry}")
    
    pu.plot_multi_line(stocks_selected_df, title=f"Stocks in {selected_industry}")