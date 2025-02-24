import streamlit as st
import pandas as pd
from utils.processing import get_stocks, get_stocks_financial, get_stocks_income_statement
from utils.plot_utils import plot_multi_line, plot_multi_bar
import utils.plot_utils as pu
import plotly.express as px
import numpy as np

import pandas as pd
import numpy as np

def run_backtest(factors, stocks_df, initial_capital=20_000_000, top_n=3):
    # Preprocess prices with forward-fill
    dates = sorted(factors['score'].index.unique())
    prices_df = stocks_df['close'].reindex(dates, method='ffill')
    
    # Initialize portfolio tracking
    trades = []
    current_positions = {}
    
    # Process each trading date
    for i, date in enumerate(dates):
        # Get top 3 stocks based on score
        top_3_symbols = factors['score'].loc[date].nlargest(top_n).index
        
        # Calculate position sizing (equal weighting)
        cash_per_stock = initial_capital / 3
        
        # Get next date for exit price (if available)
        next_date = dates[i + 1] if i < len(dates) - 1 else None
        
        # Process new and existing positions
        new_positions = set(top_3_symbols)
        existing_positions = set(current_positions.keys())
        
        # Close positions no longer in top 3
        for symbol in existing_positions - new_positions:
            if symbol in current_positions:
                position = current_positions[symbol]
                try:
                    exit_price = prices_df[symbol].loc[date]
                    if pd.isna(exit_price):
                        continue
                    position['exit_date'] = date
                    position['exit_price'] = exit_price
                    trades.append(position)
                    del current_positions[symbol]
                except (KeyError, IndexError):
                    continue
        
        # Open new positions
        for symbol in new_positions:
            if symbol not in current_positions:
                try:
                    entry_price = prices_df[symbol].loc[date]
                    if pd.isna(entry_price):
                        continue
                        
                    shares = np.floor(cash_per_stock / entry_price)
                    
                    current_positions[symbol] = {
                        'symbol': symbol,
                        'entry_date': date,
                        'entry_price': entry_price,
                        'shares': shares,
                        'position_value': shares * entry_price
                    }
                except (KeyError, IndexError):
                    continue
    
    # Close remaining positions on last date
    last_date = dates[-1]
    for symbol, position in list(current_positions.items()):
        try:
            exit_price = prices_df[symbol].loc[last_date]
            if pd.isna(exit_price):
                continue
            position['exit_date'] = last_date
            position['exit_price'] = exit_price
            trades.append(position)
        except (KeyError, IndexError):
            continue
    
    # Convert to DataFrame
    portfolio = pd.DataFrame(trades)
    
    # Calculate returns
    portfolio['return'] = (portfolio['exit_price'] - portfolio['entry_price']) / portfolio['entry_price']
    portfolio['dollar_return'] = portfolio['return'] * portfolio['position_value']
    
    # Calculate portfolio statistics
    total_return = portfolio['dollar_return'].sum() / initial_capital
    avg_trade_return = portfolio['return'].mean()
    
    stats = {
        'Total Return': total_return,
        'Average Trade Return': avg_trade_return,
        'Number of Trades': len(portfolio),
        'Win Rate': len(portfolio[portfolio['return'] > 0]) / len(portfolio)
    }
    
    return portfolio, stats

def normalize_score(series):
    """Normalize a series to a 0-100 scale using percentile ranking."""
    return (series.rank(pct=True) * 100).clip(0, 100)

# --- Data Fetching and Processing ---
def fetch_data(symbol_benchmark, symbolsDate_dict):
    """Fetch price, financial, and income statement data for given symbols."""
    if len(symbolsDate_dict['symbols']) < 1:
        st.info("Please select symbols (e.g., VNM, HPG, VCB for Vietnamese stocks).")
        st.stop()

    # Benchmark data (e.g., VNINDEX)
    benchmark_dict = symbolsDate_dict.copy()
    benchmark_dict['symbols'] = [symbol_benchmark]

    # Price data
    stocks_df = get_stocks(symbolsDate_dict, stack=True)
    stocks_df.index = pd.to_datetime(stocks_df.index).tz_localize(None)

    # Financial data
    financials_df = get_stocks_financial(symbolsDate_dict, stack=True)

    # Income statement data
    income_df = get_stocks_income_statement(symbolsDate_dict, stack=True)

    return stocks_df, financials_df, income_df

def compute_growth_factors(stocks_df, financials_df, income_df):
    """Compute individual growth factors from price, financial, and income data."""
    factors = {}
    
    # Income Statement Factors
    factors['revenue'] = income_df['revenue']    
    
    liqudity_df = stocks_df['close'] * stocks_df['volume']
    factors['liquidity'] = liqudity_df
    
    market_liquidity = liqudity_df.sum(axis=1)
    factors['market_liquidity'] = pd.DataFrame(index=liqudity_df.index, data=market_liquidity.values, columns=['market_liquidity'])
    
    # smooth window 5
    factors['market_liquidity'] = factors['market_liquidity'].rolling(window=5).mean()
    
    
    liquidity_ratio = liqudity_df.div(market_liquidity, axis=0)
    factors['liquidity_ratio'] = liquidity_ratio
    factors['liquidity_ratio'] = factors['liquidity_ratio'].rolling(window=5).mean()
    
    MA_20 = liquidity_ratio.rolling(window=20).mean()
    SD_20 = liquidity_ratio.rolling(window=20).std()
    lower_channel = MA_20 - 0.8 * SD_20
    factors['lower_channel'] = lower_channel
        
    liquidity_growth_ratio = liquidity_ratio.pct_change()
    factors['liquidity_growth_ratio'] = liquidity_growth_ratio.fillna(0)
    
    liquidity_growth_ratio_acc = liquidity_growth_ratio.cumsum()
    factors['liquidity_growth_ratio_acc'] = liquidity_growth_ratio_acc.fillna(0)
    
    liqdity_growth = liqudity_df.pct_change()
    factors['liquidity_growth'] = liqdity_growth.pct_change().rolling(window=50).sum().cumsum()
        
    liquidity_growth_acc = liqdity_growth.cumsum()
    factors['liquidity_growth_acc'] = liquidity_growth_acc
    
    # Earnings Growth (EPS Growth Rate)
    factors['eps_growth'] = financials_df['earningPerShare'].pct_change()
    
    # Revenue Growth
    factors['revenue_growth'] = factors['revenue'].pct_change()
    
    # Return on Equity (ROE)
    factors['roe'] = financials_df['roe']
    
    # Price-to-Earnings Growth (PEG) Ratio
    factors['peg'] = financials_df['priceToEarning'] / factors['eps_growth']
    
        
    # score factors
    # eps_growth_rank, revenue_growth_rank, roe_rank, peg_rank, liquidity_ratio_rank
    factors['eps_growth_rank'] = factors['eps_growth'].rank(pct=True)
    factors['revenue_growth_rank'] = factors['revenue_growth'].rank(pct=True)
    factors['roe_rank'] = factors['roe'].rank(pct=True)
    factors['peg_rank'] = factors['peg'].rank(pct=True)
    factors['liquidity_ratio_rank'] = factors['liquidity_ratio'].rank(pct=True)
    factors['liquidity_growth_rank'] = factors['liquidity_growth'].rank(pct=True)
    
    # score = average of ranks
    factors['score'] = factors['eps_growth_rank'] + factors['revenue_growth_rank'] + factors['roe_rank'] + factors['peg_rank'] + factors['liquidity_growth_rank']
    factors['score'] = factors['score'] / 5
    factors['score'] = factors['score'].ffill()
    
    return factors

# --- Main Execution ---
def run(symbol_benchmark, symbolsDate_dict):
    """Run the full analysis."""
    # Fetch data
    stocks_df, financials_df, income_df = fetch_data(symbol_benchmark, symbolsDate_dict)

    factors = compute_growth_factors(stocks_df, financials_df, income_df)
    
    # top 10 score
    last_row = factors['score'].iloc[-1]
    top_10_score = last_row.nlargest(10)
    pu.plot_single_bar(top_10_score, title='Top 10 Score', x_title='Symbol', y_title='Score')
    
    top3_symbols = top_10_score.index[:3]
    
    all_symbols = factors['score'].columns
    
    # liquidity_rank scatter
    selected_symbols = st.multiselect('Select symbols to highlight', symbolsDate_dict['symbols'], default=all_symbols)

    pu.plot_multi_line(stocks_df['close'][selected_symbols], title='Stocks Close')


    pu.plot_multi_line(factors['score'][selected_symbols], title='Stocks Score')

    # Backtest
    
    portfolio, stats = run_backtest(factors, stocks_df)
    
    st.write("## Portfolio Performance")
    st.write(stats)
    st.write(portfolio)

    # Plotting
    pu.plot_single_line(portfolio['return'].cumsum(), title='Portfolio Return', x_title='Date', y_title='Return')