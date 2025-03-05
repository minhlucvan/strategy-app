import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
import plotly.express as px
from utils.processing import get_stocks
import utils.plot_utils as pu
from itertools import product

# Momentum calculation
def calc_momentum(df):
    return df.pct_change() + 1

def rolling_return(df, window):
    return df.rolling(window).apply(np.prod)

def get_top_stocks(month, mom, filters):
    top = mom.columns
    for n, ret in filters:
        top = ret.loc[month, top].nlargest(n).index
    return top

def portfolio_performance(month, exposure, mom, filters):
    top = get_top_stocks(month, mom, filters)
    next_month_ret = mom.loc[month:, top][1:2]  # Next month
    next_month_ret -= 0.002  # Transaction cost + tax (0.1% each)
    portfolio = next_month_ret.mean(axis=1) * exposure + (1 - exposure)
    return portfolio.values[0]

def compute_strategy(df, filters, exposure=1.0):
    mom = calc_momentum(df)
    months_ret = [[n, rolling_return(mom, m)] for n, m in filters]
    returns = []
    
    for month in mom.index[:-1]:
        perf = portfolio_performance(month, exposure, mom, months_ret)
        returns.append(perf)
    
    returns = pd.Series(returns[12:], index=mom.index[13:])  # Skip first 12 months
    return returns

def sharpe_ratio(returns, risk_free_rate=0.02):
    """Calculate annualized Sharpe Ratio"""
    excess_returns = returns - risk_free_rate / 12  # Assuming monthly returns
    return np.sqrt(12) * excess_returns.mean() / excess_returns.std()

def optimize_filters(df, n_range, m_range, exposure=1.0, metric='sharpe'):
    """Optimize filter parameters based on specified metric"""
    results = []
    
    # Test all combinations of n and m
    for n, m in product(n_range, m_range):
        filters = [[n, m]]
        returns = compute_strategy(df, filters, exposure)
        
        if metric == 'sharpe':
            score = sharpe_ratio(returns)
        elif metric == 'total_return':
            score = returns.cumprod().iloc[-1] - 1
        
        results.append({
            'n': n,
            'm': m,
            'score': score,
            'returns': returns
        })
    
    # Sort by score and return best parameters
    best_result = max(results, key=lambda x: x['score'])
    return best_result

# Streamlit UI
def run(symbol_benchmark, symbolsDate_dict):
    st.title("VNINDEX Momentum Strategy with Optimization")
    st.info("Portfolio rebalances monthly based on past performance. Check back on the 1st of each month for updates.")
    
    st.write("**Manual Filters (Top N stocks over M months):**")
    filters = []
    default_filters = [[1, 2]]
    
    for i in range(len(default_filters)):
        col1, col2 = st.columns(2)
        n = col1.number_input(f"Top stocks (Filter {i+1})", min_value=1, max_value=100, 
                            value=default_filters[i][0], step=1)
        m = col2.number_input(f"Months (Filter {i+1})", min_value=1, max_value=36, 
                            value=default_filters[i][1], step=1)
        filters.append([n, m])
    
    # Optimization parameters
    st.write("**Optimization Parameters:**")
    col1, col2 = st.columns(2)
    n_min = col1.number_input("Min Top Stocks", min_value=1, max_value=20, value=5, step=1)
    n_max = col1.number_input("Max Top Stocks", min_value=1, max_value=20, value=5, step=1)
    m_min = col2.number_input("Min Months", min_value=1, max_value=24, value=1, step=1)
    m_max = col2.number_input("Max Months", min_value=1, max_value=24, value=6, step=1)
    
    optimize_metric = st.selectbox("Optimization Metric", ['Sharpe Ratio', 'Total Return'])
    metric_key = 'sharpe' if optimize_metric == 'Sharpe Ratio' else 'total_return'
    
    run_manual = st.button("Run Manual Strategy")
    run_optimize = st.button("Run Optimization")

    df = get_stocks(symbolsDate_dict, 'close')
    vnindex = get_stocks(symbolsDate_dict, 'close', benchmark=True)[symbol_benchmark]
    vnindex_returns = vnindex.pct_change() + 1

    if run_manual:
        returns = compute_strategy(df, filters)
        display_results(df, returns, vnindex_returns, filters)

    if run_optimize:
        n_range = range(n_min, n_max + 1)
        m_range = range(m_min, m_max + 1)
        best_result = optimize_filters(df, n_range, m_range, metric=metric_key)
        
        st.subheader(f"Optimized Parameters (Metric: {optimize_metric})")
        st.write(f"Best Top Stocks (n): {best_result['n']}")
        st.write(f"Best Lookback Months (m): {best_result['m']}")
        st.write(f"Best {optimize_metric}: {best_result['score']:.4f}")
        
        display_results(df, best_result['returns'], vnindex_returns, [[best_result['n'], best_result['m']]])

def display_results(df, returns, vnindex_returns, filters):
    # Cumulative returns
    cum_ret = pd.DataFrame({
        'Strategy': returns.cumprod(),
        'VNINDEX': vnindex_returns.cumprod()
    })
    
    col1, col2 = st.columns(2)
    col1.metric("Strategy Return", f"{(cum_ret['Strategy'].iloc[-1] - 1):.2%}")
    col2.metric("VNINDEX Return", f"{(cum_ret['VNINDEX'].iloc[-1] - 1):.2%}")
    
    fig = px.line(cum_ret - 1, labels={'value': 'Cumulative Return', 'variable': ''})
    fig.update_layout(yaxis_tickformat='.0%', hovermode='x unified')
    st.plotly_chart(fig)

    # Current portfolio
    mom = calc_momentum(df)
    months_ret = [[n, rolling_return(mom, m)] for n, m in filters]
    current_top = get_top_stocks(mom.index[-2], mom, months_ret)
    st.subheader("Current Portfolio")
    st.write(current_top)