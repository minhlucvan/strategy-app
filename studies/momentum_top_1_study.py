import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from utils.processing import get_stocks

# Momentum calculation for single period
def calc_previous_return(df):
    return df.pct_change() + 1  # Previous bar return

def get_top_stock(month, returns):
    previous_returns = returns.loc[month]
    top_stock = previous_returns.nlargest(1).index
    return top_stock

def compute_strategy(df):
    returns = calc_previous_return(df)
    portfolio_returns = []
    
    # Create trade log DataFrame
    trade_log = pd.DataFrame(columns=['Entry_Date', 'Exit_Date', 'Ticker', 'PnL', 'Return'])
    
    # Start from second bar since we need previous bar's return
    for i, month in enumerate(returns.index[1:-1]):
        # Get top stock based on previous bar
        top_stock = get_top_stock(month, returns)
        
        if top_stock.empty:
            continue
        
        # Get entry and exit prices
        entry_price = df.loc[month, top_stock]
        
        if entry_price.empty or entry_price.isna().any():
            st.warning(f"Missing price data for {top_stock[0]} on {month}")
            continue
        
        next_month = returns.index[i + 2]  # Index for exit
        exit_price = df.loc[next_month, top_stock]
        
        if exit_price.empty or exit_price.isna().any():
            st.warning(f"Missing price data for {top_stock[0]} on {next_month}")
            continue
        
        exit_price = exit_price.values[0]
        entry_price = entry_price.values[0]
        
        # Calculate returns with transaction costs
        gross_return = (exit_price / entry_price)
        net_return = gross_return - 0.002  # Apply 0.1% transaction costs each way
        pnl = (exit_price - entry_price) * 100  # Assuming 100 shares for simplicity
        
        # Log the trade
        trade_entry = {
            'Entry_Date': month,
            'Exit_Date': next_month,
            'Ticker': top_stock[0],  # Convert Index to string
            'Entry_Price': entry_price,
            'Exit_Price': exit_price,
            'PnL': pnl,  # Convert to percentage return
            'Return': net_return - 1  # Convert to percentage return
        }
        trade_log = pd.concat([trade_log, pd.DataFrame([trade_entry])], ignore_index=True)
        
        portfolio_returns.append(net_return)
    
    # Create series with proper index
    portfolio_returns = pd.Series(portfolio_returns, index=returns.index[1:len(portfolio_returns)+1])
    
    return portfolio_returns, trade_log

def sharpe_ratio(returns, risk_free_rate=0.02):
    """Calculate annualized Sharpe Ratio"""
    excess_returns = returns - risk_free_rate / 12
    return np.sqrt(12) * excess_returns.mean() / excess_returns.std()

# Streamlit UI
def run(symbol_benchmark, symbolsDate_dict):    
    
    if len(symbolsDate_dict['symbols']) < 1:
        st.info("Please select symbols (e.g., VN30 stocks).")
        return
    
    # Load data
    df = get_stocks(symbolsDate_dict, 'close')
    
    df = df.fillna(method='ffill')
    
    vnindex = get_stocks(symbolsDate_dict, 'close', benchmark=True)[symbol_benchmark]
    vnindex_returns = vnindex.pct_change() + 1

    # Calculate strategy returns and get trade log
    portfolio_returns, trade_log = compute_strategy(df)
                
    # Cumulative returns
    cum_ret = pd.DataFrame({
        'Strategy': portfolio_returns.cumprod(),
        'VNINDEX': vnindex_returns[portfolio_returns.index[0]:].cumprod()
    })
    
    cum_ret = cum_ret.fillna(method='ffill')
        
    sharpe = sharpe_ratio(portfolio_returns)
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Strategy Return", f"{(cum_ret['Strategy'].iloc[-1] - 1):.2%}")
    col2.metric("VNINDEX Return", f"{(cum_ret['VNINDEX'].iloc[-1] - 1):.2%}")
    col3.metric("Sharpe Ratio", f"{sharpe:.2f}")
    
    # Plot cumulative returns
    fig = px.line(cum_ret - 1, labels={'value': 'Cumulative Return', 'variable': ''})
    fig.update_layout(yaxis_tickformat='.0%', hovermode='x unified')
    st.plotly_chart(fig)

    # Current portfolio
    latest_returns = calc_previous_return(df)
    current_top = get_top_stock(df.index[-2], latest_returns)
    current_entry_price = df.loc[df.index[-2], current_top]
    current_exit_price = df.loc[df.index[-1], current_top]
    curent_return = (current_exit_price / current_entry_price).values[0] - 1
    current_df = pd.DataFrame({
        'Ticker': current_top,
        'Entry_Price': current_entry_price.values[0],
        'Exit_Price': current_exit_price.values[0],
        'Return': curent_return
    })
    st.subheader("Current Portfolio (Top Stock)")
    st.write(current_df)

    show_trade_log = st.checkbox("Show Trade Log")
    if show_trade_log:
        # Display trade log
        st.subheader("Trade History")
        # Format the trade log for better display
        display_log = trade_log.copy()
        display_log['PnL'] = display_log['PnL'].round(2)
        display_log['Return'] = display_log['Return'].apply(lambda x: f"{x:.2%}")
        display_log['Entry_Date'] = display_log['Entry_Date'].dt.strftime('%Y-%m-%d')
        display_log['Exit_Date'] = display_log['Exit_Date'].dt.strftime('%Y-%m-%d')
        st.dataframe(display_log[['Entry_Date', 'Exit_Date', 'Ticker', 'Entry_Price', 'Exit_Price', 'PnL', 'Return']])
