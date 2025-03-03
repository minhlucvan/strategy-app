import pandas as pd
import numpy as np
import streamlit as st
import vectorbt as vbt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
from utils.plot_utils import plot_multi_line, plot_single_line
from utils.processing import get_stocks

# --- Factor Calculation Functions ---
def calculate_historical_rv(stocks_df: pd.DataFrame, market_df: pd.Series, window: int = 21) -> pd.DataFrame:
    """Calculate annualized relative volatility using log returns for robustness."""
    stock_returns = np.log(stocks_df.div(stocks_df.shift(1)))
    market_returns = np.log(market_df.div(market_df.shift(1)))
    stock_vol = stock_returns.rolling(window=window).std() * np.sqrt(252)
    market_vol = market_returns.rolling(window=window).std() * np.sqrt(252)
    return stock_vol.div(market_vol, axis=0)

def calculate_historical_rs(stocks_df, benchmark_df, lookback_period):
    """
    Calculate true Relative Strength (RS) based on historical returns.
    RS = (1 + Stock Return) / (1 + Benchmark Return)
    """
    try:
        stock_returns = stocks_df.pct_change(periods=lookback_period)
        benchmark_returns = benchmark_df.pct_change(periods=lookback_period).iloc[:, 0]

        rs_df = (1 + stock_returns).div(1 + benchmark_returns, axis=0)
        return rs_df.replace([np.inf, -np.inf], np.nan)
    except Exception as e:
        st.error(f"Error in RS calculation: {str(e)}")
        return None

def run(symbol_benchmark, symbolsDate_dict):
    if not symbolsDate_dict.get('symbols'):
        st.info("Please select symbols.")
        return

    # Fetch stock & benchmark data
    stocks_df = get_stocks(symbolsDate_dict, 'close')
    if stocks_df.empty:
        st.warning("No valid stock data retrieved")
        return
        
    benchmark_df = get_stocks(symbolsDate_dict, 'close', benchmark=True)
        
    if benchmark_df.empty:
        st.warning("No valid benchmark data retrieved")
        return

    # User-defined lookback period
    lookback_period = st.slider('Select Lookback Period (days)', 5, 200, 60)

    # Calculate RS using historical returns
    rs_df = calculate_historical_rs(stocks_df, benchmark_df, lookback_period)
    if rs_df is None:
        return

    # Normalize RS for better visualization
    rs_df_normalized = (rs_df - rs_df.mean()) / rs_df.std()
    
    # snapshot of RS, top 5
    rs_snapshot = rs_df.iloc[-1]
    rs_snapshot = rs_snapshot.sort_values(ascending=False)
    st.write("### Relative Strength Snapshot")
    st.write(rs_snapshot.head(5))

    # Plot RS trends
    plot_multi_line(
        rs_df_normalized,
        title=f'Relative Strength vs {symbol_benchmark} (Lookback: {lookback_period} days)',
        x_title='Date',
        y_title='Normalized RS'
    )
    
    # calculate relative volatility
    rv_df = calculate_historical_rv(stocks_df, benchmark_df[symbol_benchmark], window=lookback_period)
    
    plot_multi_line(
        rv_df,
        title=f'Relative Volatility vs {symbol_benchmark} (Window: 21 days)',
        x_title='Date',
        y_title='Relative Volatility'
    )
    
    # snapshot of RV, top 5
    rv_snapshot = rv_df.iloc[-1]
    rv_snapshot = rv_snapshot.sort_values(ascending=False)
    st.write("### Relative Volatility Snapshot")
      
    stock_returns = stocks_df.pct_change(periods=lookback_period)

    # Symbol selection for detailed view
    selected_symbol = st.selectbox('Select symbols for detailed analysis', symbolsDate_dict['symbols'])
    if selected_symbol:
        selected_rs_df = rs_df[selected_symbol]
        selected_rv_df = rv_df[selected_symbol]
        selected_stock_df = stocks_df[selected_symbol]
        

        plot_single_line(
            selected_rs_df,
            title='Selected Symbols Relative Strength',
            x_title='Date',
            y_title='RS'
        ) 

        plot_single_line(
            selected_stock_df,
            title='Selected Stock Prices',
            x_title='Date',
            y_title='Price'
        )

        plot_single_line(
            selected_rv_df,
            title='Selected Symbols Relative Volatility',
            x_title='Date',
            y_title='RV'
        )       
        
        # Remove NaN values from selected_rv_df
        selected_rv_df_clean = selected_rv_df.dropna()

        aligned_rs_df = selected_rs_df.loc[selected_rv_df_clean.index]

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, subplot_titles=('RS vs RV', 'Stock Price'))

        scatter = px.scatter(
            x=aligned_rs_df.index,
            y=aligned_rs_df,
            color=selected_rv_df_clean,
            labels={'x': 'Date', 'y': 'RS'}
        ).data[0]
        fig.add_trace(scatter, row=1, col=1)

        line = go.Scatter(
            x=selected_stock_df.index,
            y=selected_stock_df,
            mode='lines',
            name='Stock Price'
        )
        fig.add_trace(line, row=2, col=1)

        fig.update_layout(title_text='RS vs RV and Stock Price', height=600)
        st.plotly_chart(fig)
        
        
        selected_stock_returns = stock_returns[selected_symbol]
        
        plot_single_line(
            selected_stock_returns,
            title='Selected Symbols Returns',
            x_title='Date',
            y_title='Returns'
        )