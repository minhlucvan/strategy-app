import streamlit as st
import pandas as pd
import numpy as np
import talib as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.processing import get_stocks
import utils.stock_utils as su
import utils.plot_utils as pu

def plot_price_with_markers(prices_df, markers_df, title, marker_name, symbol):
    """
    Creates a plot with stock prices and markers for a specific symbol.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=prices_df.index, y=prices_df['close'], name='Price', mode='lines'))
    
    for index, row in markers_df.iterrows():
        fig.add_trace(go.Scatter(x=[index], y=[prices_df.loc[index, 'close']], 
                                mode='markers', name=marker_name, 
                                marker=dict(size=10, color='red')))
    
    fig.update_layout(title=f"{title} - {symbol}")
    st.plotly_chart(fig)
    
def compute_pattern_accuracy(stock_df, patterns):
    accuracy_results = {}
    stock_df['next_day_return'] = stock_df['close'].pct_change(1).shift(-1) * 100
    
    for pattern in patterns:
        name = pattern['name']
        pattern_type = pattern['type']
        pattern_signals = stock_df[stock_df[name] != 0]
        
        if not pattern_signals.empty:
            if pattern_type == 'bullish':
                correct_signals = (pattern_signals['next_day_return'] > 0).sum()
            else:
                correct_signals = (pattern_signals['next_day_return'] < 0).sum()
            
            accuracy = correct_signals / len(pattern_signals) * 100
        else:
            accuracy = np.nan
        
        accuracy_results[name] = accuracy
    
    return accuracy_results

def compute_pattern_returns(stock_df, patterns):
    returns = {}
    stock_df['next_day_return'] = stock_df['close'].pct_change(1).shift(-1) * 100
    
    for pattern in patterns:
        name = pattern['name']
        pattern_signals = stock_df[stock_df[name] != 0]
        
        if not pattern_signals.empty:
            returns[name] = pattern_signals['next_day_return'].mean()
        else:
            returns[name] = np.nan
    
    return returns

def count_pattern_signals(stock_df, patterns):
    signal_counts = {}
    
    for pattern in patterns:
        name = pattern['name']
        signal_counts[name] = stock_df[name].abs().sum() / 100
    
    return signal_counts

def run(symbol_benchmark, symbolsDate_dict):
    """
    Main function to run stock analysis and visualization for multiple symbols.
    """
    if not symbolsDate_dict['symbols']:
        st.info("Please select symbols.")
        st.stop()
    
    # Define patterns with their TA-Lib functions and types
    patterns = [
        {'name': 'hammer', 'function': ta.CDLHAMMER, 'type': 'bullish'},
        {'name': 'hanging_man', 'function': ta.CDLHANGINGMAN, 'type': 'bearish'},
        {'name': 'bullish_engulfing', 'function': lambda o, h, l, c: (ta.CDLENGULFING(o, h, l, c) == 100).astype(int) * 100, 'type': 'bullish'},
        {'name': 'bearish_engulfing', 'function': lambda o, h, l, c: (ta.CDLENGULFING(o, h, l, c) == -100).astype(int) * 100, 'type': 'bearish'},
        {'name': 'three_white_soldiers', 'function': ta.CDL3WHITESOLDIERS, 'type': 'bullish'},
        {'name': 'three_black_crows', 'function': ta.CDL3BLACKCROWS, 'type': 'bearish'},
        {'name': 'morning_star', 'function': ta.CDLMORNINGSTAR, 'type': 'bullish'},
        {'name': 'evening_star', 'function': ta.CDLEVENINGSTAR, 'type': 'bearish'},
    ]
    
    # Get stocks data in stacked format
    stocks_df = get_stocks(symbolsDate_dict, stack=True)
    
    # Initialize dictionaries to store results for all symbols
    all_accuracy = {}
    all_returns = {}
    all_counts = {}
    
    # Process each symbol
    for symbol in symbolsDate_dict['symbols']:
        # Filter data for current symbol
        open_df = stocks_df['open'][symbol]
        high_df = stocks_df['high'][symbol]
        low_df = stocks_df['low'][symbol]
        close_df = stocks_df['close'][symbol]
        
        stock_df = pd.DataFrame({'close': close_df})
        
        # Calculate each pattern
        for pattern in patterns:
            stock_df[pattern['name']] = pattern['function'](
                open_df.values, high_df.values, low_df.values, close_df.values
            )
        
        # Plot each pattern for this symbol when len = 1
        if len(symbolsDate_dict['symbols']) == 1:
            for pattern in patterns:
                pattern_name = pattern['name'].replace('_', ' ').title()
                pattern_type = pattern['type'].capitalize()
                title = f"{pattern_name} Candlestick Pattern ({pattern_type})"
                plot_price_with_markers(
                    stock_df, 
                    stock_df[stock_df[pattern['name']] != 0], 
                    title, 
                    pattern_name,
                    symbol
                )

        # Compute metrics for this symbol
        all_accuracy[symbol] = compute_pattern_accuracy(stock_df, patterns)
        all_returns[symbol] = compute_pattern_returns(stock_df, patterns)
        all_counts[symbol] = count_pattern_signals(stock_df, patterns)
    
    # Display combined results
    st.write("## Combined Results Across All Symbols")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("### Accuracy")
        accuracy_df = pd.DataFrame(all_accuracy).T
        accuracy_df.columns = [col.replace('_', ' ').title() for col in accuracy_df.columns]
        st.dataframe(accuracy_df.style.format("{:.2f}%", na_rep="N/A"))
    
    with col2:
        st.write("### Returns")
        returns_df = pd.DataFrame(all_returns).T
        returns_df.columns = [col.replace('_', ' ').title() for col in returns_df.columns]
        st.dataframe(returns_df.style.format("{:.2f}%", na_rep="N/A"))
    
    with col3:
        st.write("### Counts")
        counts_df = pd.DataFrame(all_counts).T
        counts_df.columns = [col.replace('_', ' ').title() for col in counts_df.columns]
        st.dataframe(counts_df.astype(int))
        
    # Combine all symbols stats to analyze the overall performance
    st.write("## Overall Performance")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("### Accuracy")
        st.write(accuracy_df.mean())
    
    with col2:
        st.write("### Returns")
        st.write(returns_df.mean())
    
    with col3:
        st.write("### Counts")
        st.write(counts_df.sum())