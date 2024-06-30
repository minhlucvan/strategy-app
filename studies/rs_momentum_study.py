import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objs as go
from plotly.subplots import make_subplots

from utils.plot_utils import plot_multi_bar, plot_multi_line, plot_single_bar
from utils.processing import get_stocks
import vectorbt as vbt



def plot_AnimatedRSVIX(rs_df, vix_df, symbols, tail_length):
    # Determine dynamic ranges for the axes
    x_min = vix_df.min().min()
    x_max = vix_df.max().max()
    y_min = rs_df.min().min()
    y_max = rs_df.max().max()

    # Create the frame figure
    fig_dict = {
        "data": [],
        "layout": {},
        "frames": []
    }

    # Fill in most of layout
    fig_dict["layout"]["xaxis"] = {"range": [x_min, x_max], "title": "VIX", "showgrid": True}
    fig_dict["layout"]["yaxis"] = {"range": [y_min, y_max], "title": "RS", "showgrid": True}
    fig_dict["layout"]["hovermode"] = "closest"
    fig_dict["layout"]["width"] = 1000
    fig_dict["layout"]["height"] = 600
    fig_dict["layout"]["updatemenus"] = [
        {
            "buttons": [
                {
                    "args": [None, {"frame": {"duration": 100, "redraw": False},
                                    "fromcurrent": True, "transition": {"duration": 300,
                                                                        "easing": "quadratic-in-out"}}],
                    "label": "Play",
                    "method": "animate"
                },
                {
                    "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate",
                                    "transition": {"duration": 0}}],
                    "label": "Pause",
                    "method": "animate"
                }
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 87},
            "showactive": False,
            "type": "buttons",
            "x": 0.1,
            "xanchor": "right",
            "y": 0,
            "yanchor": "top"
        }
    ]

    sliders_dict = {
        "active": 0,
        "yanchor": "top",
        "xanchor": "left",
        "currentvalue": {
            "font": {"size": 20},
            "prefix": "Date:",
            "visible": True,
            "xanchor": "right"
        },
        "transition": {"duration": 300, "easing": "cubic-in-out"},
        "pad": {"b": 10, "t": 50},
        "len": 0.9,
        "x": 0.1,
        "y": 0,
        "steps": []
    }

    # Make data
    for symbol in symbols:
        data_dict = {
            "x": list(vix_df[symbol][-tail_length:]),
            "y": list(rs_df[symbol][-tail_length:]),
            "mode": "lines+markers+text",
            "marker": {
                'symbol': "circle-open-dot",
                'size': 6,
            },
            "line": {
                'width': 4,
            },
            "name": symbol,
            "hovertemplate": '<b>%{hovertext}</b>',
            "hovertext": [d.strftime("%Y-%m-%d") for d in rs_df.index[-tail_length:]]
        }
        fig_dict["data"].append(data_dict)
        data_dict = {
            "x": list(vix_df[symbol][-1:]),
            "y": list(rs_df[symbol][-1:]),
            "mode": "markers+text",
            "marker": {
                'symbol': "circle",
                'size': 12,
            },
            "text": symbol,
            "name": symbol,
            "hovertemplate": '<b>%{hovertext}</b>',
            "hovertext": [d.strftime("%Y-%m-%d") for d in rs_df.index[-1:]],
            "showlegend": False
        }
        fig_dict["data"].append(data_dict)

    # Make frames
    for i in range(len(rs_df) - tail_length + 1):
        d = rs_df.index[i].strftime("%Y-%m-%d")
        frame = {"data": [], "name": str(d)}
        for symbol in symbols:
            data_dict = {
                "x": list(vix_df[symbol][i: i + tail_length]),
                "y": list(rs_df[symbol][i: i + tail_length]),
                "mode": "lines+markers",
                "marker": {
                    'symbol': "circle-open-dot",
                    'size': 6,
                },
                "line": {
                    'width': 4,
                },
                "name": symbol,
                "hovertemplate": '<b>%{hovertext}</b>',
                "hovertext": [d.strftime("%Y-%m-%d") for d in rs_df.index[i: i + tail_length]]
            }
            frame["data"].append(data_dict)
            data_dict = {
                "x": list(vix_df[symbol][i + tail_length - 1: i + tail_length]),
                "y": list(rs_df[symbol][i + tail_length - 1: i + tail_length]),
                "mode": "markers+text",
                "marker": {
                    'symbol': "circle",
                    'size': 12,
                },
                "text": symbol,
                "name": symbol,
                "hovertemplate": '<b>%{hovertext}</b>',
                "hovertext": [d.strftime("%Y-%m-%d") for d in rs_df.index[i + tail_length - 1: i + tail_length]],
                "showlegend": False
            }
            frame["data"].append(data_dict)

        fig_dict["frames"].append(frame)
        slider_step = {"args": [
            [d],
            {"frame": {"duration": 300, "redraw": False},
             "mode": "immediate",
             "transition": {"duration": 300}}
        ],
            "label": d,
            "method": "animate"}
        sliders_dict["steps"].append(slider_step)

    fig_dict["layout"]["sliders"] = [sliders_dict]
    fig_dict["layout"]["title"] = 'RS vs VIX Animated Scatter Plot'

    fig = go.Figure(fig_dict)
    st.plotly_chart(fig, use_container_width=True)

def calculate_hist_volatility(prices, window=21):
    # calculate the log returns
    log_returns = np.log(prices / prices.shift(1))
    
    # calculate the variance
    variance = log_returns.rolling(window=window).std() ** 2
    
    # calculate the volatility
    volatility = variance.rolling(window=window).std()
    
    return volatility

# Calculate 30-day variance by interpolating the two variances,
# depending on the time to expiration of each. Take the square root to get volatility as standard deviation.
# Multiply the volatility (standard deviation) by 100. The result is the VIX index value.
# https://www.macroption.com/vix-calculation/#:~:text=VIX%20Calculation%20Step%20by%20Step,-Select%20the%20options&text=Calculate%2030%2Dday%20variance%20by,is%20the%20VIX%20index%20value.
def calculate_vix_index(prices):
    # calculate the log returns
    log_returns = np.log(prices / prices.shift(1))
    
    # calculate the variance
    variance = log_returns.rolling(window=21).std() ** 2
    
    # calculate the variance of the variance
    variance_of_variance = variance.rolling(window=2).std()
    
    # calculate the VIX index
    vix_index = 100 * variance_of_variance
    
    return vix_index

def calculate_rs(prices_df, market_df):
    """
    Calculate the relative strength (RS) of stocks compared to the market.
    
    Parameters:
    prices_df (pd.DataFrame): DataFrame containing stock prices with dates as the index.
    market_df (pd.DataFrame): DataFrame containing market index prices with dates as the index.
    
    Returns:
    pd.DataFrame: DataFrame with the relative strength of each stock compared to the market.
    """
    # Calculate the daily returns for the stocks and the market
    stock_returns = prices_df.pct_change()
    market_returns = market_df.pct_change()

    # Calculate the cumulative returns for the stocks and the market
    cumulative_stock_returns = (1 + stock_returns).cumprod() - 1
    cumulative_market_returns = (1 + market_returns).cumprod() - 1

    # Calculate the relative strength
    rs_df = cumulative_stock_returns.divide(cumulative_market_returns, axis=0)
    
    # rank the relative strength by axis=1
    rs_df = rs_df.rank(axis=1)
    
    return rs_df


# Example usage
def run(symbol_benchmark, symbolsDate_dict):
    
    if len(symbolsDate_dict['symbols']) < 1:
        st.info("Please select symbols.")
        st.stop()
    
    # Fetching stock data
    stocks_df = get_stocks(symbolsDate_dict, 'close')
    benchmark_df = get_stocks(symbolsDate_dict, 'close', benchmark=True)

    # Calculating RS
    rs_df = pd.DataFrame(index=stocks_df.index)
    for symbol in stocks_df.columns:
        r = stocks_df[symbol] / benchmark_df[symbol_benchmark]
        r_zscore = (r - r.mean()) / r.std()
        rs_df[symbol] = r_zscore

    # Calculating VIX
    vix_df = calculate_vix_index(stocks_df)
    
    vix_df = vix_df.rolling(5).mean()
    
    # calculate the rsi
    rsi_ind = vbt.RSI.run(stocks_df, window=14)

    rsi_df = rsi_ind.rsi[14]
    
    hv_df = calculate_hist_volatility(stocks_df)
    
    market_df = benchmark_df[symbol_benchmark]
    
    rs_df = calculate_rs(stocks_df, market_df)
        
    # Plotting 
    plot_AnimatedRSVIX(rsi_df, vix_df, stocks_df.columns, tail_length=3)

    plot_multi_line(rsi_df, title='Relative Strength')
    
    plot_multi_line(benchmark_df, title='Benchmark Prices')
   