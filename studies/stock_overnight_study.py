
import streamlit as st
import pandas as pd
import plotly.express as px

from utils.plot_utils import plot_multi_bar, plot_multi_line, plot_single_bar, plot_single_line, plot_multi_scatter
from utils.processing import get_stocks, get_stocks_foregin_flow


def plot_scatter_2_sources(df1, df2, title, x_title, y_title, legend_title):
    data = pd.DataFrame({
        'Ticker': df1.columns.repeat(len(df1)),
        'Price Gap': df1.values.flatten(),
        'Price Change after 3 Days': df2.values.flatten(),
    })

    # Plot using Plotly Express scatter plot
    fig = px.scatter(data, x='Price Gap', y='Price Change after 3 Days', color='Ticker', 
                    title='Correlation between Price Gaps and Changes after 3 Days',
                    labels={'Price Gap': 'Price Gap (%)', 'Price Change after 3 Days': 'Price Change after 3 Days (%)'})

    # Customize layout (optional)
    fig.update_layout(
        xaxis_title='Price Gap (%)',
        yaxis_title='Price Change after 3 Days (%)',
        title='Correlation between Price Gaps and Changes after 3 Days'
    )

    # Show the plot
    st.plotly_chart(fig, use_container_width=True)

    
def run(symbol_benchmark, symbolsDate_dict):
    
    if len(symbolsDate_dict['symbols']) < 1:
        st.info("Please select symbols.")
        st.stop()
    
    # copy the symbolsDate_dict
    benchmark_dict = symbolsDate_dict.copy()
    benchmark_dict['symbols'] = [symbol_benchmark]
    
    benchmark_df = get_stocks(benchmark_dict,'close')
    closes_df = get_stocks(symbolsDate_dict,'close')
    opens_df = get_stocks(symbolsDate_dict,'open')
    
    gaps_df = (opens_df - closes_df.shift(1)) / closes_df.shift(1)
    
    opens_future_df = opens_df.shift(-3)
    
    opens_change_df = (opens_future_df - opens_df) / opens_df
    
    # filter the gaps_df > 0.02 or < -0.02
    gaps_df = gaps_df[(gaps_df > 0.07) | (gaps_df < -0.07)]
    
    ratio_df = opens_change_df / gaps_df
    
    
    
    plot_multi_bar(gaps_df, title='Stocks Gaps', x_title='Date', y_title='Gap', legend_title='Stocks')
    
    plot_multi_line(closes_df, title='Stocks Prices', x_title='Date', y_title='Price', legend_title='Stocks')
    
    plot_multi_bar(ratio_df, title='Open Change vs Gap', x_title='Date', y_title='Ratio', legend_title='Stocks')
    
    # gaps_down_df = gaps_df[gaps_df < -0.07]

    # gaps_df = gaps_down_df

    # Prepare data for plotting
    # Flatten the DataFrames to make them suitable for Plotly

    plot_scatter_2_sources(gaps_df, opens_change_df, title='Correlation between Price Gaps and Changes after 3 Days', x_title='Price Gap', y_title='Price Change after 3 Days', legend_title='Ticker')