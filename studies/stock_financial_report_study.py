from utils.processing import get_stocks, get_stocks_financial
from utils.plot_utils import plot_multi_line, plot_multi_bar
import streamlit as st
import pandas as pd

def calculate_real_ratios(stocks_df, financials_dfs):
    eps_df = pd.DataFrame()
    pe_df = pd.DataFrame()
    pb_df = pd.DataFrame()
    real_pe_df = pd.DataFrame()
    real_pb_df = pd.DataFrame()

    for symbol in financials_dfs:
        financials_df = financials_dfs[symbol].copy()
        financials_df.index = pd.to_datetime(financials_df.index).tz_localize(None)
        financials_df = financials_df[financials_df.index >= stocks_df.index[0]]

        if financials_df.empty:
            continue
        
        if symbol not in stocks_df.columns:
            continue
        
        stock_df = stocks_df[symbol]
        union_df = financials_df.reindex(stock_df.index, method='ffill')
        
        union_df['close'] = stock_df.astype(float)
        union_df['realPriceToEarning'] = union_df['close'] / union_df['earningPerShare']
        union_df['realPriceToBook'] = union_df['close'] / union_df['bookValuePerShare']

        eps_df = pd.concat([eps_df, pd.DataFrame({symbol: union_df['earningPerShare']})], axis=1)
        pe_df = pd.concat([pe_df, pd.DataFrame({symbol: union_df['priceToEarning']})], axis=1)
        pb_df = pd.concat([pb_df, pd.DataFrame({symbol: union_df['priceToBook']})], axis=1)
        real_pe_df = pd.concat([real_pe_df, pd.DataFrame({symbol: union_df['realPriceToEarning']})], axis=1)
        real_pb_df = pd.concat([real_pb_df, pd.DataFrame({symbol: union_df['realPriceToBook']})], axis=1)

    return eps_df, pe_df, pb_df, real_pe_df, real_pb_df

def run(symbol_benchmark, symbolsDate_dict):
    if len(symbolsDate_dict['symbols']) < 1:
        st.info("Please select symbols.")
        st.stop()

    benchmark_dict = symbolsDate_dict.copy()
    benchmark_dict['symbols'] = [symbol_benchmark]

    stocks_df = get_stocks(symbolsDate_dict, 'close')
    stocks_df.index = pd.to_datetime(stocks_df.index).tz_localize(None)

    financials_dfs = get_stocks_financial(symbolsDate_dict, raw=True)

    eps_df, pe_df, pb_df, real_pe_df, real_pb_df, real_pe_change_df = calculate_real_ratios(stocks_df, financials_dfs)

    plot_multi_line(eps_df, title='Earning Per Share', x_title='Date', y_title='Earning Per Share', legend_title='Stocks')
    plot_multi_line(pe_df, title='Price to Earning', x_title='Date', y_title='Price to Earning', legend_title='Stocks')
    plot_multi_line(real_pe_df, title='Real Price to Earning', x_title='Date', y_title='Real Price to Earning', legend_title='Stocks')
    plot_multi_line(pb_df, title='Price to Book', x_title='Date', y_title='Price to Book', legend_title='Stocks')
    plot_multi_line(real_pb_df, title='Real Price to Book', x_title='Date', y_title='Real Price to Book', legend_title='Stocks')
    plot_multi_bar(real_pe_change_df, title='Real Price to Earning Change', x_title='Date', y_title='Real Price to Earning Change', legend_title='Stocks', price_df=stocks_df)
