from utils.processing import get_stocks, get_stocks_financial
from utils.plot_utils import plot_multi_line, plot_multi_bar
import streamlit as st
import pandas as pd
import utils.plot_utils as pu

def run(symbol_benchmark, symbolsDate_dict):
    if len(symbolsDate_dict['symbols']) < 1:
        st.info("Please select symbols.")
        st.stop()

    benchmark_dict = symbolsDate_dict.copy()
    benchmark_dict['symbols'] = [symbol_benchmark]

    stocks_df = get_stocks(symbolsDate_dict, 'close')
    stocks_df.index = pd.to_datetime(stocks_df.index).tz_localize(None)

    financials_df = get_stocks_financial(symbolsDate_dict, stack=True)

    roe_df = financials_df['roe']
    roa_df = financials_df['roa']
    earning_per_share_df = financials_df['earningPerShare']
    price_to_earning_df = financials_df['priceToEarning']
    price_to_book_df = financials_df['priceToBook']
    
    pu.plot_multi_line(stocks_df, title='price')
    
    pu.plot_multi_line(roe_df, title='Return on Equity', x_title='Date', y_title='ROE')
    pu.plot_multi_line(roa_df, title='Return on Asset', x_title='Date', y_title='ROA')
    pu.plot_multi_line(earning_per_share_df, title='Earning Per Share', x_title='Date', y_title='EPS')
    pu.plot_multi_line(price_to_earning_df, title='Price to Earning', x_title='Date', y_title='PE')
    pu.plot_multi_line(price_to_book_df, title='Price to Book', x_title='Date', y_title='PB')
    
    
    # realtime pe
    earning_per_share_fill_df = earning_per_share_df.reindex(stocks_df.index, method='ffill')
    pe_realtime_df = stocks_df / earning_per_share_fill_df

    pu.plot_multi_line(pe_realtime_df, title='RPE')

    # mean pe
    
    pe_mean_df = price_to_earning_df.mean(axis=1)
    pe_mean_df = pe_mean_df.to_frame(name='market_mean')
    
    pe_ratio_df = price_to_earning_df.div(pe_mean_df['market_mean'], axis=0)
        
    
    pu.plot_multi_line(pe_ratio_df, title='PE Ratio')