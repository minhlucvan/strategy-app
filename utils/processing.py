import datetime
from functools import lru_cache
from typing import List
import pandas as pd
import requests
import streamlit as st

from utils.akdata import AKData
    
@st.cache_data
def get_stocks(symbolsDate_dict: dict, column='close', stack=False, stack_level='factor', timeframe='D', volume_filter=1000):
    datas = AKData(symbolsDate_dict['market'])
    stocks_dfs = {}
    for symbol in symbolsDate_dict['symbols']:
        if symbol != '':
            stock_df = datas.get_stock(
                symbol, symbolsDate_dict['start_date'], symbolsDate_dict['end_date'], timeframe)
            if stock_df.empty:
                print(
                    f"Warning: stock '{symbol}' is invalid or missing. Ignore it")
            else:
                mean_vol = stock_df['volume'].rolling(window=20).mean().iloc[-1]
                if volume_filter is not None and mean_vol < volume_filter:
                    print(
                        f"Warning: stock '{symbol}' has low volume. Ignore it")
                else:
                    stock_df['value'] = stock_df['close'] * stock_df['volume']
                    stock_df['price_change'] = stock_df['close'].pct_change()
                    stock_df['volume_change'] = stock_df['volume'].pct_change()
                    stock_df['volume_weighted'] = stock_df['price_change'] * stock_df['volume']
                    
                    stock_df['price_change_weighted'] = stock_df['price_change'] * abs(stock_df['volume_change'])
                    
                    stock_df['value_change_weighted'] = stock_df['price_change'] * stock_df['volume_change']
                    
                    stocks_dfs[symbol] = stock_df if stack else stock_df[column]
                    
                    stocks_dfs[symbol].index = stocks_dfs[symbol].index.tz_localize(None)
    
    stocks_df = pd.DataFrame()
    if stack and stack_level == 'factor':
        # each frame represents one stock with all factors open, close, high, low, volume
        # stack the dataframes is 2 levels
        # level 0 is the factor name, level 1 is the stock symbol
        # eg. stocks_df['close']['AAPL'] = 12.3
        factor_dfs = {}
        for symbol in stocks_dfs:
            for column in stocks_dfs[symbol].columns:
                if column not in factor_dfs:
                    factor_dfs[column] = pd.DataFrame()
                factor_dfs[column][symbol] = stocks_dfs[symbol][column]
        
        stocks_df = pd.concat(factor_dfs, axis=1)
        
        # drop date column
        stocks_df = stocks_df.drop(columns='date')
    elif stack:
        stocks_df = pd.concat(stocks_dfs, axis=1)
    else:
        stocks_df = pd.DataFrame(stocks_dfs)
                
    return stocks_df

@st.cache_data
def get_stocks_funamental(symbolsDate_dict: dict, column='close',  stack=False, stack_level='factor'):
    datas = AKData(symbolsDate_dict['market'])
    stocks_dfs = {}
    for symbol in symbolsDate_dict['symbols']:
        if symbol != '':
            stock_df = datas.get_fundamental(symbol)
            if stock_df.empty:
                print(
                    f"Warning: stock '{symbol}' is invalid or missing. Ignore it")
            else:
                stocks_dfs[symbol] = stock_df if stack else stock_df[column]
    
    stocks_df = pd.DataFrame()
    if stack and stack_level == 'factor':
        # each frame represents one stock with all factors
        # stack the dataframes is 2 levels
        # level 0 is the factor name, level 1 is the stock symbol
        # eg. stocks_df['pe']['AAPL'] = 12.3
        factor_dfs = {}
        for symbol in stocks_dfs:
            for column in stocks_dfs[symbol].columns:
                if column not in factor_dfs:
                    factor_dfs[column] = pd.DataFrame()
                factor_dfs[column][symbol] = stocks_dfs[symbol][column]
        stocks_df = pd.concat(factor_dfs, axis=1)
    elif stack:
        stocks_df = pd.concat(stocks_dfs, axis=1)
    else:
        stocks_df = pd.DataFrame(stocks_dfs)
        
    return stocks_df

@st.cache_data
def get_stocks_financial(symbolsDate_dict: dict, column=None,  stack=False, stack_level='factor', raw=False):
    datas = AKData(symbolsDate_dict['market'])
    stocks_dfs = {}
    
    for symbol in symbolsDate_dict['symbols']:
        if symbol != '':
            stock_df = datas.get_financial(symbol, symbolsDate_dict['start_date'], symbolsDate_dict['end_date'])
            if stock_df.empty:
                print(
                    f"Warning: stock '{symbol}' is invalid or missing. Ignore it")
            elif stack and stack_level == 'factor' and column is not None:
                stocks_dfs[symbol] = stock_df[column]
                stocks_dfs[symbol] = stock_df
            else:
                stocks_dfs[symbol] = stock_df
    
    if raw:
        return stocks_dfs
    
    stocks_df = pd.DataFrame()
    
    if stack and stack_level == 'factor':
        # each frame represents one stock with all factors
        # stack the dataframes is 2 levels
        # level 0 is the factor name, level 1 is the stock symbol
        # eg. stocks_df['pe']['AAPL'] = 12.3
        factor_dfs = {}
        for symbol in stocks_dfs:
            for column in stocks_dfs[symbol].columns:
                if column not in factor_dfs:
                    factor_dfs[column] = pd.DataFrame()
                factor_dfs[column][symbol] = stocks_dfs[symbol][column]
        stocks_df = pd.concat(factor_dfs, axis=1)
    elif stack:
        stocks_df = pd.concat(stocks_dfs, axis=1)
    else:
        for symbol in stocks_dfs:
            stock_df = stocks_dfs[symbol]
            stock_df['symbol'] = symbol
            stocks_df = pd.concat([stocks_df, stock_df])
            
    return stocks_df

st.cache_data
def get_stocks_events(symbolsDate_dict: dict, column='label',  stack=False, stack_level='factor'):
    print(f"get_stocks_events: {symbolsDate_dict} {column} {stack} {stack_level}")
    datas = AKData(symbolsDate_dict['market'])
    stocks_dfs = {}
    for symbol in symbolsDate_dict['symbols']:
        if symbol != '':
            stock_df = datas.get_events(symbol, symbolsDate_dict['start_date'], symbolsDate_dict['end_date'])
            if stock_df.empty:
                print(
                    f"Warning: stock '{symbol}' is invalid or missing. Ignore it")
            else:
                stocks_dfs[symbol] = stock_df if stack else stock_df[column]
    
    stocks_df = pd.DataFrame()
    if stack and stack_level == 'factor':
        # each frame represents one stock with all factors
        # stack the dataframes is 2 levels
        # level 0 is the factor name, level 1 is the stock symbol
        # eg. stocks_df['pe']['AAPL'] = 12.3
        factor_dfs = {}
        for symbol in stocks_dfs.keys():
            for column in stocks_dfs[symbol].columns:
                if column not in factor_dfs:
                    factor_dfs[column] = pd.DataFrame()
                factor_dfs[column][symbol] = stocks_dfs[symbol][column]
        stocks_df = pd.concat(factor_dfs, axis=1)
    elif stack:
        stocks_df = pd.concat(stocks_dfs, axis=1)
    else:
        stocks_df = pd.DataFrame(stocks_dfs)
                
    return stocks_df

@st.cache_data
def get_stocks_news(symbolsDate_dict: dict, column='title',  stack=False, stack_level='factor', channel_id='-1'):
    datas = AKData(symbolsDate_dict['market'])
    stocks_dfs = {}
    for symbol in symbolsDate_dict['symbols']:
        if symbol != '':
            stock_df = datas.get_news(symbol, symbolsDate_dict['start_date'], symbolsDate_dict['end_date'], channel_id=channel_id)
            if stock_df.empty:
                print(
                    f"Warning: stock '{symbol}' is invalid or missing. Ignore it")
            else:
                stock_df = stock_df.groupby(stock_df.index).agg(lambda x: ', '.join(x))
                stock_df = stock_df[~stock_df.index.duplicated()]
                
                stocks_dfs[symbol] = stock_df if stack else stock_df[column]
    
    stocks_df = pd.DataFrame()
    if stack and stack_level == 'factor':
        # each frame represents one stock with all factors
        # stack the dataframes is 2 levels
        # level 0 is the factor name, level 1 is the stock symbol
        # eg. stocks_df['pe']['AAPL'] = 12.3
        factor_dfs = {}
        for symbol in stocks_dfs.keys():
            for column in stocks_dfs[symbol].columns:
                if column not in factor_dfs:
                    factor_dfs[column] = pd.DataFrame()
                factor_dfs[column][symbol] = stocks_dfs[symbol][column]
        stocks_df = pd.concat(factor_dfs, axis=1)
    elif stack:
        stocks_df = pd.concat(stocks_dfs, axis=1)
    else:
        # deduplicate
        for symbol in stocks_dfs:
            stock_df = stocks_dfs[symbol]
            stocks_dfs[symbol] = stock_df
            
        stocks_df = pd.DataFrame(stocks_dfs)
                
    return stocks_df

@st.cache_data
def get_stocks_foregin_flow(symbolsDate_dict: dict, column='close',  stack=False, stack_level='factor'):
    datas = AKData(symbolsDate_dict['market'])
    stocks_dfs = {}
    for symbol in symbolsDate_dict['symbols']:
        if symbol != '':
            stock_df = datas.get_stock_foregin_flow(symbol, symbolsDate_dict['start_date'], symbolsDate_dict['end_date'])
            if stock_df.empty:
                print(
                    f"Warning: stock '{symbol}' is invalid or missing. Ignore it")
            else:
                stocks_dfs[symbol] = stock_df if stack else stock_df[column]
    
    stocks_df = pd.DataFrame()
    if stack and stack_level == 'factor':
        # each frame represents one stock with all factors
        # stack the dataframes is 2 levels
        # level 0 is the factor name, level 1 is the stock symbol
        # eg. stocks_df['pe']['AAPL'] = 12.3
        factor_dfs = {}
        for symbol in stocks_dfs.keys():
            for column in stocks_dfs[symbol].columns:
                if column not in factor_dfs:
                    factor_dfs[column] = pd.DataFrame()
                factor_dfs[column][symbol] = stocks_dfs[symbol][column]
        stocks_df = pd.concat(factor_dfs, axis=1)
    elif stack:
        stocks_df = pd.concat(stocks_dfs, axis=1)
    else:
        stocks_df = pd.DataFrame(stocks_dfs)
                
    return stocks_df

@st.cache_data
def get_stocks_document(symbolsDate_dict: dict, column='title', doc_type='1', stack=False, stack_level='factor', group_by_date=False):
    datas = AKData(symbolsDate_dict['market'])
    stocks_dfs = {}
    for symbol in symbolsDate_dict['symbols']:
        if symbol != '':
            stock_df = datas.get_document(symbol, symbolsDate_dict['start_date'], symbolsDate_dict['end_date'], doc_type=doc_type)
            if stock_df.empty:
                print(
                    f"Warning: stock '{symbol}' is invalid or missing. Ignore it")
            else:
                stocks_dfs[symbol] = stock_df if stack else stock_df[column]
                if group_by_date:
                    stocks_dfs[symbol].index = pd.to_datetime(stocks_dfs[symbol].index.date)
                    stocks_dfs[symbol] = stocks_dfs[symbol].groupby(stocks_dfs[symbol].index).agg(lambda x: ', '.join(x))
                    stocks_dfs[symbol] = stocks_dfs[symbol][~stocks_dfs[symbol].index.duplicated()]
                    
                stocks_dfs[symbol].index = stocks_dfs[symbol].index.tz_localize(None)
                    
    
    stocks_df = pd.DataFrame()
    if stack and stack_level == 'factor':
        # each frame represents one stock with all factors
        # stack the dataframes is 2 levels
        # level 0 is the factor name, level 1 is the stock symbol
        # eg. stocks_df['pe']['AAPL'] = 12.3
        factor_dfs = {}
        for symbol in stocks_dfs.keys():
            for column in stocks_dfs[symbol].columns:
                if column not in factor_dfs:
                    factor_dfs[column] = pd.DataFrame()
                factor_dfs[column][symbol] = stocks_dfs[symbol][column]
        stocks_df = pd.concat(factor_dfs, axis=1)
    elif stack:
        stocks_df = pd.concat(stocks_dfs, axis=1)
    else:
        stocks_df = pd.DataFrame(stocks_dfs)
                
    return stocks_df
    

@st.cache_data
def get_stocks_valuation(symbolsDate_dict: dict, indicator='pe'):
    datas = AKData(symbolsDate_dict['market'])
    stocks_df = pd.DataFrame()
    for symbol in symbolsDate_dict['symbols']:
        if symbol != '':
            stock_df = datas.get_valuation(symbol, indicator)
            if stock_df.empty:
                print(
                    f"Warning: stock '{symbol}' is invalid or missing. Ignore it")
            else:
                stocks_df[symbol] = stock_df[indicator]
                
    return stocks_df

