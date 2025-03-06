import datetime
import pytz
import requests
import pandas as pd
from functools import lru_cache

import akshare as ak
import streamlit as st
import vectorbt as vbt

from utils.db import load_symbol
from utils import stock_utils
from utils import vietstock
import os
from utils.stock_utils import get_stock_bars_very_long_term_cached, get_stock_balance_sheet, load_stock_balance_sheet_to_dataframe

def get_intervals():
    return ['D', '3D', 'W', 'M', '60', '30', '15', '5', '1']

@lru_cache
def get_vn_stock(symbol: str, start_date: str, end_date: str, timeframe='1D') -> pd.DataFrame:
    """get vietnam stock data

    Args:
        ak_params symbol:str, start_date:str 20170301, end_date:str

    Returns:
        pd.DataFrame: _description_
    """
    print(f"get_vn_stock: {symbol}")
    
    # 20180101 ->  '%Y-%m-%d'
    if len(start_date) == 8:
        start_date = start_date[0:4] + '-' + start_date[4:6] + '-' + start_date[6:]
    
    if len(end_date) == 8:
        end_date = end_date[0:4] + '-' + end_date[4:6] + '-' + end_date[6:]

    stock_df = get_stock_bars_very_long_term_cached(
        ticker=symbol,
        stock_type='stock',
        count_back=300,
        resolution=timeframe,
        start_date=start_date,
        end_date=end_date,
        refresh=True,
        force_fetch=False
    )
        
    stock_df['volume'] = stock_df['volume']
    stock_df['date'] = stock_df.index
    
    # sort by index
    stock_df = stock_df.sort_index()
    
    stock_df = stock_df[['date', 'open', 'close', 'high', 'low', 'volume']]
    
    return stock_df   

@lru_cache
def get_vn_index(symbol: str, start_date: str, end_date: str, timeframe='D') -> pd.DataFrame:
    """get vietnam stock data

    Args:
        ak_params symbol:str, start_date:str 20170301, end_date:str

    Returns:
        pd.DataFrame: _description_
    """
    
    # 20180101 ->  '%Y-%m-%d'
    if len(start_date) == 8:
        start_date = start_date[0:4] + '-' + start_date[4:6] + '-' + start_date[6:]
    
    if len(end_date) == 8:
        end_date = end_date[0:4] + '-' + end_date[4:6] + '-' + end_date[6:]

    stock_df = get_stock_bars_very_long_term_cached(
        ticker=symbol,
        stock_type='index',
        count_back=300,
        resolution=timeframe,
        start_date=start_date,
        end_date=end_date,
    )

    stock_df['volume'] = stock_df['volume']
    stock_df['date'] = stock_df.index
    
    # sort by index
    stock_df = stock_df.sort_index()
    
    stock_df = stock_df[['date', 'open', 'close', 'high', 'low', 'volume']]
    
    return stock_df    

@lru_cache
def get_vn_warrant(symbol: str, start_date: str, end_date: str, timeframe='D') -> pd.DataFrame:
    """get vietnam stock data

    Args:
        ak_params symbol:str, start_date:str 20170301, end_date:str

    Returns:
        pd.DataFrame: _description_
    """
    
    # 20180101 ->  '%Y-%m-%d'
    if len(start_date) == 8:
        start_date = start_date[0:4] + '-' + start_date[4:6] + '-' + start_date[6:]
    
    if len(end_date) == 8:
        end_date = end_date[0:4] + '-' + end_date[4:6] + '-' + end_date[6:]

    stock_df = get_stock_bars_very_long_term_cached(
        ticker=symbol,
        stock_type='coveredWarr',
        count_back=300,
        resolution=timeframe,
        start_date=start_date,
        end_date=end_date,
        refresh=True,
        force_fetch=True
    )

    stock_df['volume'] = stock_df['volume']
    stock_df['date'] = stock_df.index
    
    # sort by index
    stock_df = stock_df.sort_index()
    
    stock_df = stock_df[['date', 'open', 'close', 'high', 'low', 'volume']]
    
    return stock_df

@lru_cache
def get_vn_derivative(symbol: str, start_date: str, end_date: str, timeframe='D') -> pd.DataFrame:
    """get vietnam stock data

    Args:
        ak_params symbol:str, start_date:str 20170301, end_date:str

    Returns:
        pd.DataFrame: _description_
    """
    
    # 20180101 ->  '%Y-%m-%d'
    if len(start_date) == 8:
        start_date = start_date[0:4] + '-' + start_date[4:6] + '-' + start_date[6:]
    
    if len(end_date) == 8:
        end_date = end_date[0:4] + '-' + end_date[4:6] + '-' + end_date[6:]

    stock_df = get_stock_bars_very_long_term_cached(
        ticker=symbol,
        stock_type='derivative',
        count_back=300,
        resolution=timeframe,
        start_date=start_date,
        end_date=end_date,
        refresh=True,
        force_fetch=True
    )

    stock_df['volume'] = stock_df['volume']
    stock_df['date'] = stock_df.index
    
    # sort by index
    stock_df = stock_df.sort_index()
    
    stock_df = stock_df[['date', 'open', 'close', 'high', 'low', 'volume']]
    
    return stock_df

@lru_cache
def get_vn_etf(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """get vietnam etf data

    Args:
        ak_params symbol:str, start_date:str 20170301, end_date:str

    Returns:
        pd.DataFrame: _description_
    """
    import vnquant as vnquant
    
    # 20180101 ->  '%Y-%m-%d'
    if len(start_date) == 8:
        start_date = start_date[0:4] + '-' + start_date[4:6] + '-' + start_date[6:]
    
    if len(end_date) == 8:
        end_date = end_date[0:4] + '-' + end_date[4:6] + '-' + end_date[6:]

    loader = vnquant.data.DataLoader(
        symbols=symbol,
        start=start_date,
        end=end_date,
        data_source='CAFE',
        minimal=True,
        table_style='stack')

    stock_df = loader.download()

    stock_df['volume'] = stock_df['volume_match']
    stock_df['date'] = stock_df.index
    
    # covert to date
    # stock_df['date'] = stock_df['date'].
    
    st.write(stock_df)
    
    # sort by index
    stock_df = stock_df.sort_index()
    
    stock_df = stock_df[['date', 'open', 'close', 'high', 'low', 'volume']]
    return stock_df    

@lru_cache
def get_vn_fundamental(symbol: str) -> pd.DataFrame:
    """get vietnam stock pe data乐咕乐股-A 股个股指标: 市盈率, 市净率, 股息率

    """
    data = get_stock_balance_sheet(symbol)
    
    data_df = load_stock_balance_sheet_to_dataframe(data)
    
    # drop columns
    data_df = data_df.drop(columns=['ticker', 'quarter', 'year', 'date', 'start_date'])
        
    return data_df

@lru_cache
def get_vn_financial(symbol: str) -> pd.DataFrame:
    """get vietnam stock data

    Args:
        ak_params symbol:str, start_date:str 20170301, end_date:str

    Returns:
        pd.DataFrame: _description_
    """
    print(f"get_vn_financial: {symbol}")
    
    # Define cache file path
    cache_file = f'data/financials/financial_{symbol}.csv'
    
    try:
        # Try to read from cache
        finance_df = pd.read_csv(cache_file, index_col=0)
        print(f"Loaded {symbol} financial data from cache")
    except FileNotFoundError:
        # If cache does not exist, fetch data and save to cache
        finance = stock_utils.get_stock_financial(symbol)
        finance_df = stock_utils.load_stock_financial_to_dataframe(finance)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        
        finance_df.to_csv(cache_file)
        print(f"Saved {symbol} financial data to cache")
    
    # drop columns
    finance_df = finance_df.drop(columns=['ticker', 'quarter', 'year', 'date'])
    
    return finance_df

@lru_cache
def get_vn_valuation(symbol: str, indicator: str) -> pd.DataFrame:
    """get vietnam stock data

    Args:
        ak_params symbol:str, start_date:str 20170301, end_date:str

    Returns:
        pd.DataFrame: _description_
    """
    print(f"get_vn_valuation: {symbol}, {indicator}")
    
    # Define cache file path
    cache_file = f'data/valuation/valuation_{symbol}_{indicator}.csv'
    
    try:
        # Try to read from cache
        evaluation_df = pd.read_csv(cache_file, index_col=0)
        print(f"Loaded {symbol} valuation data from cache")
    except FileNotFoundError:
        # If cache does not exist, fetch data and save to cache
        evaluation = stock_utils.get_stock_evaluation(symbol)
        evaluation_df = stock_utils.load_stock_evaluation_to_dataframe(evaluation)
        
        evaluation_df['value'] = evaluation_df[indicator]
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        
        evaluation_df.to_csv(cache_file)
        print(f"Saved {symbol} valuation data to cache")
    
    return evaluation_df

@lru_cache
def get_vn_events(symbol: str, start_date: datetime.datetime, end_date: datetime.datetime) -> pd.DataFrame:
    """get vietnam stock data

    Args:
        ak_params symbol:str, start_date:str 20170301, end_date:str

    Returns:
        pd.DataFrame: _description_
    """
    print(f"get_vn_events: {symbol}")
    events = stock_utils.get_stock_events(symbol, start_date, end_date)
    events_df = stock_utils.load_stock_events_to_dataframe(events)
    
    return events_df

@lru_cache
def get_vn_foregin_flow(symbol: str, start_date: datetime.datetime, end_date: datetime.datetime) -> pd.DataFrame:
    """get vietnam stock data

    Args:
        ak_params symbol:str, start_date:str 20170301, end_date:str

    Returns:
        pd.DataFrame: _description_
    """
    print(f"get_vn_foregin_flow: {symbol}")
    foregin_flow = stock_utils.get_stock_vol_foreign(symbol)
    foregin_flow_df = stock_utils.load_stock_vol_foreign_to_dataframe(foregin_flow)

    # foregin_flow_df = foregin_flow_df[foregin_flow_df.index >= start_date]
    # foregin_flow_df = foregin_flow_df[foregin_flow_df.index <= end_date]
    
    # st.write(foregin_flow_df)
    
    return foregin_flow_df

@lru_cache
def get_vn_news(symbol: str, start_date: datetime.datetime, end_date: datetime.datetime, channel_id='-1') -> pd.DataFrame:
    """get vietnam stock data

    Args:
        ak_params symbol:str, start_date:str 20170301, end_date:str

    Returns:
        pd.DataFrame: _description_
    """
    print(f"get_vn_news: {symbol} {start_date} {end_date}")
    news_df = vietstock.get_stock_news_all(symbol, start_date, end_date, channel_id=channel_id)
    
    return news_df

@lru_cache
def get_vn_document(symbol: str, start_date: datetime.datetime = None, end_date: datetime.datetime=None, doc_type='1') -> pd.DataFrame:
    """get vietnam stock data

    Args:
        ak_params symbol:str, start_date:str 20170301, end_date:str

    Returns:
        pd.DataFrame: _description_
    """
    print(f"get_vn_document: {symbol} {start_date} {end_date}")
    docs = vietstock.get_stock_documents(symbol, doc_type=doc_type)
    
    docs_df = vietstock.load_stock_documents_to_df(docs)
    
    return docs_df

@lru_cache
def get_vn_stock_info(symbol: str) -> pd.DataFrame:
    """get vietnam stock data

    Args:
        ak_params symbol:str, start_date:str 20170301, end_date:str

    Returns:
        pd.DataFrame: _description_
    """
    print(f"get_vn_stock_info: {symbol}")
    stock_info = stock_utils.get_stock_info_data(tickers=[symbol])
    stock_info_df = stock_utils.load_cw_info_to_dataframe(stock_info)
    
    return stock_info_df

@lru_cache
def get_vn_income_statement(symbol: str) -> pd.DataFrame:
    """get vietnam stock data

    Args:
        ak_params symbol:str, start_date:str 20170301, end_date:str

    Returns:
        pd.DataFrame: _description_
    """
    print(f"get_vn_income_statement: {symbol}")
    
    # Define cache file path
    cache_file = f'data/incomes/income_{symbol}.csv'

    try:
        # Try to read from cache
        income_statement_df = pd.read_csv(cache_file, index_col=0)
        print(f"Loaded {symbol} income statement from cache")
    except FileNotFoundError:
        # If cache does not exist, fetch data and save to cache
        income_statement = stock_utils.get_stock_income_statement(symbol)
        income_statement_df = stock_utils.load_stock_income_statement_to_dataframe(income_statement)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        
        income_statement_df.to_csv(cache_file)
        print(f"Saved {symbol} income statement to cache")
    
    return income_statement_df