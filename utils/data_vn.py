import datetime
import pytz
import requests
import pandas as pd
from functools import lru_cache

import akshare as ak
import streamlit as st
import vectorbt as vbt
import vnquant as vnquant

from utils.db import load_symbol
from utils import stock_utils
from utils import vietstock
from utils.stock_utils import get_stock_bars_very_long_term_cached, get_stock_balance_sheet, load_stock_balance_sheet_to_dataframe

def get_intervals():
    return ['D', 'W', 'M', '60', '30', '15', '5', '1']

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
    finance = stock_utils.get_stock_financial(symbol)
    finance_df = stock_utils.load_stock_financial_to_dataframe(finance)
    
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
    evaluation = stock_utils.get_stock_evaluation(symbol)
    evaluation_df = stock_utils.load_stock_evaluation_to_dataframe(evaluation)
    
    evaluation_df['value'] = evaluation_df['pe']

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