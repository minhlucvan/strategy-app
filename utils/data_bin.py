from functools import lru_cache
from binance import Client, ThreadedWebsocketManager, ThreadedDepthCacheManager
from dotenv import load_dotenv
import pandas as pd
import os
import datetime
import utils.config as cfg
import requests 
import json 
import time 

import streamlit as st

apikey = cfg.get_config('binance.api_key')
apisecret = cfg.get_config('binance.api_secret')

client = Client(apikey, apisecret)


def get_all_tickers():
    tickers = client.get_all_tickers()
    tickers_df = pd.DataFrame(tickers)
    return tickers_df

def get_all_USDT_tickers():
    all_tickers = get_all_tickers()
    tickers_df = all_tickers[all_tickers['symbol'].str.endswith('USDT')]
    
    tickers_df['first'] = tickers_df['symbol'].str.split('USDT').str[0]
    tickers_df['second'] = tickers_df['symbol'].str.split('USDT').str[1]
    
    # drop first contains USD
    tickers_df = tickers_df[~tickers_df['first'].str.contains('USD')]    
    
    return tickers_df

def get_symbol_groups(name):
    if name == 'overview':
        return get_all_USDT_tickers()
    if name == 'overview10':
        return get_all_USDT_tickers()[:10]
    if name == 'overview100':
        return get_all_USDT_tickers()[:100]
    
    return []

def get_intervals():
    return ['1d', '1w', '1M', '12h', '6h', '4h', '2h', '1h', '30m', '15m', '5m', '3m', '1m']

def map_interval_to_enum(interval):
    interval_map = {
        '1m': Client.KLINE_INTERVAL_1MINUTE,
        '3m': Client.KLINE_INTERVAL_3MINUTE,
        '5m': Client.KLINE_INTERVAL_5MINUTE,
        '15m': Client.KLINE_INTERVAL_15MINUTE,
        '30m': Client.KLINE_INTERVAL_30MINUTE,
        '1h': Client.KLINE_INTERVAL_1HOUR,
        '2h': Client.KLINE_INTERVAL_2HOUR,
        '4h': Client.KLINE_INTERVAL_4HOUR,
        '6h': Client.KLINE_INTERVAL_6HOUR,
        '8h': Client.KLINE_INTERVAL_8HOUR,
        '12h': Client.KLINE_INTERVAL_12HOUR,
        '1d': Client.KLINE_INTERVAL_1DAY,
        '3d': Client.KLINE_INTERVAL_3DAY,
        '1w': Client.KLINE_INTERVAL_1WEEK,
        '1M': Client.KLINE_INTERVAL_1MONTH,
    }
    return interval_map[interval]

def convert_date_to_string(date):
    # date
    if isinstance(date, datetime.datetime):
        return date.strftime('%d %b %Y')
    if isinstance(date, datetime.date):
        return date.strftime('%d %b %Y')
    if isinstance(date, pd.Timestamp):
        return date.strftime('%d %b %Y')
    
    return date 

def get_historical_klines(
    symbol,
    interval, # 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M
    start_str,  # 5 July 2020
    end_str=None):    
    interval = map_interval_to_enum(interval)
    start_str = convert_date_to_string(start_str)
    end_str = convert_date_to_string(end_str)
    
    print(f'Fetching API3 {symbol} {interval} {start_str} {end_str}')
    
    hist = client.get_historical_klines(symbol, interval, start_str, end_str)
    hist_df = pd.DataFrame(hist)
    
    hist_df.columns = ['Open Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close Time', 'Quote Asset Volume', 
                    'Number of Trades', 'TB Base Volume', 'TB Quote Volume', 'Ignore']
    
    hist_df['Open Time'] = pd.to_datetime(hist_df['Open Time']/1000, unit='s')
    hist_df['Close Time'] = pd.to_datetime(hist_df['Close Time']/1000, unit='s')
    
    numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Quote Asset Volume', 'TB Base Volume', 'TB Quote Volume']
    
    hist_df[numeric_columns] = hist_df[numeric_columns].apply(pd.to_numeric, axis=1)
    
    hist_df['tradingDate'] = hist_df['Close Time']
    hist_df['close'] = hist_df['Close']
    
    return hist_df

def invalidate_cache(file_path):
    is_valid = True
    if is_valid:
        return False
    
    if os.path.exists(file_path):
        os.remove(file_path)
        
    return True

def process_klines(klines_df):
    klines_df['tradingDate'] = klines_df['Close Time']
    klines_df['date'] = klines_df['Close Time']
    klines_df['close'] = klines_df['Close']
    klines_df['volume'] = klines_df['Volume']
    klines_df['open'] = klines_df['Open']
    klines_df['high'] = klines_df['High']
    klines_df['low'] = klines_df['Low']
    
    # change column to datetime 2021-12-01 
    klines_df['tradingDate'] = pd.to_datetime(klines_df['tradingDate'])
    
    # set index to tradingDate
    klines_df.set_index('tradingDate', inplace=True)
    
    return klines_df

def get_historical_klines_cached(
    symbol,
    interval, # 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M
    start_str,  # 5 July 2020
    end_str=None):
        print(f'Fetching {symbol} {interval} {start_str} {end_str}')
        cache_file = f'data/binance/{symbol}_{interval}_{start_str}_{end_str}.csv'
        
        if os.path.exists(cache_file) and not invalidate_cache(cache_file):
            print(f'Fetching from cache {symbol} {interval} {start_str} {end_str}')
            csv = pd.read_csv(cache_file)
            return process_klines(csv)
        
        data = get_historical_klines(symbol, interval, start_str, end_str)
        data.to_csv(cache_file, index=False)
        
        
        return process_klines(data)
    
@lru_cache
def get_bin_crypto(symbol: str, start_date: str, end_date: str, timeframe='1D') -> pd.DataFrame:
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

    stock_df = get_historical_klines(symbol, interval='1d', start_str=start_date, end_str=end_date)
    
    # set date to Open Time
    return process_klines(stock_df)

def load_market_overview():

    
    url = "https://api.livecoinwatch.com/coins/list"
    
    payload = json.dumps({ 
        "currency": "USD", 
        "sort": "rank", 
        "order": "ascending", 
        "offset": 0, 
        "limit": 100, 
        "meta": True
        }) 
    
    headers = { 
    'content-type': 'application/json', 
    'x-api-key': 'XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX'
    } 
    
    response = requests.request("POST", url,  
                                headers=headers,  
                                data=payload) 
    response_json = response.json() 
    
    return response_json