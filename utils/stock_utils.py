import requests_cache 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
import os
from time import sleep
from datetime import datetime
from datetime import timezone
from bs4 import BeautifulSoup
import yfinance as yf
import re
from utils.calendar_utils import get_last_working_day_before
import utils.config as cfg
import requests

import pytz

from utils.misc import retry

CACHE_TTL = 60 * 60 * 24  # 1 day
requests = requests_cache.CachedSession('cache/tcbs_cache', expire_after=CACHE_TTL, allowable_codes=[200])

MAX_RETRIES = 5
RETRY_WAIT_TIME = 30


def get_stock_bars(ticker, time, stock_type, count_back, resolution='1'):
    url = f"https://apipubaws.tcbs.com.vn/stock-insight/v2/stock/bars"
    if stock_type == "derivative":
        # https://apipubaws.tcbs.com.vn/futures-insight/v2/stock/bars?ticker=VN30F1M&type=derivative&resolution=1&to=1703821260&countBack=54
        url = "https://apipubaws.tcbs.com.vn/futures-insight/v2/stock/bars"

    params = {
        "ticker": ticker,
        "type": stock_type,
        "resolution": resolution,
        "to": time,
        "countBack": count_back
    }

    headers = {
        "Accept-language": "vi",
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Referer": "https://tcinvest.tcbs.com.vn/",
        "sec-ch-ua-platform": "macOS"
    }

    response = requests.get(url, params=params, headers=headers)

    if response.status_code == 200:
        return response.json()
    else:
        print(f"Failed to retrieve data. Status code: {response.status_code}")
        return None

@retry(times=MAX_RETRIES, exceptions=(Exception), delay=RETRY_WAIT_TIME)
def get_stock_bars_long_term(ticker, stock_type, time=None, count_back=500, resolution='D'):
    timestamp = int(datetime.now().timestamp()) if time is None else time
    url = f"https://apipubaws.tcbs.com.vn/stock-insight/v2/stock/bars-long-term"

    if stock_type == "derivative":
        #      https://apipubaws.tcbs.com.vn/futures-insight/v2/stock/bars-long-term?ticker=VN30F1M&type=derivative&resolution=D&to=1502236800&countBack=1074
        url = "https://apipubaws.tcbs.com.vn/futures-insight/v2/stock/bars-long-term"

    # https://apipubaws.tcbs.com.vn/stock-insight/v2/stock/bars-long-term?ticker=CFPT2303&type=coveredWarr&resolution=D&to=1701388800&countBack=848
    if stock_type == "coveredWarr":
        url = "https://apipubaws.tcbs.com.vn/stock-insight/v2/stock/bars-long-term"
#
    params = {
        "ticker": ticker,
        "type": stock_type,
        "resolution": resolution,
        "to": timestamp,
        "countBack": count_back
    }

    headers = {
        "Accept-language": "vi",
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Referer": "https://tcinvest.tcbs.com.vn/",
        "sec-ch-ua-platform": "macOS"
    }

    response = requests.get(url, params=params, headers=headers)

    if response.status_code == 200:
        return response.json()
    else:
        print(f"Failed to retrieve data. Status code: {response.status_code}")
        return None

def get_stock_bars_very_long_term(ticker, stock_type, count_back=300, resolution='D', total_page=10):
    timestamp = int(datetime.now().timestamp())
    page = 1
    data = []

    retry_count = 0

    while page < total_page:
        try:
            res = {
                'data': []
            }
            if resolution == 'D' or resolution == 'W' or resolution == 'M':
                res = get_stock_bars_long_term(
                    ticker=ticker, stock_type=stock_type, time=timestamp, count_back=count_back, resolution=resolution)
            else:
                res = get_stock_bars(ticker=ticker, stock_type=stock_type,
                                     time=timestamp, count_back=count_back, resolution=resolution)

            if res is None or not res.get('data'):
                raise ValueError('Failed to fetch data or empty response')

            data_reversed = res['data'][::-1]

            # De-duplicate by tradingDate
            for item in data_reversed:
                if item['tradingDate'] not in [x['tradingDate'] for x in data]:
                    data.insert(0, item)

            page += 1
            last_item = data_reversed[-1]
            last_tradingDate = last_item['tradingDate']
            # tradingDate format 2023-09-08T07:00:00.000Z
            last_timestamp = int(datetime.strptime(
                last_tradingDate, '%Y-%m-%dT%H:%M:%S.%fZ').timestamp())
            timestamp = last_timestamp

        except (ValueError, KeyError) as e:
            print(f"Error: {e}")
            retry_count += 1
            if retry_count <= MAX_RETRIES:
                sleep(RETRY_WAIT_TIME)
                continue
            else:
                print("Max retries reached. Terminating data retrieval.")
                break

    return {
        'data': data,
        'total': len(data)
    }


def invalidate_cache_file(file_path, mindays=3):
    df = pd.read_csv(file_path, parse_dates=['tradingDate'], header=0)

    last_value = df.iloc[-1]['tradingDate'].to_pydatetime().replace(tzinfo=None)

    # valid if last value is within 3 days
    is_valid = (datetime.now() - last_value).days < mindays

    if not is_valid:
        print(f'Cache file is invalid: {file_path}')
        # remove cache file
        os.remove(file_path)

    return not is_valid

def filter_data_by_date(df, start_date, end_date):
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
    
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, '%Y-%m-%d')
        
    print(f"Filtering data from {start_date} to {end_date}")
    
    if start_date is not None:
        df = df[df['tradingDate'].dt.date >= start_date.date()]

    if end_date is not None:
        df = df[df['tradingDate'].dt.date <= end_date.date()]

    return df

# check csv file exists
# if exists, load data from csv file
# else, fetch data from api and save to csv file
def get_stock_bars_very_long_term_cached(
    ticker,
    stock_type,
    count_back=300,
    refresh=False,
    no_fetch=False,
    resolution='D',
    total_page=10,
    force_fetch=False,
    set_index=True,
    start_date=None,
    end_date=None
):  
    print(f"Fetching data for {ticker} {stock_type} {resolution}")
    
    # Default combine period
    combine_period = 1
    
    # Normalize resolution input
    if resolution is None:
        resolution = 'D'
    elif resolution == '1D':
        resolution = 'D'
    elif resolution == '1W':
        resolution = 'W'
    elif resolution == '1M':
        resolution = 'M'
    elif resolution == '3D':
        resolution = 'D'  # Base resolution is daily
        combine_period = 3  # Combine 3 daily bars
    
    # Minimum days for cache validation
    mindays = 3
    if resolution == 'W':
        mindays = 8
    elif resolution == 'M':
        mindays = 32
        
    resolution_slug = f"_{resolution.lower()}" if resolution != 'D' else ''
    long_terms_cache_file = f'./data/prices/{stock_type}_{ticker}{resolution_slug}.csv'
    short_terms_cache_file = f'./data/caches/{stock_type}_{ticker}{resolution_slug}_{datetime.now().strftime("%Y%m%d")}.csv'

    df = pd.DataFrame()

    # Load from cache or fetch data
    if os.path.exists(long_terms_cache_file) and not invalidate_cache_file(long_terms_cache_file, mindays=mindays) and not refresh:
        print(f'Loading data from csv file: {long_terms_cache_file}')
        df = pd.read_csv(long_terms_cache_file, parse_dates=['tradingDate'], header=0)
        df = load_data_into_dataframe(df, set_index=set_index)
        df = filter_data_by_date(df, start_date, end_date)
        
    elif os.path.exists(short_terms_cache_file) and not force_fetch:
        print(f'Loading data from cached csv file: {short_terms_cache_file}')
        df = pd.read_csv(short_terms_cache_file, parse_dates=['tradingDate'], header=0)
        df = load_data_into_dataframe(df, set_index=set_index)
        df = filter_data_by_date(df, start_date, end_date)

    elif no_fetch and not force_fetch:
        df = pd.DataFrame()
    elif stock_type == 'yf':
        print(f'Fetching data from Yahoo Finance API: {stock_type}_{ticker}')
        yf_resolution = '1d' if resolution == 'D' else resolution.lower()
        data = yf.download(ticker, period="max", interval=yf_resolution)
        if not data.empty:
            data['tradingDate'] = data.index
            data['close'] = data['Close']
            data['volume'] = data['Volume']
            df = data[['tradingDate', 'Open', 'High', 'Low', 'close', 'volume']]
            df.columns = ['tradingDate', 'open', 'high', 'low', 'close', 'volume']
    else:
        print(f'Fetching data from API: {stock_type} {ticker} {resolution}')
        data = get_stock_bars_very_long_term(
            ticker=ticker, stock_type=stock_type, count_back=count_back, resolution=resolution, total_page=total_page)
        df = load_data_into_dataframe(data, set_index=set_index)
        if not df.empty:
            df.to_csv(long_terms_cache_file, index=False)
            df.to_csv(short_terms_cache_file, index=False)
        df = filter_data_by_date(df, start_date, end_date)

    # Combine bars if 3D timeframe is requested
    if combine_period > 1 and not df.empty:
        # Ensure tradingDate is available and sorted
        if 'tradingDate' not in df.columns:
            df['tradingDate'] = df.index
            
        df['tradingDate'] = pd.to_datetime(df['tradingDate'])  # Ensure datetime type
        df = df.sort_values('tradingDate')
        
        # Assign a group number to every 3 bars
        df['group'] = (df.index - df.index[0]) // combine_period
        
        # Aggregate the bars
        df = df.groupby('group').agg({
            'tradingDate': 'first',  # Use the first date of the group
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).reset_index(drop=True)

    return df


def minutes_to_9am_ho_chi_minh(date_time_input):
    # Check if the input is a string and convert to datetime if needed
    if isinstance(date_time_input, str):
        try:
            date_time = datetime.strptime(date_time_input, '%Y-%m-%d %H:%M:%S')
        except ValueError:
            return "Invalid date format. Please provide the date and time in the format 'YYYY-MM-DD HH:MM:SS'"
    elif isinstance(date_time_input, datetime):
        date_time = date_time_input
    else:
        return "Invalid input type. Please provide a string or datetime object."

    # Set the target time to 9:00 AM in the Ho Chi Minh timezone (ICT)
    ho_chi_minh_tz = pytz.timezone('Asia/Ho_Chi_Minh')
    date_time = ho_chi_minh_tz.localize(date_time)
    target_time = ho_chi_minh_tz.localize(datetime.combine(
        date_time.date(), datetime.min.time())).replace(hour=9, minute=0)

    # Calculate the difference in minutes between the given date and 9:00 AM in Ho Chi Minh timezone
    minutes_difference = int((target_time - date_time).total_seconds() / 60)

    return minutes_difference

# Function to load data into a Pandas DataFrame


def load_data_into_dataframe(data, set_index=True):
    stock_data = data
    if 'data' in data:
        stock_data = data['data']

    df = data if isinstance(
        stock_data, pd.DataFrame) else pd.DataFrame(stock_data)

    if 'tradingDate' in df.columns:
        df['tradingDate'] = pd.to_datetime(df['tradingDate'])
        
        df['tradingDate'] = df['tradingDate'].dt.tz_localize(None)
        
                
        # df['tradingDate'] = df['tradingDate'].dt.date
        
        df['tradingDate'] = pd.to_datetime(df['tradingDate'])
    
        # sort by tradingDate
        df.sort_values(by=['tradingDate'], inplace=True)

        df['index'] = df['tradingDate']

        df.set_index('index', inplace=True)

    return df


def load_price_data_into_yf_dataframe(df, set_index=True):
    # df = df.rename(columns={'tradingDate': 'date', 'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close', 'volume': 'volume'})
    df['date'] = df['tradingDate']
    df['datetime'] = df['date']

    # to datetime
    df['date'] = pd.to_datetime(df['date'])
    df['datetime'] = pd.to_datetime(df['datetime'])

    if set_index:
        df.set_index('date', inplace=True)

    return df


def load_dividend_data_into_dataframe(df):
    if df is not None:
        # when 'ExDiviendDate' is isoformat, convert to %d/%m/%Y
        def convert_date(x):
            if isinstance(x, str):
                if 'T' in x and 'Z' in x:
                    return datetime.strptime(x, '%Y-%m-%dT%H:%M:%S.%fZ').strftime('%d/%m/%Y')
                elif 'T' in x:
                    return datetime.strptime(x, '%Y-%m-%dT%H:%M:%S').strftime('%d/%m/%Y')
            return x

        df['ExDiviendDate'] = df['ExDiviendDate'].apply(convert_date)

        df['ticker'] = df['Ticker'].astype(str)
        date_format = "%d/%m/%Y - %H:%M:%S"
        df['Date'] = pd.to_datetime(df['Date'], format=date_format)

        date_format_2 = "%d/%m/%Y"
        df['ExDiviendDate'] = pd.to_datetime(
            df['ExDiviendDate'], format=date_format_2)

        df = df[df['LastRegistrationDate'].notna()]

        # if type is cash, set StockRate to 0
        df.loc[df['Type'] == 'cash', 'StockRate'] = 0

        # if type is stock, set CashRate to 0
        df.loc[df['Type'] == 'stocks', 'CashRate'] = 0

        # sort by date
        df.sort_values(by=['Date'], inplace=True)

        # drop ticker blank
        df = df[df['Ticker'] != '']

        df = clean_dividend_data(df)
        return df

    else:
        print("No data to load into DataFrame.")
        return None


def calculate_spread(df):

    # Calculate spread (Price1 - Price2)
    df['Spread'] = df['close_vn30f1m'] - df['close_vn30']

    # Calculate spread mean and standard deviation for each day
    df['Spread_Mean'] = df.groupby(df['tradingDate'].dt.date)[
        'Spread'].transform('mean')
    df['Spread_Std'] = df.groupby(df['tradingDate'].dt.date)[
        'Spread'].transform('std')

    # Drop duplicate rows to keep unique dates
    df.drop_duplicates(subset=['tradingDate'], inplace=True)

    return df


def fetch_daytrade_his(ticker='VN30F2310', page=0, size=50, headIndex=-1, type='futures'):
    url = f'https://apipubaws.tcbs.com.vn/{type}-insight/v1/intraday/{ticker}/his/paging?page={page}&size={size}&headIndex={headIndex}'
    headers = {
        'Host': 'apipubaws.tcbs.com.vn',
        'sec-ch-ua': '"Google Chrome";v="117", "Not;A=Brand";v="8", "Chromium";v="117"',
        'DNT': '1',
        'Accept-language': 'vi',
        'sec-ch-ua-mobile': '?0',
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36',
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'Referer': 'https://tcinvest.tcbs.com.vn/',
        'sec-ch-ua-platform': '"macOS"'
    }

    response = requests.get(url, headers=headers)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Return the JSON content
        return response.json()
    else:
        # If the request was unsuccessful, print the status code and response content
        print('Request failed with status code:', response.status_code)
        print('Response content:', response.text)


def fetch_daytrade_his_long(ticker, headIndex, limit, type='futures'):
    data = []
    page = 0
    total = limit

    while True:
        response = fetch_daytrade_his(
            ticker=ticker, page=page, size=100, headIndex=headIndex, type=type)
        page_data = response.get('data', [])
        total = response.get('total', 0)

        if not page_data:
            # No more data available
            break

        data.extend(page_data)
        page += 1

        if limit and len(data) >= limit or len(data) >= total:
            # Reached the specified limit
            break

    return data


def calculate_market_depth_and_dominant_levels(dataframe):
    # Sort the DataFrame by price for bids (descending) and asks (ascending)
    sorted_data = dataframe.sort_values(by=['price'], ascending=[False])

    # Calculate cumulative bid and ask volumes
    sorted_data['cumulative_bid_volume'] = sorted_data['buy_volume'].cumsum()
    sorted_data['cumulative_ask_volume'] = sorted_data['sell_volume'].cumsum()

    # Calculate the rate of change in bid and ask volumes
    sorted_data['bid_volume_change'] = sorted_data['cumulative_bid_volume'].diff()
    sorted_data['ask_volume_change'] = sorted_data['cumulative_ask_volume'].diff()

    # Find the dominant bid and ask price levels based on rate of change
    dominant_bid_price = sorted_data.loc[sorted_data['bid_volume_change'].idxmax(
    ), 'price'] if not sorted_data.empty else None
    dominant_ask_price = sorted_data.loc[sorted_data['ask_volume_change'].idxmax(
    ), 'price'] if not sorted_data.empty else None

    # The best bid is the highest price among buy orders
    best_bid = sorted_data.iloc[0]['price'] if not sorted_data.empty else None

    # The best ask is the lowest price among sell orders
    best_ask = sorted_data.iloc[-1]['price'] if not sorted_data.empty else None

    return sorted_data, best_bid, best_ask, dominant_bid_price, dominant_ask_price


def get_vsd_stock_news(ticker):
    url = 'https://www.vsd.vn/isuisser-tcdk/search'
    headers = {
        'Accept': '*/*',
        'Accept-Language': 'en-US,en;q=0.9,vi;q=0.8,vi-VN;q=0.7,fr-FR;q=0.6,fr;q=0.5,de;q=0.4',
        'Connection': 'keep-alive',
        'Content-Type': 'application/json;charset=UTF-8',
        'DNT': '1',
        'Origin': 'https://www.vsd.vn',
        'Pragma': 'no-cache',
        'Sec-Fetch-Dest': 'empty',
        'Sec-Fetch-Mode': 'cors',
        'Sec-Fetch-Site': 'same-origin',
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36',
        'X-Requested-With': 'XMLHttpRequest',
        'sec-ch-ua': '"Google Chrome";v="117", "Not;A=Brand";v="8", "Chromium";v="117"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"macOS"'
    }
    data = {
        "SearchKey": "817",
        "CurrentPage": 1,
        "RecordOnPage": 10
    }
    try:
        response = requests.post(url, headers=headers, data=data)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            data_list = []
            news_list = soup.find_all('li')

            for news in news_list:
                title = news.find('h3').text.strip()
                time = news.find('div', class_='time-news').text.strip()
                link = news.find('a')['href']
                data = {
                    'title': title,
                    'time': time,
                    'link': link
                }
                data_list.append(data)

            return data_list
        else:
            print('Request failed. Status code:', response.status_code)
    except requests.exceptions.RequestException as e:
        print('An error occurred:', e)


def clean_dividend_data(diviend_data):
    # Replace 'unknown' with NaN in cash rate
    diviend_data['CashRate'] = diviend_data['CashRate'].replace(
        'unknown', np.nan)

    # Ensure all values in the column are treated as strings
    diviend_data['CashRate'] = diviend_data['CashRate'].astype(str)
    diviend_data['StockRate'] = diviend_data['StockRate'].astype(str)

    # when cash rate is 4186.9, convert to 4186
    diviend_data['CashRate'] = diviend_data['CashRate'].apply(
        lambda x: x if not isinstance(x, str) else x.split('.')[0] if '.' in x else x)

    # Remove commas from cash rate
    diviend_data['CashRate'] = diviend_data['CashRate'].str.replace(',', '')

    # Remove periods from cash rate
    diviend_data['CashRate'] = diviend_data['CashRate'].str.replace('.', '')

    # Apply .str.replace('%', '') only to rows of string type
    diviend_data['StockRate'] = diviend_data['StockRate'].apply(
        lambda x: x if not isinstance(x, str) else x.replace('%', ''))

    # Fill empty stock rate and cash rate with NaN
    diviend_data['StockRate'] = diviend_data['StockRate'].replace('', np.nan)
    diviend_data['CashRate'] = diviend_data['CashRate'].replace('', np.nan)

    # when diviend_cash_rate has space, remove part after space
    diviend_data['CashRate'] = diviend_data['CashRate'].apply(
        lambda x: x if not isinstance(x, str) else x.split(' ')[0] if ' ' in x else x)

    # Convert None to '' in stock rate
    diviend_data['StockRate'] = diviend_data['StockRate'].replace('None', 0)

    # Convert None to '' in cash rate
    diviend_data['CashRate'] = diviend_data['CashRate'].replace('None', 0)

    # convert stock rate 100:28 to 28
    diviend_data['StockRate'] = diviend_data['StockRate'].apply(
        lambda x: x if not isinstance(x, str) else x.split(':')[1] if ':' in x else x)

    # Convert stock rate to float (NaN will remain as is)
    diviend_data['StockRate'] = diviend_data['StockRate'].astype(float)

    # Convert cash rate to float (NaN will remain as is)
    diviend_data['CashRate'] = diviend_data['CashRate'].astype(float)

    # remove all Not a Time (NaT) values
    diviend_data = diviend_data[diviend_data['ExDiviendDate'].notna()]

    diviend_data['ExDiviendDate'] = pd.to_datetime(
        diviend_data['ExDiviendDate'])

    # correct LastRegistrationDate logic
    for index, row in diviend_data.iterrows():
        diviend_data.loc[index, 'LastRegistrationDate'] = row['ExDiviendDate']

        ex_dividend_date = row['ExDiviendDate'].date()
        last_working_day = get_last_working_day_before(ex_dividend_date)
        diviend_data.loc[index, 'ExDiviendDate'] = last_working_day

    # convert last_working_day from 'Timestamp' object to padnas 'datetime' object
    diviend_data['ExDiviendDate'] = pd.to_datetime(
        diviend_data['ExDiviendDate'])
    diviend_data['LastRegistrationDate'] = pd.to_datetime(
        diviend_data['LastRegistrationDate'])

    print(f"{len(diviend_data)} Dividend data cleaned successfully.")

    return diviend_data


def load_diviend_data_from_api(offset=0, limit=1000, where=""):
    url = "http://147.182.235.249:8080/api/v1/db/data/noco/phrwhwj4d9xr7sa/dividend/views/dividend"

    headers = {"xc-token": "mwX6RTqmw542zhtEvu8HCyF-hvxDvDXqArcD6gvX"}

    params = {
        'offset': offset,
        'limit': limit,
        'where': where
    }

    response = requests.get(url, headers=headers, params=params)

    if response.status_code == 200:
        data = response.json()  # Assuming the response is in JSON format
        return data['list']
    else:
        print(f"Failed to load data. Status code: {response.status_code}")
        return None


def load_diviend_data_from_api_many():
    offset = 0
    limit = 1000
    where = ""
    data = []
    while True:
        response = load_diviend_data_from_api(
            offset=offset, limit=limit, where=where)
        if not response:
            break
        data.extend(response)
        offset += limit
    return data


def load_issuance_data_from_api(offset=0, limit=1000, where=""):
    url = "http://147.182.235.249:8080/api/v1/db/data/noco/phrwhwj4d9xr7sa/Issuance/views/Issuance"

    headers = {"xc-token": "mwX6RTqmw542zhtEvu8HCyF-hvxDvDXqArcD6gvX"}

    params = {
        'offset': offset,
        'limit': limit,
        'where': where
    }

    response = requests.get(url, headers=headers, params=params)

    if response.status_code == 200:
        data = response.json()  # Assuming the response is in JSON format
        return data['list']
    else:
        print(f"Failed to load data. Status code: {response.status_code}")
        return None


def load_issuance_data_from_api_many():
    offset = 0
    limit = 1000
    where = ""
    data = []
    while True:
        response = load_issuance_data_from_api(
            offset=offset, limit=limit, where=where)
        if not response:
            break
        data.extend(response)
        offset += limit
    return data


def load_issuance_data_into_dataframe(df):
    if df is not None:
        df['ticker'] = df['Ticker'].astype(str)
        date_format = "%d/%m/%Y - %H:%M:%S"
        df['Date'] = pd.to_datetime(df['Date'], format=date_format)

        # sort by date
        df.sort_values(by=['Date'], inplace=True)

        # drop ticker blank
        df = df[df['Ticker'] != '']

        # remove all Not a Time (NaT) values
        df = df[df['LastRegistrationDate'] != '']

        df = df[df['LastRegistrationDate'] != None]

        df = df[df['LastRegistrationDate'].notna()]

        df['LastRegistrationDate'] = pd.to_datetime(df['LastRegistrationDate'])

        df['EffectiveStartDate'] = pd.to_datetime(
            df['EffectiveStartDate'], dayfirst=True)

        df['EffectiveEndDate'] = pd.to_datetime(
            df['EffectiveEndDate'], dayfirst=True)

        df = df[df['LastRegistrationDate'].notna()]

        # correct LastRegistrationDate logic
        for index, row in df.iterrows():
            last_regestration_date = row['LastRegistrationDate'].date()
            ex_diviend_date = get_last_working_day_before(
                last_regestration_date)
            df.loc[index, 'ExDiviendDate'] = ex_diviend_date

        # convert last_working_day from 'Timestamp' object to padnas 'datetime' object
        df['ExDiviendDate'] = pd.to_datetime(df['ExDiviendDate'])
        df['LastRegistrationDate'] = pd.to_datetime(df['LastRegistrationDate'])

        # drop ImplementationRatio blank
        df = df[df['ImplementationRatio'] != '']

        # convert ImplementationRatio to float (NaN will remain as is)
        df = df[df['ImplementationRatio'].notna()]
        df['ImplementationRatio'] = df['ImplementationRatio'].astype(float)
        df = df[df['ImplementationRatio'].notna()]

        # drop ImplementationRatio zeros
        df = df[df['ImplementationRatio'] != 0]

        # convert IssuancePrice to float (NaN will remain as is)
        df = df[df['IssuancePrice'].notna()]
        df['IssuancePrice'] = df['IssuancePrice'].astype(int)

        # drop IssuancePrice zero
        df = df[df['IssuancePrice'] != 0]

        df = df[df['IssuancePrice'] != 0]

        df = df[df['ImplementationRatio'].notna()]
        df = df[df['IssuancePrice'].notna()]

        df = df[df['LastRegistrationDate'].notna()]
        df = df[df['EffectiveStartDate'].notna()]
        df = df[df['EffectiveEndDate'].notna()]

        # Filter out rows where effective end date is before effective start date
        df = df[df['EffectiveEndDate'] > df['EffectiveStartDate']]

        # if duplicate ticker, and last registration date is the same, keep the first one
        df = df.drop_duplicates(
            subset=['ticker', 'LastRegistrationDate'], keep='first')

        # # Remove rows containing "Thay đổi" in the 'Title' column
        # df = df[~df['Title'].str.contains('Thay đổi')]

        # # Remove rows containing "Gia hạn" in the 'Title' column
        # df = df[~df['Title'].str.contains('Gia hạn')]

        return df

    else:
        print("No data to load into DataFrame.")
        return None


def load_activities_data_into_dataframe(df):
    if df is not None:
        df['ticker'] = df['Ticker'].astype(str)
        date_format = "%d/%m/%Y - %H:%M:%S"
        df['PublishDate'] = pd.to_datetime(df['PublishDate'])

        df['date'] = df['PublishDate']

        df['ticker'] = df['Ticker'].astype(str)

        # remove "'" from PriceChangeRatio
        df['PriceChangeRatio'] = df['PriceChangeRatio'].str.replace("'", "")

        # remove "'" from PriceChangeRatio1W
        df['PriceChangeRatio1W'] = df['PriceChangeRatio1W'].str.replace(
            "'", "")

        # remove "'" from PriceChangeRatio1M
        df['PriceChangeRatio1M'] = df['PriceChangeRatio1M'].str.replace(
            "'", "")

        # convert 'PriceChangeRatio' to float (NaN will remain as is)
        df['PriceChangeRatio'] = df['PriceChangeRatio'].astype(float)

        # convert 'PriceChangeRatio1W' to float (NaN will remain as is)
        df['PriceChangeRatio1W'] = df['PriceChangeRatio1W'].astype(float)

        # convert 'PriceChangeRatio1M' to float (NaN will remain as is)
        df['PriceChangeRatio1M'] = df['PriceChangeRatio1M'].astype(float)

        # sort by date
        df.sort_values(by=['PublishDate'], inplace=True)

        return df
    else:
        print("No data to load into DataFrame.")
        return None


def load_activities_data_from_api(offset=0, limit=1000, where=""):
    view = "Activities"

    if where == "":
        view = "Positive"

    url = f"http://147.182.235.249:8080/api/v1/db/data/noco/phrwhwj4d9xr7sa/Activities/views/{view}"

    headers = {"xc-token": "mwX6RTqmw542zhtEvu8HCyF-hvxDvDXqArcD6gvX"}

    params = {
        'offset': offset,
        'limit': limit,
        'where': where
    }

    response = requests.get(url, headers=headers, params=params)

    if response.status_code == 200:
        data = response.json()  # Assuming the response is in JSON format
        return data['list']
    else:
        print(f"Failed to load data. Status code: {response.status_code}")
        return None


def load_activities_data_from_api_many(ticker=None):
    offset = 0
    limit = 1000
    where = f"(Ticker,eq,{ticker})" if ticker else ""
    data = []
    while True:
        response = load_activities_data_from_api(
            offset=offset, limit=limit, where=where)
        if not response:
            break
        data.extend(response)
        offset += limit
    return data


def get_vn30_data():
    data_df = pd.read_csv('./data/market/vn30.csv')

    # convert string to int x.xx -> xxx
    data_df['floatShares'] = data_df['floatShares'].str.replace('.', '')

    # convert floatShares to int
    data_df['floatShares'] = data_df['floatShares'].astype(int)

    return data_df


def get_warrants_data():
    url = 'https://apipubaws.tcbs.com.vn/tcbs-hfc-data/v1/cw-info?lang=vi'
    headers = {
        'sec-ch-ua': '"Google Chrome";v="119", "Chromium";v="119", "Not?A_Brand";v="24"',
        'DNT': '1',
        'Accept-language': 'vi',
        'sec-ch-ua-mobile': '?0',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'Referer': 'https://tcinvest.tcbs.com.vn/',
        'sec-ch-ua-platform': '"Windows"',
    }

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        json_data = response.json()
        return json_data
    else:
        print(f"Request failed with status code {response.status_code}")
        return None

def get_stock_info_data(tickers):
    print(f"Fetching stock info data for {len(tickers)} tickers")
    url = 'https://iboard-query.ssi.com.vn/v2/stock/multiple'
    headers = {
        'authority': 'iboard-query.ssi.com.vn',
        'accept': 'application/json, text/plain, */*',
        'accept-language': 'vi',
        'content-type': 'application/json',
        'device-id': 'B62311E1-AA20-4712-93CD-DF66FC4D5FDC',
        'dnt': '1',
        'newrelic': 'eyJ2IjpbMCwxXSwiZCI6eyJ0eSI6IkJyb3dzZXIiLCJhYyI6IjM5NjY4NDAiLCJhcCI6IjU5NDQzMzA3MiIsImlkIjoiNjY0MTA0YjI1NDE3ZWFlNyIsInRyIjoiNmRiYTk3YTgzZTE5MTgxMDJiOWE0NGMyMmFjODEyMDAiLCJ0aSI6MTcwMTI5OTMwNTA1OX19',
        'origin': 'https://iboard.ssi.com.vn',
        'pragma': 'no-cache',
        'referer': 'https://iboard.ssi.com.vn/',
        'sec-ch-ua': '"Google Chrome";v="119", "Chromium";v="119", "Not?A_Brand";v="24"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"macOS"',
        'sec-fetch-dest': 'empty',
        'sec-fetch-mode': 'cors',
        'sec-fetch-site': 'same-site',
        'traceparent': '00-6dba97a83e1918102b9a44c22ac81200-664104b25417eae7-01',
        'tracestate': '3966840@nr=0-1-3966840-594433072-664104b25417eae7----1701299305059',
        'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
    }

    data = {
        'stocks': tickers,
    }
    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}")
        return None


def load_cw_info_to_dataframe(cw_info_data):
    # Rename columns for cleaner representation
    column_mapping = {
        "sn": "Security_Number",
        "ap": "Average_Price",
        "b1": "Bid_Price_1",
        "b1v": "Bid_Volume_1",
        "b2": "Bid_Price_2",
        "b2v": "Bid_Volume_2",
        "b3": "Bid_Price_3",
        "b3v": "Bid_Volume_3",
        "c": "Closing_Price",
        "cpe": "Corporate_Events",
        "cs": "Corporate_Sector",
        "cwt": "Covered_Warrant_Type",
        "e": "Exchange",
        "ep": "Exercise_Price",
        "er": "Exercise_Ratio",
        "f": "Floor",
        "h": "High_Price",
        "isn": "Issuer_Name",
        "l": "Lot_Size",
        "ltd": "Last_Trading_Date",
        "lv": "Lot_Volume",
        "md": "Maturity_Date",
        "mp": "Market_Price",
        "mtv": "Market_Turnover_Value",
        "o": "Open_Price",
        "o1": "Offer_Price_1",
        "o1v": "Offer_Volume_1",
        "o2": "Offer_Price_2",
        "o2v": "Offer_Volume_2",
        "o3": "Offer_Price_3",
        "o3v": "Offer_Volume_3",
        "pcp": "Previous_Close_Price",
        "r": "Reference_Price",
        "sen": "Security_Name",
        "ss": "Stock_Symbol",
        "st": "Security_Type",
        "ts": "Trading_Status",
        "tsh": "Total_Shares_Held",
        "tu": "Total_Units",
        "us": "Underlying_Stock",
        "ce": "Corporate_Exercises",
        "cv": "Conversion_Value",
        "os": "Option_Style",
        "s": "Settlement_Type",
        "bfq": "Buy_Fill_Quantity",
        "rfq": "Remaining_Fill_Quantity",
        "sfq": "Sell_Fill_Quantity",
        "mv": "Market_Value",
        "cp": "Change_Percentage",
        "lmp": "Last_Matched_Price",
        "lmv": "Last_Matched_Volume",
        "lpc": "Last_Percentage_Change",
        "lpcp": "Last_Percentage_Change_Period",
        "mtq": "Market_Turnover_Quantity",
        "pc": "Price_Change",
    }

    cw_info_df = pd.DataFrame(cw_info_data['data'])
    cw_info_df.rename(columns=column_mapping, inplace=True)

    # conver Exercise_Ratio from string "4:1" to float 4
    cw_info_df['Exercise_Ratio'] = cw_info_df['Exercise_Ratio'].apply(
        lambda x: x if not isinstance(x, str) else x.split(':')[0] if ':' in x else x)

    # convert Exercise_Ratio to float (NaN will remain as is)
    cw_info_df['Exercise_Ratio'] = cw_info_df['Exercise_Ratio'].astype(float)

    # conveer Exercise_Price to float
    cw_info_df['Exercise_Price'] = cw_info_df['Exercise_Price'].astype(float)
    
    # convert Matuirty_Date to datetime
    # 20241009 -> 2024-10-09
    cw_info_df['Maturity_Date'] = pd.to_datetime(
        cw_info_df['Maturity_Date'], format='%Y%m%d')

    return cw_info_df


def warrant_break_even_point(current_warrant_price, execution_price, conversion_ratio):
    """
    Calculate the break-even point for a warrant.

    Parameters:
    - current_warrant_price (float): Current price of the warrant.
    - execution_price (float): Execution price of the warrant.
    - conversion_ratio (float): Conversion ratio of the warrant.

    Returns:
    - float: Break-even point of the warrant.
    """
    break_even_point = (current_warrant_price *
                        conversion_ratio + execution_price)
    return break_even_point


def caculate_rolling_histotical_volatility(df, window_size=252):
    factor = np.sqrt(window_size)
    hv = df['close'].pct_change().rolling(window_size).std() * factor

    return hv


def process_warrants_data(cw_data_merged_df, risk_free_rate=0.05):
    # caculate transaction_value
    cw_data_merged_df['transaction_value'] = cw_data_merged_df.apply(
        lambda x: x['close_cw'] * x['volume_cw'],
        axis=1
    )

    # break_even_price = cw_data_merged_df['Exercise_Price'] / cw_data_merged_df['Exercise_Ratio'] + cw_data_merged_df['Exercise_Price'] * cw_data_merged_df['Exercise_Ratio'] * cw_data_merged_df['Total_Units'] / cw_data_merged_df['Total_Units']
    cw_data_merged_df['break_even_price'] = cw_data_merged_df.apply(
        lambda x: warrant_break_even_point(
            current_warrant_price=x['close_cw'],
            execution_price=x['Exercise_Price'],
            conversion_ratio=x['Exercise_Ratio'],
        ),
        axis=1
    )

    # caculate stock price to break even Close - break_even_price
    cw_data_merged_df['stock_price_to_break_even'] = cw_data_merged_df.apply(
        lambda x: abs((x['close'] - x['break_even_price'])),
        axis=1
    )

    # caculate risk_price =  (exercise_price - Close) / stock_price_to_break_even
    cw_data_merged_df['risk_price'] = cw_data_merged_df.apply(
        lambda x: (x['Exercise_Price'] - x['close']) /
        (x['break_even_price'] - x['Exercise_Price']),
        axis=1
    )

    # caculate risk_price_per_day = risk_price / days_to_expired
    cw_data_merged_df['risk_price_per_day'] = cw_data_merged_df.apply(
        lambda x: x['risk_price'] / x['days_to_expired'],
        axis=1
    )

    # caculate price_risk_price_per_day = close_cw / risk_price_per_day
    cw_data_merged_df['price_risk_price_per_day'] = cw_data_merged_df.apply(
        lambda x: x['risk_price_per_day'] / x['close_cw'],
        axis=1
    )

    # caculate stock percent to break even
    cw_data_merged_df['stock_price_to_break_even_per_day'] = cw_data_merged_df.apply(
        lambda x: (x['stock_price_to_break_even']) / x['days_to_expired'],
        axis=1
    )

    # caculate stock to break even / days to expired
    cw_data_merged_df['stock_percent_to_break_even'] = cw_data_merged_df.apply(
        lambda x: x['stock_price_to_break_even'] / x['close'] * 100,
        axis=1
    )

    # caculate stock to break even / days to expired
    cw_data_merged_df['stock_percent_to_break_even_per_day'] = cw_data_merged_df.apply(
        lambda x: x['stock_percent_to_break_even'] / x['days_to_expired'],
        axis=1
    )

    cw_data_merged_df['price_per_day'] = cw_data_merged_df.apply(
        lambda x: x['close_cw'] /
        x['stock_price_to_break_even_per_day'] if x['stock_price_to_break_even_per_day'] > 0 else 0,
        axis=1
    )

    # Modify the price_per_day metric to be a percentage change relative to the current stock price
    cw_data_merged_df['price_percentage_per_day'] = cw_data_merged_df.apply(
        lambda x: (x['close_cw'] / x['stock_percent_to_break_even_per_day']
                   ) if x['stock_percent_to_break_even_per_day'] > 0 else 0,
        axis=1
    )

    # caculate risk reward ratio = price_per_day / price_risk_price_per_day
    cw_data_merged_df['risk_reward_ratio'] = cw_data_merged_df.apply(
        lambda x:  x['price_risk_price_per_day'] /
        x['price_per_day'] if x['price_risk_price_per_day'] > 0 else 0,
        axis=1
    )

    return cw_data_merged_df

# https://apipubaws.tcbs.com.vn/tcbs-hfc-data/v1/pmd/list?fType=market&fData=A&fTime=6M&fOverall=PushUp&page=0&size=100


def get_stock_warnings(
        data='HCM',  # Ticker, Market, Sector
        f_type='tickers',  # tickers, market, sector
        f_time='6M',  # 1W, 1M, 3M, 6M
        f_overall='A',  # A, PushUp,
        page='0',
        size='100'):
    url = 'https://apipubaws.tcbs.com.vn/tcbs-hfc-data/v1/pmd/list'

    headers = {
        'sec-ch-ua': '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
        'DNT': '1',
        'Accept-language': 'vi',
        'sec-ch-ua-mobile': '?0',
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Content-type': 'application/json',
        'Accept': 'application/json',
        'Referer': 'https://tcinvest.tcbs.com.vn/',
        'sec-ch-ua-platform': '"macOS"'
    }

    params = {
        'fType': f_type,
        'fData': data,
        'fTime': f_time,
        'fOverall': f_overall,
        'page': page,
        'size': size
    }

    response = requests.get(url, params=params, headers=headers)

    if response.status_code == 200:
        json = response.json()
        return json['value']
    else:
        print(f"Request failed with status code {response.status_code}")
        return []


def get_stock_warnings_long_term(
    data='HCM',  # Ticker, Market, Sector
    f_type='tickers',  # tickers, market, sector
    f_time='6M',  # 1W, 1M, 3M, 6M,
    f_overall='A',  # A, PushUp,
):
    print(f"Get stock warnings for {data}")
    url = 'https://apipubaws.tcbs.com.vn/tcbs-hfc-data/v1/pmd/list'

    headers = {
        'sec-ch-ua': '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
        'DNT': '1',
        'Accept-language': 'vi',
        'sec-ch-ua-mobile': '?0',
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Content-type': 'application/json',
        'Accept': 'application/json',
        'Referer': 'https://tcinvest.tcbs.com.vn/',
        'sec-ch-ua-platform': '"macOS"'
    }

    params = {
        'fType': f_type,
        'fData': data,
        'fTime': f_time,
        'fOverall': f_overall,
        'page': '0',
        'size': '100'
    }

    response = requests.get(url, params=params, headers=headers)

    if response.status_code != 200:
        print(f"Request failed with status code {response.status_code}")
        return None

    json = response.json()
    # {
    # "pageSize": 100,
    # "currentPage": 0,
    # "totalPage": 8,
    # "totalRow": 836,
    # "statistic": "299/299 mã có dấu hiệu tăng nóng(NAP: 17|HAG: 12|MCO: 12)",
    # "value": []
    # }
    total_page = json['totalPage']
    print(f"Total page: {total_page}")
    data = json['value']

    for page in range(1, total_page):
        sleep(3)
        print(f"Page: {page}")
        page_data = get_stock_warnings(
            data=data,
            f_type=f_type,
            f_time=f_time,
            f_overall=f_overall,
            page=f'{page}',
            size='100'
        )

        data.extend(page_data)

    return data


def get_stock_warnings_long_term_cache(data='HCM', f_type='tickers', f_time='6M', f_overall='A', refresh=False, retries=3):
    cache_file = f'./data/stock_warnings/{data}_{f_type}_{f_time}_{f_overall}.csv'
    if os.path.exists(cache_file) and not refresh:
        data_df = pd.read_csv(cache_file)
        return data_df
    else:
        data = get_stock_warnings_long_term_safe(
            data=data, f_type=f_type, f_time=f_time, f_overall=f_overall, retries=3)
        data_df = load_stock_warnings_to_dataframe(data)
        data_df.to_csv(cache_file)
        return data_df


def get_stock_warnings_long_term_safe(data='HCM', f_type='tickers', f_time='6M', f_overall='A', retries=3, sleep_time=5):
    for i in range(retries):
        try:
            return get_stock_warnings_long_term(data=data, f_type=f_type, f_time=f_time, f_overall=f_overall)
        except Exception as e:
            print(f"Error: {i}")
            print(e)
            sleep(sleep_time)

    return None


def load_stock_warnings_to_dataframe(value, ma_period=14):
    # samole value
    # [{
    #     "t": "HCM", // ticker
    #     "date": "09-01-2024", // date
    #     "dts": "Cực kỳ hiếm", // signal rarity "Cực kỳ hiếm", "Hiếm", "Bình thường"
    #     "uts": "Hiếm", // consecutive rarity "Cực kỳ hiếm", "Hiếm", "Bình thường"
    #     "pt": "Bình thường", // price trend "Tăng", "Giảm", "Bình thường"
    #     "lt": "Tăng", // liquidity trend "Tăng", "Giảm", "Bình thường"
    #     "sbo": -7775986, // sell buy order
    #     "os": "Trung tính", // overal signal "Trung tính", "Tăng nóng", "Giảm nóng"
    #     "p": 34200, // price
    #     "lsb": [ // list sell buy
    #         -1340988.0,
    #         -486121.0,
    #         -2117719.0,
    #         -2482730.0,
    #         -7775986.0
    #     ],
    #     "ld": [ // list date
    #         "25/12/23",
    #         "26/12/23",
    #         "27/12/23",
    #         "28/12/23",
    #         "29/12/23"
    #     ]
    # }]

    df = pd.DataFrame(value)
    print(df.columns)

    df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')

    df['dts'] = df['dts'].astype('category')
    df['uts'] = df['uts'].astype('category')
    df['pt'] = df['pt'].astype('category')
    df['lt'] = df['lt'].astype('category')
    df['os'] = df['os'].astype('category')

    # rename columns
    # t -> ticker
    df = df.rename(columns={'t': 'ticker',
                            'dts': 'signal_rarity',
                            'uts': 'consecutive_rarity',
                            'pt': 'price_trend',
                            'lt': 'liquidity_trend',
                            'sbo': 'sell_buy_order',
                            'os': 'overall_signal',
                            'p': 'price',
                            'lsb': 'list_sell_buy',
                            'ld': 'list_date'
                            })
    # sort by date
    df.sort_values(by=['date'], inplace=True)

    # convert list_sell_buy to list of int with error handling
    def convert_to_int_list(x):
        try:
            return [int(i) for i in x]
        except (ValueError, TypeError) as e:
            # Handle the error as per your requirement
            print(f"Error converting values in list_sell_buy: {e}")
            return []

    df['list_sell_buy'] = df['list_sell_buy'].apply(convert_to_int_list)

    # Assuming df['list_date'] is a column containing lists of date strings like ['02/10/23', '03/11/24']

    def convert_to_datetime(date_str_list):
        try:
            return [datetime.strptime(date_str, '%d/%m/%y') for date_str in date_str_list]
        except ValueError as e:
            print(f"Error converting date: {e}")
            return None  # or handle the error in a way that suits your needs

    # Apply the function to the 'list_date' column
    df['list_date'] = df['list_date'].apply(convert_to_datetime)

    # buy_sell_order = sell_buy_order * -1
    df['buy_sell_order'] = df['sell_buy_order'] * -1

    # create signal strength
    df['signal_strength'] = 0
    # signal strength = 1 when overall signal is "Tăng nóng"
    df['signal_strength'] = df.apply(
        lambda x: 1 if x['overall_signal'] == 'Tăng nóng' else x['signal_strength'],
        axis=1
    )

    # buy_sell_order_ma10
    df['buy_sell_order_ma'] = df['buy_sell_order'].rolling(ma_period).mean()

    # buy_sell_order_std
    df['buy_sell_order_std'] = df['buy_sell_order'].rolling(ma_period).std()

    # buy_sell_order_zscore
    df['buy_sell_order_zscore'] = (
        df['buy_sell_order'] - df['buy_sell_order_ma']) / df['buy_sell_order_std']

    # buy_sell_order_change_pct
    df['buy_sell_order_change_pct'] = df['buy_sell_order'].pct_change()

    # buy_sell_order_direction = 1 when buy_sell_order_change_pct > 0 else -1
    df['buy_sell_order_direction'] = df.apply(
        lambda x: 1 if x['buy_sell_order_change_pct'] > 0 else -1,
        axis=1
    )

    # buy_sell_order_direction_change_pct = buy_sell_order_change_pct * buy_sell_order_direction
    df['buy_sell_order_direction_change_pct'] = df['buy_sell_order_change_pct'].abs(
    ) * df['buy_sell_order_direction']

    # buy_sell_order_cumsum
    df['buy_sell_order_cumsum'] = df['buy_sell_order'].cumsum()

    for i in range(1, 6):
        # # buy_sell_order_3d
        # df[f'buy_sell_order_{i}d'] = df['list_sell_buy'].apply(lambda x: -sum(x[-i:]))
        df[f'buy_sell_order_{i}d'] = df['buy_sell_order'].rolling(i).sum()

        # f buy_sell_order_{i}d_ma
        df[f'buy_sell_order_{i}d_ma'] = df[f'buy_sell_order_{i}d'].rolling(
            ma_period).mean()

        # f buy_sell_order_{i}d_std
        df[f'buy_sell_order_{i}d_std'] = df[f'buy_sell_order_{i}d'].rolling(
            ma_period).std()

        # f buy_sell_order_{i}d_zscore
        df[f'buy_sell_order_{i}d_zscore'] = (
            df[f'buy_sell_order_{i}d'] - df[f'buy_sell_order_{i}d_ma']) / df[f'buy_sell_order_{i}d_std']

        # f buy_sell_order_{i}d_change_pct
        df[f'buy_sell_order_{i}d_change_pct'] = df[f'buy_sell_order_{i}d'].pct_change(
        )

        # f buy_sell_order_{i}d_direction = 1 whenf buy_sell_order_{i}d_change_pct > 0 else -1
        df[f'buy_sell_order_{i}d_direction'] = df.apply(
            lambda x: 1 if x[f'buy_sell_order_{i}d_change_pct'] > 0 else -1,
            axis=1
        )

        # f buy_sell_order_{i}d_direction_change_pct = buy_sell_order_change_pct * buy_sell_order_direction
        df[f'buy_sell_order_{i}d_direction_change_pct'] = df[f'buy_sell_order_{i}d_change_pct'].abs(
        ) * df[f'buy_sell_order_{i}d_direction']

    # sort by date
    df.sort_values(by=['date'], inplace=True)

    return df

# https://apipubaws.tcbs.com.vn/tcanalysis/v1/company/MWG/insider-dealing?page=0&size=20


def get_stock_insider_dealing(
    ticker='HCM',
    page='0',
    size='20',
):
    url = f'https://apipubaws.tcbs.com.vn/tcanalysis/v1/company/{ticker}/insider-dealing'

    headers = {
        'sec-ch-ua': '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
        'DNT': '1',
        'Accept-language': 'vi',
        'sec-ch-ua-mobile': '?0',
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Content-type': 'application/json',
        'Accept': 'application/json',
        'Referer': 'https://tcinvest.tcbs.com.vn/',
        'sec-ch-ua-platform': '"macOS"'
    }

    params = {
        'page': page,
        'size': size
    }

    response = requests.get(url, params=params, headers=headers)

    if response.status_code == 200:
        json = response.json()
        return json['listInsiderDealing']
    else:
        print(f"Request failed with status code {response.status_code}")
        return []


def get_stock_insider_dealing_long_term_cache(
        ticker='HCM',
        size='20',
        refresh=False,
):
    cache_file = f'./data/stock_insider/insider_dealing_{ticker}.csv'
    if os.path.exists(cache_file) and not refresh:
        data_df = pd.read_csv(cache_file)
        return data_df
    else:
        data = get_stock_insider_dealing_long_term_safe(
            ticker=ticker, size=size)
        data_df = load_stock_insider_dealing_to_dataframe(data)
        data_df.to_csv(cache_file)
        return data_df


def get_stock_insider_dealing_long_term_safe(ticker='HCM', size='20', retries=3, refresh=False):
    for i in range(retries):
        try:
            return get_stock_insider_dealing(ticker=ticker, size=size)
        except Exception as e:
            print(f"Error: {i}")
            print(e)

    return None


def load_stock_insider_dealing_to_dataframe(value, ma_period=14):
    # samole value
    # [
    #     {
    #         "no": 1,
    #         "ticker": "MWG",
    #         "anDate": "10/01/24",
    #         "dealingMethod": 0,
    #         "dealingAction": "0",
    #         "quantity": 200000.0,
    #         "price": 42250.0,
    #         "ratio": 0.085
    #     },
    # ]

    df = pd.DataFrame(value)
    df['anDate'] = pd.to_datetime(df['anDate'], format='%d/%m/%y')
    df['date'] = df['anDate']
    # action = buy when dealingAction = 0 else sell
    df['action'] = df.apply(
        lambda x: 'buy' if x['dealingAction'] == '0' else 'sell',
        axis=1
    )
    df['price'] = df['price'].astype(float)
    df['quantity'] = df['quantity'].astype(float)
    df['ratio'] = df['ratio'].astype(float)
    df['value'] = df['price'] * df['quantity']

    df['quantity_abs'] = df['quantity'].abs()
    df['value_abs'] = df['value'].abs()

    df['buy_quantity'] = df.apply(
        lambda x: x['quantity_abs'] if x['action'] == 'buy' else 0,
        axis=1
    )

    df['sell_quantity'] = df.apply(
        lambda x: x['quantity_abs'] if x['action'] == 'sell' else 0,
        axis=1
    )

    df['buy_value'] = df.apply(
        lambda x: x['value_abs'] if x['action'] == 'buy' else 0,
        axis=1
    )

    df['sell_value'] = df.apply(
        lambda x: x['value_abs'] if x['action'] == 'sell' else 0,
        axis=1
    )

    df['sell_ratio'] = df.apply(
        lambda x: x['ratio'] if x['action'] == 'sell' else 0,
        axis=1
    )

    df['buy_ratio'] = df.apply(
        lambda x: x['ratio'] if x['action'] == 'buy' else 0,
        axis=1
    )

    # df group by date
    df = df.groupby(['date']).agg({
        'buy_quantity': 'sum',
        'sell_quantity': 'sum',
        'buy_value': 'sum',
        'sell_value': 'sum',
        'sell_ratio': 'sum',
        'buy_ratio': 'sum',
    }).reset_index()

    df['quantity_abs'] = df['buy_quantity'] - df['sell_quantity']

    df['ratio_abs'] = df['buy_ratio'] - df['sell_ratio']

    df['value_abs'] = df['buy_value'] - df['sell_value']

    df = df.sort_values(by=['date'], ascending=True)

    return df

def load_stock_evaluation_snapshot_to_dataframe(value):
    if value is None:
        raise ValueError("The input value is None")
    # Convert dictionary to DataFrame-friendly format
    # {'typeId': 'CT', 'index': {'pe': 14.4, 'pb': 1.7, 'evebitda': 20.1}, 'industry': {'pe': 26.6, 'pb': 4.4, 'evebitda': 16.2}, 'top5': [{'ticker': 'CMG', 'pe': 29.2, 'pb': 3.3, 'evebitda': 12.6}, {'ticker': 'SGT', 'pe': 22.7, 'pb': 1.7, 'evebitda': 19.3}, {'ticker': 'SAM', 'pe': 32.2, 'pb': 0.7, 'evebitda': 71.7}, {'ticker': 'ELC', 'pe': 26.9, 'pb': 2.5, 'evebitda': 22.8}, {'ticker': 'ICT', 'pe': 13.7, 'pb': 0.7, 'evebitda': 13.2}], 'cashFlow': [{'year': 2025, 'freeCashFlow': 10654}, {'year': 2026, 'freeCashFlow': 12586}, {'year': 2027, 'freeCashFlow': 15511}, {'year': 2028, 'freeCashFlow': 18608}, {'year': 2029, 'freeCashFlow': 22539}], 'eps': 5336, 'bvps': 20300, 'ebitda': 9540, 'enterpriseValue': 247036522525438, 'cash': 9315440438884, 'shortTermDebt': -14446238451323, 'longTermDebt': -501115537075, 'netDebt': 5631913549514, 'minorityInterest': 0, 'shareOutstanding': 1471069183, 'growth': 0.05, 'wacc': 0.12}
    data_dict = {
        'index_pe': value['index']['pe'],
        'index_pb': value['index']['pb'],
        'index_evebitda': value['index']['evebitda'],
        'industry_pe': value['industry']['pe'],
        'industry_pb': value['industry']['pb'],
        'industry_evebitda': value['industry']['evebitda'],
        'eps': value['eps'],
        'bvps': value['bvps'],
        'ebitda': value['ebitda'],
        'enterprise_value': value['enterpriseValue'],
        'net_debt': value['netDebt'],
        'share_outstanding': value['shareOutstanding'],
        'growth': value['growth'],
        'wacc': value['wacc'],
    }
    
    # Convert top5 stocks to DataFrame
    top5_df = pd.DataFrame(value['top5'])
    
    # Convert cash flow to DataFrame
    cashflow_df = pd.DataFrame(value['cashFlow'])
    
    return pd.DataFrame([data_dict]), top5_df, cashflow_df

def get_stock_evaluation_snapshot(ticker='MWG'):
    # https://apiextaws.tcbs.com.vn/tcanalysis/v1/evaluation/BFC/evaluation
    url = f'https://apiextaws.tcbs.com.vn/tcanalysis/v1/evaluation/{ticker}/evaluation'
    
    tcbs_config = cfg.get_config('tcbs.info')
    
    auth_token = tcbs_config.get('authToken')
    
    headers = {
        'Authorization': f'Bearer {auth_token}',
        'sec-ch-ua': '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
        'DNT': '1',
        'Accept-language': 'vi',
        'sec-ch-ua-mobile': '?0',
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome',
        'Content-type': 'application/json',
        'Accept': 'application/json',
        'Referer': 'https://tcinvest.tcbs.com.vn/',
        'sec-ch-ua-platform': '"macOS"'
    }
    
    response = requests.get(url, headers=headers)
    
    
    if response.status_code == 200:
        json = response.json()

        return json
    
    print(f"Request failed with status code {response.status_code}")
    
    return None

def get_stock_evaluation(ticker='MWG'):
    # https://apipubaws.tcbs.com.vn/tcanalysis/v1/evaluation/HDG/historical-chart?period=5&tWindow=D
    url = f'https://apipubaws.tcbs.com.vn/tcanalysis/v1/evaluation/{ticker}/historical-chart?period=5&tWindow=D'
    headers = {
        'sec-ch-ua': '"Not A(Brand";v="99", "Google Chrome";v="121", "Chromium";v="121"',
        'DNT': '1',
        'Accept-language': 'vi',
        'sec-ch-ua-mobile': '?0',
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'Referer': 'https://tcinvest.tcbs.com.vn/',
        'sec-ch-ua-platform': '"macOS"'
    }

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        json = response.json()
        return json
    else:
        print(f"Request failed with status code {response.status_code}")
        return None


def load_stock_evaluation_to_dataframe(value):
    #     "industry": "Bất động sản",
    # "medianPe": 12.9,
    # "medianPb": 1.4,
    # "medianIndexPe": 14.7,
    # "medianIndexPb": 1.7,
    # "medianIndustryPe": 15.4,
    # "medianIndustryPb": 1.3,
    # "data": [
    #     {
    #         "pe": 10.7,
    #         "pb": 1.6,
    #         "industryPe": 15.0,
    #         "industryPb": 1.4,
    #         "indexPe": 13.1,
    #         "indexPb": 1.6,
    #         "from": "2023-06-01",
    #         "to": "2023-06-01"
    #     },
    # columns = ['pe', 'pb', 'industryPe', 'industryPb', 'indexPe', 'indexPb', 'from', 'to', 'industry', 'medianPe', 'medianPb', 'medianIndexPe', 'medianIndexPb', 'medianIndustryPe', 'medianIndustryPb']
    # Convert the input data to a DataFrame
    df = pd.DataFrame(value['data'])

    # Convert 'to' and 'from' columns to datetime
    # Convert 'to' and 'from' columns to timezone-aware datetime
    df['to'] = pd.to_datetime(
        df['to'], format='%Y-%m-%d').dt.tz_localize(timezone.utc)
    df['from'] = pd.to_datetime(
        df['from'], format='%Y-%m-%d').dt.tz_localize(timezone.utc)

    # set date = to
    df['date'] = df['to']

    # Add additional columns to the DataFrame
    df['industry'] = value['industry']
    df['medianPe'] = value['medianPe']
    df['medianPb'] = value['medianPb']
    df['medianIndexPe'] = value['medianIndexPe']
    df['medianIndexPb'] = value['medianIndexPb']
    df['medianIndustryPe'] = value['medianIndustryPe']
    df['medianIndustryPb'] = value['medianIndustryPb']

    return df


def evaluate_stock_price_fundamental_factor(response_json, method="pe"):
    """
    Evaluate stock price based on the specified method.

    Parameters:
    - response_json: JSON object containing stock information.
    - method: Method for evaluation (options: "pe", "pb", "evebitda").

    Returns:
    - Stock price based on the specified method.
    """
    if method not in ["pe", "pb", "evebitda"]:
        raise ValueError(
            "Invalid evaluation method. Use 'pe', 'pb', or 'evebitda'.")

    if method == "pe":
        ratio_key = "pe"
    elif method == "pb":
        ratio_key = "pb"
    else:
        ratio_key = "evebitda"

    industry_ratio = response_json["industry"][ratio_key]
    growth_rate = response_json["growth"]
    eps = response_json["eps"]
    bvps = response_json["bvps"]
    ebitda = response_json["ebitda"]

    # Calculate stock price based on the selected method
    if method == "pe":
        stock_price = eps * industry_ratio * (1 + growth_rate)
    elif method == "pb":
        stock_price = bvps * industry_ratio * (1 + growth_rate)
    else:
        stock_price = ebitda * industry_ratio * (1 + growth_rate)

    return stock_price


def get_stock_balance_sheet(ticker='MWG', yearly=0, is_all=True):
    # https://apipubaws.tcbs.com.vn/tcanalysis/v1/finance/HDG/balancesheet?yearly=0&isAll=true
    url = f'https://apipubaws.tcbs.com.vn/tcanalysis/v1/finance/{ticker}/balancesheet'

    headers = {
        'sec-ch-ua': '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
        'DNT': '1',
        'Accept-language': 'vi',
        'sec-ch-ua-mobile': '?0',
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome',
        'Content-type': 'application/json',
        'Accept': 'application/json',
        'Referer': 'https://tcinvest.tcbs.com.vn/',
        'sec-ch-ua-platform': '"macOS"'
    }

    params = {
        'yearly': yearly,
        'isAll': is_all
    }

    response = requests.get(url, params=params, headers=headers)

    if response.status_code == 200:
        json = response.json()
        return json
    else:
        print(f"Request failed with status code {response.status_code}")
        return None


def load_stock_balance_sheet_to_dataframe(data):
    # [
    # {
    #     "ticker": "HDG",
    #     "quarter": 1,
    #     "year": 2024,
    #     "shortAsset": 3313,
    #     "cash": 578,
    #     "shortInvest": 347,
    #     "shortReceivable": 1416,
    #     "inventory": 931,
    #     "longAsset": 10947,
    #     "fixedAsset": 8913,
    #     "asset": 14260,
    #     "debt": 6767,
    #     "shortDebt": 571,
    #     "longDebt": 4743,
    #     "equity": 7493,
    #     "capital": 3058,
    #     "centralBankDeposit": null,
    #     "otherBankDeposit": null,
    #     "otherBankLoan": null,
    #     "stockInvest": null,
    #     "customerLoan": null,
    #     "badLoan": null,
    #     "provision": null,
    #     "netCustomerLoan": null,
    #     "otherAsset": null,
    #     "otherBankCredit": null,
    #     "oweOtherBank": null,
    #     "oweCentralBank": null,
    #     "valuablePaper": null,
    #     "payableInterest": null,
    #     "receivableInterest": null,
    #     "deposit": null,
    #     "otherDebt": 199,
    #     "fund": null,
    #     "unDistributedIncome": 0,
    #     "minorShareHolderProfit": 1355,
    #     "payable": 6767
    # },

    df = pd.DataFrame(data)
    df['quarter'] = df['quarter'].astype(int)
    df['year'] = df['year'].astype(int)
    df['shortAsset'] = df['shortAsset'].astype(float)
    df['cash'] = df['cash'].astype(float)
    df['shortInvest'] = df['shortInvest'].astype(float)
    df['shortReceivable'] = df['shortReceivable'].astype(float)
    df['inventory'] = df['inventory'].astype(float)
    df['longAsset'] = df['longAsset'].astype(float)
    df['fixedAsset'] = df['fixedAsset'].astype(float)
    df['asset'] = df['asset'].astype(float)
    df['debt'] = df['debt'].astype(float)
    df['shortDebt'] = df['shortDebt'].astype(float)
    df['longDebt'] = df['longDebt'].astype(float)
    df['equity'] = df['equity'].astype(float)
    df['capital'] = df['capital'].astype(float)
    df['otherDebt'] = df['otherDebt'].astype(float)
    df['unDistributedIncome'] = df['unDistributedIncome'].astype(float)
    df['minorShareHolderProfit'] = df['minorShareHolderProfit'].astype(float)
    df['payable'] = df['payable'].astype(float)
    
    # Convert 'year' to datetime at the beginning of the year
    df['date'] = pd.to_datetime(df['year'], format='%Y')
    
    # start date of the quarter
    df['start_date'] = df.apply(lambda row: pd.Timestamp(f"{row['year']}-{(row['quarter'] - 1) * 3 + 1}-01"), axis=1)
    
    # Calculate the number of months to add based on the quarter
    df['date'] = df.apply(lambda row: row['date'] + pd.DateOffset(months=(row['quarter']) * 3), axis=1)
    
    # set index
    df['index'] = df['date']
    df.set_index('index', inplace=True)

    return df


def load_stock_income_statement_to_dataframe(data):
    # [{
    #     "ticker": "VND",
    #     "quarter": 4,
    #     "year": 2024,
    #     "revenue": 1212,
    #     "yearRevenueGrowth": -0.373,
    #     "quarterRevenueGrowth": -0.046,
    #     "costOfGoodSold": -630,
    #     "grossProfit": 582,
    #     "operationExpense": -124,
    #     "operationProfit": 458,
    #     "yearOperationProfitGrowth": -0.637,
    #     "quarterOperationProfitGrowth": -0.423,
    #     "interestExpense": -182,
    #     "preTaxProfit": 275,
    #     "postTaxProfit": 251,
    #     "shareHolderIncome": 251,
    #     "yearShareHolderIncomeGrowth": -0.694,
    #     "quarterShareHolderIncomeGrowth": -0.502,
    #     "investProfit": null,
    #     "serviceProfit": null,
    #     "otherProfit": null,
    #     "provisionExpense": null,
    #     "operationIncome": null,
    #     "ebitda": 475
    # }]
    
    df = pd.DataFrame(data)
    df['quarter'] = df['quarter'].astype(int)
    df['year'] = df['year'].astype(int)
    df['revenue'] = df['revenue'].astype(float)
    df['yearRevenueGrowth'] = df['yearRevenueGrowth'].astype(float)
    df['quarterRevenueGrowth'] = df['quarterRevenueGrowth'].astype(float)
    df['costOfGoodSold'] = df['costOfGoodSold'].astype(float)
    df['grossProfit'] = df['grossProfit'].astype(float)
    df['operationExpense'] = df['operationExpense'].astype(float)
    df['operationProfit'] = df['operationProfit'].astype(float)
    df['yearOperationProfitGrowth'] = df['yearOperationProfitGrowth'].astype(float)
    df['quarterOperationProfitGrowth'] = df['quarterOperationProfitGrowth'].astype(float)
    df['interestExpense'] = df['interestExpense'].astype(float)
    df['preTaxProfit'] = df['preTaxProfit'].astype(float)
    df['postTaxProfit'] = df['postTaxProfit'].astype(float)
    df['shareHolderIncome'] = df['shareHolderIncome'].astype(float)
    df['yearShareHolderIncomeGrowth'] = df['yearShareHolderIncomeGrowth'].astype(float)
    df['quarterShareHolderIncomeGrowth'] = df['quarterShareHolderIncomeGrowth'].astype(float)
    df['ebitda'] = df['ebitda'].astype(float)
    
     # Convert 'year' to datetime at the beginning of the year
    df['date'] = pd.to_datetime(df['year'], format='%Y')
    
    # start date of the quarter
    df['start_date'] = df.apply(lambda row: pd.Timestamp(f"{row['year']}-{(row['quarter'] - 1) * 3 + 1}-01"), axis=1)
    
    # Calculate the number of months to add based on the quarter
    df['date'] = df.apply(lambda row: row['date'] + pd.DateOffset(months=(row['quarter']) * 3), axis=1)
    
    # set index
    df['index'] = df['date']
    
    # dt.tz_localize(None)
    df['index'] = df['index'].dt.tz_localize(None)
    df['start_date'] = df['start_date'].dt.tz_localize(None)
    
    df.set_index('index', inplace=True)

    return df
    

def get_stock_income_statement(ticker='MWG', yearly=0, is_all=True):
    print(f"Getting income statement for {ticker}")
    # https://apiextaws.tcbs.com.vn/tcanalysis/v1/finance/VND/incomestatement?yearly=0&isAll=true
    
    url = f'https://apiextaws.tcbs.com.vn/tcanalysis/v1/finance/{ticker}/incomestatement'
    
    config = cfg.get_config('tcbs.info')
    token = config.get('authToken')
    
    headers = {
        'Authorization': f'Bearer {token}',
        'sec-ch-ua': '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
        'DNT': '1',
        'Accept-language': 'vi',
        'sec-ch-ua-mobile': '?0',
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/',
        'Content-type': 'application/json',
        'Accept': 'application/json',
        'Referer': 'https://tcinvest.tcbs.com.vn/',
        'sec-ch-ua-platform': '"macOS"'
    }
    
    params = {
        'yearly': yearly,
        'isAll': is_all
    }
    
    response = requests.get(url, params=params, headers=headers)
    
    if response.status_code == 200:
        json = response.json()
        return json
    
    print(f"Get income statement failed with status code {response.status_code}")
    
    return None
    
def get_stock_financial(ticker='MWG', yearly=0, is_all=True):
    # https://apipubaws.tcbs.com.vn/tcanalysis/v1/finance/VHM/financialratio?yearly=0&isAll=true
    
    url = f'https://apipubaws.tcbs.com.vn/tcanalysis/v1/finance/{ticker}/financialratio'
    
    headers = {
        'sec-ch-ua': '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
        'DNT': '1',
        'Accept-language': 'vi',
        'sec-ch-ua-mobile': '?0',
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/',
        'Content-type': 'application/json',
        'Accept': 'application/json',
        'Referer': 'https://tcinvest.tcbs.com.vn/',
        'sec-ch-ua-platform': '"macOS"'
    }
    
    params = {
        'yearly': yearly,
        'isAll': is_all
    }
    
    response = requests.get(url, params=params, headers=headers)
    
    if response.status_code == 200:
        json = response.json()
        return json
    else:
        print(f"Request failed with status code {response.status_code}")
        return None
   
# Your existing load_stock_cash_flow_df function
def load_stock_cash_flow_df(data):
    df = pd.DataFrame(data)
    
    # Type conversions
    numeric_columns = ['initLiquid', 'deltaLiquid', 'deltaOtherLiquid', 'deltaReceivable', 
                      'deltaInventory', 'deltaPayble', 'initDebt', 'capex', 'dividend', 
                      'ebitda', 'tax', 'deltaDebt', 'preTax', 'depreciation', 
                      'interestExpense', 'financialInvest', 'raiseCapital', 
                      'dividendPayment', 'longAndShortDebt', 'initCash', 'endCash', 
                      'capexDebt', 'deltaLiquidDebt', 'endLiquid', 'endDebt']
    
    for col in ['quarter', 'year']:
        df[col] = df[col].astype(int)
    for col in numeric_columns:
        df[col] = df[col].astype(float)
    
    # Date handling
    # start of the year
    df['start_date'] = df.apply(lambda row: pd.Timestamp(f"{row['year']}-01-01"), axis=1)
    # end of the yeat
    df['date'] = df.apply(lambda row: row['start_date'] + pd.DateOffset(months=(row['quarter']) * 3), axis=1)
    
    df.set_index('date', inplace=True)
    return df

# Your existing get_stock_cash_flow function
def get_stock_cash_flow(ticker='MWG', yearly=1, is_all=True):
    url = f'https://apiextaws.tcbs.com.vn/tcanalysis/v1/finance/{ticker}/cashflowanalyze'
    tcbs_config = cfg.get_config('tcbs.info')
    
    auth_token = tcbs_config.get('authToken')
    
    
    headers = {
        'Authorization': f'Bearer {auth_token}',
        'sec-ch-ua': '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
        'DNT': '1',
        'Accept-language': 'vi',
        'sec-ch-ua-mobile': '?0',
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome',
        'Content-type': 'application/json',
        'Accept': 'application/json',
        'Referer': 'https://tcinvest.tcbs.com.vn/',
        'sec-ch-ua-platform': '"macOS"'
    }
    
    params = {'yearly': yearly, 'isAll': is_all}
    response = requests.get(url, params=params, headers=headers)
    
    if response.status_code == 200:
        return load_stock_cash_flow_df(response.json())
    
    print(f"Request failed with status code {response.status_code}")
    
    return None

def load_stock_financial_to_dataframe(data):
    # [
    # {
    #     "ticker": "VHM",
    #     "quarter": 1,
    #     "year": 2024,
    #     "priceToEarning": 7.7,
    #     "priceToBook": 0.9,
    #     "valueBeforeEbitda": 9.1,
    #     "dividend": null,
    #     "roe": 0.130,
    #     "roa": 0.052,
    #     "daysReceivable": 122,
    #     "daysInventory": 424,
    #     "daysPayable": 136,
    #     "ebitOnInterest": 0.8,
    #     "earningPerShare": 5073,
    #     "bookValuePerShare": 41853,
    #     "interestMargin": null,
    #     "nonInterestOnToi": null,
    #     "badDebtPercentage": null,
    #     "provisionOnBadDebt": null,
    #     "costOfFinancing": null,
    #     "equityOnTotalAsset": 0.392,
    #     "equityOnLoan": null,
    #     "costToIncome": null,
    #     "equityOnLiability": 0.7,
    #     "currentPayment": 1.2,
    #     "quickPayment": 0.9,
    #     "epsChange": -0.333,
    #     "ebitdaOnStock": 5964,
    #     "grossProfitMargin": 0.216,
    #     "operatingProfitMargin": 0.105,
    #     "postTaxMargin": 0.108,
    #     "debtOnEquity": 0.3,
    #     "debtOnAsset": 0.1,
    #     "debtOnEbitda": 2.0,
    #     "shortOnLongDebt": 0.6,
    #     "assetOnEquity": 2.5,
    #     "capitalBalance": 40321,
    #     "cashOnEquity": 0.049,
    #     "cashOnCapitalize": 0.056,
    #     "cashCirculation": 410,
    #     "revenueOnWorkCapital": 3.0,
    #     "capexOnFixedAsset": -1.928,
    #     "revenueOnAsset": 0.2,
    #     "postTaxOnPreTax": 0.6,
    #     "ebitOnRevenue": 0.105,
    #     "preTaxOnEbit": 1.6,
    #     "preProvisionOnToi": null,
    #     "postTaxOnToi": null,
    #     "loanOnEarnAsset": null,
    #     "loanOnAsset": null,
    #     "loanOnDeposit": null,
    #     "depositOnEarnAsset": null,
    #     "badDebtOnAsset": null,
    #     "liquidityOnLiability": null,
    #     "payableOnEquity": 1.4,
    #     "cancelDebt": null,
    #     "ebitdaOnStockChange": -0.137,
    #     "bookValuePerShareChange": 0.018,
    #     "creditGrowth": null
    # },
    
    df = pd.DataFrame(data)
    
    df['quarter'] = df['quarter'].astype(int)
    df['year'] = df['year'].astype(int)
    df['priceToEarning'] = df['priceToEarning'].astype(float)
    df['priceToBook'] = df['priceToBook'].astype(float)
    df['valueBeforeEbitda'] = df['valueBeforeEbitda'].astype(float)
    df['roe'] = df['roe'].astype(float)
    df['roa'] = df['roa'].astype(float)
    df['daysReceivable'] = df['daysReceivable'].astype(float)
    df['daysInventory'] = df['daysInventory'].astype(float)
    df['daysPayable'] = df['daysPayable'].astype(float)
    df['ebitOnInterest'] = df['ebitOnInterest'].astype(float)
    df['earningPerShare'] = df['earningPerShare'].astype(float)
    df['bookValuePerShare'] = df['bookValuePerShare'].astype(float)
    df['equityOnTotalAsset'] = df['equityOnTotalAsset'].astype(float)
    df['equityOnLiability'] = df['equityOnLiability'].astype(float)
    df['currentPayment'] = df['currentPayment'].astype(float)
    df['quickPayment'] = df['quickPayment'].astype(float)
    df['epsChange'] = df['epsChange'].astype(float)
    df['ebitdaOnStock'] = df['ebitdaOnStock'].astype(float)
    df['grossProfitMargin'] = df['grossProfitMargin'].astype(float)
    df['operatingProfitMargin'] = df['operatingProfitMargin'].astype(float)
    df['postTaxMargin'] = df['postTaxMargin'].astype(float)
    df['debtOnEquity'] = df['debtOnEquity'].astype(float)
    df['debtOnAsset'] = df['debtOnAsset'].astype(float)
    df['debtOnEbitda'] = df['debtOnEbitda'].astype(float)
    df['shortOnLongDebt'] = df['shortOnLongDebt'].astype(float)
    df['assetOnEquity'] = df['assetOnEquity'].astype(float)
    df['capitalBalance'] = df['capitalBalance'].astype(float)
    df['cashOnEquity'] = df['cashOnEquity'].astype(float)
    df['cashOnCapitalize'] = df['cashOnCapitalize'].astype(float)
    df['cashCirculation'] = df['cashCirculation'].astype(float)
    df['revenueOnWorkCapital'] = df['revenueOnWorkCapital'].astype(float)
    df['capexOnFixedAsset'] = df['capexOnFixedAsset'].astype(float)
    df['revenueOnAsset'] = df['revenueOnAsset'].astype(float)
    df['postTaxOnPreTax'] = df['postTaxOnPreTax'].astype(float)
    df['ebitOnRevenue'] = df['ebitOnRevenue'].astype(float)
    df['preTaxOnEbit'] = df['preTaxOnEbit'].astype(float)
    df['payableOnEquity'] = df['payableOnEquity'].astype(float)
    df['ebitdaOnStockChange'] = df['ebitdaOnStockChange'].astype(float)
    df['bookValuePerShareChange'] = df['bookValuePerShareChange'].astype(float)
    
    # Convert 'year' to datetime at the beginning of the year 2008
    df['date'] = pd.to_datetime(df['year'], format='%Y')
    
    # Calculate the number of months to add based on the quarter
    df['date'] = df.apply(lambda row: row['date'] + pd.DateOffset(months=(row['quarter']) * 3), axis=1)
    
    # convert date to datetime
    df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
    
    # set index
    df['index'] = df['date']
    df.set_index('index', inplace=True)
    
    return df

def evaluate_stock_price_dcf(response_json, discount_rate):
    """
    Evaluate stock price using the Discounted Cash Flow (DCF) method.

    Parameters:
    - response_json: JSON object containing stock information.
    - discount_rate: Discount rate used in the DCF calculation.

    Returns:
    - Stock price based on the DCF method.
    """
    free_cash_flows = [cf["freeCashFlow"] for cf in response_json["cashFlow"]]
    terminal_value = free_cash_flows[-1] * (1 + response_json["growth"]) / (
        discount_rate - response_json["growth"])

    dcf_value = 0
    for i in range(len(free_cash_flows)):
        dcf_value += free_cash_flows[i] / (1 + discount_rate) ** (i + 1)

    dcf_value += terminal_value / (1 + discount_rate) ** len(free_cash_flows)

    stock_price_dcf = dcf_value / response_json["shareOutstanding"]

    return stock_price_dcf


def evaluate_stock_price_combined(response_json, methods=["pe", "pb", "evebitda", "dcf"], discount_rate=0.1):
    """
    Evaluate stock price using a combination of methods.

    Parameters:
    - methods: List of methods to use for evaluation (options: "pe", "pb", "evebitda", "dcf").
    - discount_rate: Discount rate used in the DCF calculation.

    Returns:
    - Dictionary containing stock prices based on the specified methods.
    """
    stock_prices = {}

    for method in methods:
        if method == "dcf":
            stock_prices[method] = evaluate_stock_price_dcf(
                response_json, discount_rate)
        else:
            stock_prices[method] = evaluate_stock_price_fundamental_factor(
                response_json, method)

    # calculate the average stock price
    stock_prices["average"] = sum(stock_prices.values()) / len(stock_prices)

    return stock_prices

@retry(times=MAX_RETRIES, exceptions=(Exception), delay=RETRY_WAIT_TIME)
def get_stock_events(ticker='VND', from_date=None, to_date=None, resolution='D'):
    print(f"Getting stock events for {ticker} from {from_date} to {to_date} with resolution {resolution}")
    # https://apipubaws.tcbs.com.vn/icalendar-service/v1/event-info/trading-view?ticker=VND&from=1360288800&to=1419991200&resolution=D
    url = f'https://apipubaws.tcbs.com.vn/icalendar-service/v1/event-info/trading-view'

    headers = {
        'sec-ch-ua': '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
        'DNT': '1',
        'Accept-language': 'vi',
        'sec-ch-ua-mobile': '?0',
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/'
    }
    
    if from_date is None:
        # default from 2018-01-01
        from_date = pd.Timestamp('2018-01-01').timestamp()
    
    if to_date is None:
        # default to now
        to_date = pd.Timestamp.now().timestamp()
    
    if isinstance(from_date, datetime):
        from_date = from_date.timestamp()
        
    if isinstance(to_date, datetime):
        to_date = to_date.timestamp()
        
    params = {
        'ticker': ticker,
        'from': int(from_date),
        'to': int(to_date),
        'resolution': resolution
    }
    
    response = requests.get(url, params=params, headers=headers)
    
    if response.status_code == 200:
        json = response.json()
        data = json['data']
        print(f"Got {len(data)} events")
        return data
    else:
        print(f'Request failed for url: {response.url}')
        print(f"Request failed with status code {response.status_code} - {response.text}")
        raise Exception(f"Request failed with status code {response.status_code}")
    
    import re

def extract_dividend_amount(html_text):
    # 1. Cổ tức Cả năm/2022 bằng tiền, 500 đ/CP
    # Regular expression to extract the dividend amount bằng tiền, {anything} đ/CP => anything
    # bằng tiền, 500 đ/CP => 500
    # bằng tiền, 1.2 đ/CP => 1.2
    # # bằng tiền, any_string đ/CP => any_string (not only number)
    dividend_amount_pattern = re.compile(r"bằng tiền, ([^ ]+) đ/CP")

    # Find the dividend amount
    dividend_amount_match = dividend_amount_pattern.search(html_text)
    if dividend_amount_match:
        return dividend_amount_match.group(1)
    else:
        return None
    
def extract_stock_dividend_ratio(html_text):
    # 2. Cổ tức Bằng cổ phiếu, tỷ lệ 5.00%
    # Regular expression to extract the stock dividend ratio
    stock_dividend_ratio_pattern = re.compile(r"bằng cổ phiếu, tỷ lệ ([^%]+)%")

    # Find the stock dividend ratio
    stock_dividend_ratio_match = stock_dividend_ratio_pattern.search(html_text)
    if stock_dividend_ratio_match:
        return stock_dividend_ratio_match.group(1)
    else:
        return None
    
def extract_stock_issue_price(html_text):
    # 2. Quyền mua cổ phiếu ... giá 10.000 đ/CP ...
    if "Quyền mua" in html_text:
        # Regular expression to extract the stock issue ratio
        stock_issue_price_pattern = re.compile(r"giá ([^ ]+) đ/CP")
        stock_issue_price_match = stock_issue_price_pattern.search(html_text)
        if stock_issue_price_match:
            return stock_issue_price_match.group(1)
        else:
            return None        
    else:
        return None
        

def extract_exdividend_date(html_text):
    # Regular expression to extract the ex-dividend date
    exdividend_date_pattern = re.compile(r"Ngày GDKHQ:\s*(\d{2}-\d{2}-\d{4})")

    # Find the ex-dividend date
    exdividend_date_match = exdividend_date_pattern.search(html_text)
    if exdividend_date_match:
        return exdividend_date_match.group(1)
    else:
        return None

def load_stock_events_to_dataframe(data):
    # {"data":[{"id":"F22012014","label":"F","date":"2014-01-22T00:00:00Z","listTitle":["1. Announcement of quarterly financial statements","Ngày công bố: 22-01-2014"]}    
    # {"id":"I06112014","label":"I","date":"2014-11-06T00:00:00Z","listTitle":["1. Issue bonus shares, ratio 5.00%","Ngày GDKHQ: 06-11-2014","----------------------","2. Rights for existing shareholders with ratio 50.00%, price 10.000 VND/share","Ngày GDKHQ: 06-11-2014"]}]}
    print(f"Loading {len(data)} events")
    df = pd.DataFrame(data)
    
    if df.empty:
        return df
    
    print(f"Loaded {len(df)} {df.columns}")
    
    df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
    
    df['title'] = df['listTitle']
    
    # join listTitle to title
    df['niceTitle'] = df['title'].apply(lambda x: ', '.join(x))
        
    df['cashDividend'] = df['niceTitle'].apply(extract_dividend_amount)
    # df['exDividendDate'] = df['title'].apply(extract_exdividend_date)
    
    df['cashDividend'] = df['cashDividend'].astype(float)
    
    # if cashDividend < 10 => cashDividend = cashDividend * 1000
    df['cashDividend'] = df['cashDividend'].apply(lambda x: x if x > 10.0 else x * 1000.0)
    
    # df['exDividendDate'] = pd.to_datetime(df['exDividendDate'], format='%d-%m-%Y')
    
    # stock dividend
    # bằng cổ phiếu, tỷ lệ 5.00%
    df['stockDividend'] = df['niceTitle'].apply(extract_stock_dividend_ratio)
    
    # stock issue ratio
    df['stockIssuePrice'] = df['niceTitle'].apply(extract_stock_issue_price)
    
    df['stockIssuePrice'] = df['stockIssuePrice'].astype(float)

    df['stockDividend'] = df['stockDividend'].astype(float)

    
    # set index to date
    df.set_index('date', inplace=True)
    
    return df

def get_stock_overview(ticker='MWG'):
    # https://apipubaws.tcbs.com.vn/tcanalysis/v1/ticker/BMP/overview
    url = f'https://apipubaws.tcbs.com.vn/tcanalysis/v1/ticker/{ticker}/overview'
    
    headers = {
        'sec-ch-ua': '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
        'DNT': '1',
        'Accept-language': 'vi',
        'sec-ch-ua-mobile': '?0',
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/'
    }
    
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        json = response.json()
        return json
    else:
        print(f"Request failed with status code {response.status_code}")
        return None
    
def load_stock_overview_cached(ticker='MWG', refresh=False):
    cache_file = f'./data/overview/overview_{ticker}.json'
    
    if os.path.exists(cache_file) and not refresh:
        with open(cache_file, 'r') as f:
            data = json.load(f)
            return data
    else:
        data = get_stock_overview(ticker)
        if data:
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            with open(cache_file, 'w') as f:
                json.dump(data, f)
        return data
    
def get_stock_ratio(ticker='MWG'):
    # https://apipubaws.tcbs.com.vn/tcanalysis/v1/ticker/BMP/stockratio
    url = f'https://apipubaws.tcbs.com.vn/tcanalysis/v1/ticker/{ticker}/stockratio'
    
    headers = {
        'sec-ch-ua': '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
        'DNT': '1',
        'Accept-language': 'vi',
        'sec-ch-ua-mobile': '?0',
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/'
    }
    
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        json = response.json()
        return json
    else:
        print(f"Request failed with status code {response.status_code}")
        return None

# https://apipubaws.tcbs.com.vn/tcanalysis/v1/data-charts/vol-foreign?ticker=MWG
@retry(times=MAX_RETRIES, exceptions=(Exception), delay=RETRY_WAIT_TIME)
def get_stock_vol_foreign(ticker='MWG'):
    print(f"Getting stock vol foreign for {ticker}")
    url = f'https://apipubaws.tcbs.com.vn/tcanalysis/v1/data-charts/vol-foreign'
    
    headers = {
        'sec-ch-ua': '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
        'DNT': '1',
        'Accept-language': 'vi',
        'sec-ch-ua-mobile': '?0',
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/'
    }
    
    params = {
        'ticker': ticker
    }
    
    response = requests.get(url, params=params, headers=headers)
    
    if response.status_code == 200:
        json = response.json()
        return json
    else:
        print(f"Request failed with status code {response.status_code}")
        return None 
    
def load_stock_vol_foreign_to_dataframe(data):
    print(f"Loading stock vol foreign data")
    #  {"listVolumeForeignInfoDto":[{"ticker":"MWG","foreignBuy":799100,"foreignSell":-1971900,"netForeignVol":-1172800,"accNetFVol":-1172800,"totalVolume":9444782,"rsRank":58.0,"dateReport":"07/03/2024"},   
    list_data = data['listVolumeForeignInfoDto']
    df = pd.DataFrame(list_data)

    df['foreignBuy'] = df['foreignBuy'].astype(float)
    df['foreignSell'] = df['foreignSell'].astype(float)
    df['netForeignVol'] = df['netForeignVol'].astype(float)
    df['accNetFVol'] = df['accNetFVol'].astype(float)
    df['totalVolume'] = df['totalVolume'].astype(float)
    df['rsRank'] = df['rsRank'].astype(float)

    df['dateReport'] = pd.to_datetime(df['dateReport'], format='%d/%m/%Y')

    # set index to dateReport
    df.set_index('dateReport', inplace=True)

    return df

def get_last_trading_history(tickers, stock_type="coveredWarr"):
    dts = {}
    
    for ticker in tickers:
        res = get_stock_bars_long_term(ticker=ticker, stock_type=stock_type, count_back=1, resolution='D')
        
        if res is None:
            dts[ticker] = None
            continue
        
        data = res['data']
        
        if len(data) > 0:
            last_data = data[-1]
            dts[ticker] = last_data
        else:
            dts[ticker] = None
    df = pd.DataFrame(dts)
    
    # open, high, low, close, volume, tradingDate
    df = df.T
    
    # to numeric
    df['open'] = pd.to_numeric(df['open'])
    df['high'] = pd.to_numeric(df['high'])
    df['low'] = pd.to_numeric(df['low'])
    df['close'] = pd.to_numeric(df['close'])
    df['volume'] = pd.to_numeric(df['volume'])
    
    # drop tradingDate
    # df.drop('tradingDate', axis=1, inplace=True)
    
    return df
    
def get_last_trading_date():
    # Get the last trading date
    res = get_stock_bars_long_term(ticker='VN30', stock_type='index', count_back=1)
    
    data = res['data']
    
    last_data = data[-1]
    
    tradingDate = last_data['tradingDate'] #'2024-06-07T00:00:00.000Z'
    
    tradingDate = datetime.strptime(tradingDate, '%Y-%m-%dT%H:%M:%S.%fZ')
    
    return tradingDate
    
def get_first_trade_date_of_week():
    # Get the first trading date of the week
    res = get_stock_bars_long_term(ticker='VN30', stock_type='index',  resolution='W', count_back=1)
    
    data = res['data']
    
    first_data = data[-1]
    
    tradingDate = first_data['tradingDate'] #'2024-06-07T00:00:00.000Z'
    
    tradingDate = datetime.strptime(tradingDate, '%Y-%m-%dT%H:%M:%S.%fZ')
    
    return tradingDate

# https://apipubaws.tcbs.com.vn/stock-insight/v1/intraday/cash-flow-latest?tickers=SSI,BCM,VHM,VIC,VRE,VJC,MSN,VNM,POW,MWG,GVR,GAS,HPG,PLX,SAB,BVH,ACB,BID,HDB,MBB,SHB,SSB,STB,TCB,TPB,VCB,VIB,VPB,FPT,CTG&timeFrame=1W
@retry(times=MAX_RETRIES, exceptions=(Exception), delay=RETRY_WAIT_TIME)
def get_intraday_cash_flow_latest(tickers, timeFrame='1W'):
    url = f'https://apipubaws.tcbs.com.vn/stock-insight/v1/intraday/cash-flow-latest'
    
    headers = {
        'sec-ch-ua': '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
        'DNT': '1',
        'Accept-language': 'vi',
        'sec-ch-ua-mobile': '?0',
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/'
    }
    
    params = {
        'tickers': ','.join(tickers),
        'timeFrame': timeFrame
    }
    
    response = requests.get(url, params=params, headers=headers)
    
    if response.status_code == 200:
        json = response.json()
        return json
    else:
        print(f"Request failed with status code {response.status_code}")
        return None
    
def load_intraday_cash_flow_latest_to_dataframe(data, timeFrame='1W'):
    # {
    # "totalPage": 1,
    # "data": [
    #     {
    #         "ticker": "ACB",
    #         "comp_name": "ACB",
    #         "ind_name": "Ngân hàng",
    #         "ind_code": "8300",
    #         "min_pp": -0.016,
    #         "max_pp": -0.016,
    #         "data": [
    #             {
    #                 "bp": 0.449,
    #                 "op": 0.643,
    #                 "p": 24400.00,
    #                 "pp": -0.016,
    #                 "vol": 7679882,
    #                 "val": 187766636350,
    #                 "mc": 108986453052800.00,
    #                 "nstp": -33.400,
    #                 "peravgVolume5d": 0.76,
    #                 "rsi14": 57.48,
    #                 "rs3d": 51.00,
    #                 "avgrs": 55.00,
    #                 "pPerMa5": 1.00,
    #                 "pPerMa20": 1.01,
    #                 "isignal": 0.042,
    #                 "fnetVol": 0.00,
    #                 "t": "10/06/24",
    #                 "seq": 1717977600
    #             }
    #         ]
    #     }]}
    list_data = data['data'] if 'data' in data else data
    df = pd.DataFrame()
    
    for item in list_data:
        ticker = item['ticker']
        comp_name = item['comp_name']
        ind_name = item['ind_name']
        ind_code = item['ind_code']
        
        data = item['data']
        
        df_temp = pd.DataFrame(data)
        
        df_temp['ticker'] = ticker
        df_temp['comp_name'] = comp_name
        df_temp['ind_name'] = ind_name
        df_temp['ind_code'] = ind_code
        
        def convert_time(time):
            print(f"Converting time {time}")
            # t format could be '10/06/24', '10/06/2024 09:00" or "6-2023"
            #  10/06/2024 09:00
            if re.match(r'\d{2}/\d{2}/\d{4} \d{2}:\d{2}', time):
                return pd.to_datetime(time, format='%d/%m/%Y %H:%M')
            elif re.match(r'\d{2}/\d{2}/\d{2}', time):
                return pd.to_datetime(time, format='%d/%m/%y')
            elif re.match(r'\d{2}-\d{4}', time):
                return pd.to_datetime(time, format='%m-%Y')
            else:
                return time
                
        
        df_temp['t'] = df_temp['t'].apply(convert_time)
        
                    
        df = pd.concat([df, df_temp])    
    
    # # Renaming the columns
    # df.rename(columns={
    #     'ticker': 'ticker',
    #     'comp_name': 'comp_name',
    #     'ind_name': 'ind_name',
    #     'ind_code': 'ind_code',
    #     'min_pp': 'min_pp',
    #     'max_pp': 'max_pp',
    #     'bp': 'buyup_pct',
    #     'op': 'open_pct',
    #     'p': 'price',
    #     'pp': 'price_change_pct',
    #     'vol': 'volume',
    #     'val': 'value',
    #     'mc': 'market_cap',
    #     'nstp': 'performance_indicator',
    #     'peravgVolume5d': 'average_volume_5d',
    #     'rsi14': 'RSI (14)',
    #     'rs3d': 'RSI 3d',
    #     'avgrs': 'average_RS',
    #     'pPerMa5': 'price/MA(5)',
    #     'pPerMa20': 'price/MA(20)',
    #     'isignal': 'iSignal',
    #     'fnetVol': 'foreign_net_volume',
    #     't': 'date',
    #     'seq': 'sequence'
    # }, inplace=True)
    
    # set date  = t
    df['date'] = df['t']
    
    if timeFrame == '1M':
        # convert 08-2023 to end of month
        df['date'] = df['date'].apply(lambda x: pd.to_datetime(x) + pd.offsets.MonthEnd(0))
    
        
    # set index to date
    df.set_index('date', inplace=True)
    
    return df
        
        
# https://apipubaws.tcbs.com.vn/stock-insight/v1/intraday/cash-flow-list?tickers=SSI,BCM,VHM,VIC,VRE,VJC,MSN,VNM,POW,MWG,GVR,GAS,HPG,PLX,SAB,BVH,ACB,BID,HDB,MBB,SHB,SSB,STB,TCB,TPB,VCB,VIB,VPB,FPT,CTG&timeFrame=1D&page=0
def get_intraday_cash_flow_list(tickers, timeFrame='1D', page=0):
    url = f'https://apipubaws.tcbs.com.vn/stock-insight/v1/intraday/cash-flow-list'
    
    headers = {
        'sec-ch-ua': '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
        'DNT': '1',
        'Accept-language': 'vi',
        'sec-ch-ua-mobile': '?0',
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/'
    }
    
    params = {
        'tickers': ','.join(tickers),
        'timeFrame': timeFrame,
        'page': page
    }
    
    response = requests.get(url, params=params, headers=headers)
    
    if response.status_code == 200:
        json = response.json()
        return json
    else:
        print(f"Request failed with status code {response.status_code}")
        return None

def get_intraday_cash_flow_list_all(tickers, timeFrame='1D'):
    page = 0
    res = get_intraday_cash_flow_list(tickers, timeFrame, page)
    
    data = res['data']
    
    totalPage = res['totalPage']
    
    while page < totalPage:
        page += 1
        res = get_intraday_cash_flow_list(tickers, timeFrame, page)
        
        data += res['data']
        
    return data

# https://apipubaws.tcbs.com.vn/stock-insight/v1/stock/intraday-snapshots?tickers=A32,AAH,AAS,ABB,ABC,ABI,ABW,ACE,ACM,ACS,ACV,AFX,AG1,AGF,AGP,AGX,AIC,ALV,AMD,AMP,AMS,ANT,APC,APF,APL,APP,APT,ART,ASA,ATA,ATB,ATG,AVC,AVF,B82,BAL,BBH,BBM,BBT,BCA,BCB,BCO,BCP,BCR,BCV,BDG,BDT,BDW,BEL,BGW
@retry(times=MAX_RETRIES, exceptions=(Exception), delay=RETRY_WAIT_TIME)
def get_intraday_snapshots(tickers):
    print(f"Getting intraday snapshots for {len(tickers)} tickers")
    url = f'https://apipubaws.tcbs.com.vn/stock-insight/v1/stock/intraday-snapshots'
    
    headers = {
        'sec-ch-ua': '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
        'DNT': '1',
        'Accept-language': 'vi',
        'sec-ch-ua-mobile': '?0',
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/'
    }
    
    params = {
        'tickers': ','.join(tickers)
    }
    
    response = requests.get(url, params=params, headers=headers)
    
    if response.status_code == 200:
        json = response.json()
        data = json['data']

        return data
    else:
        print(f"Request failed with status code {response.status_code}")
        return None

def load_intraday_snapshots_to_dataframe(data):
    # {
    #     "NTP": {
    #         "startTrad": 1718330400,
    #         "data": [
    #             [
    #                 62800.0,
    #                 17500.0,
    #                 0.0
    #             ],
    #             [
    #                 62800.0,
    #                 2300.0,
    #                 3.0
    #             ],
    #             [
    #                 63800.0,
    #                 24800.0,
    #                 6.0
    #             ],
    #     }
    # }
    stocks_dfs = {}
    
    for ticker, item in data.items():
        startTrad = item['startTrad']
        data = item['data']
        
        df_temp = pd.DataFrame(data, columns=['price', 'volume', 'trades'])
        
        df_temp['ticker'] = ticker
        df_temp['startTrad'] = startTrad
        
        # convert startTrad to datetime
        df_temp['startTrad'] = pd.to_datetime(df_temp['startTrad'], unit='s')
        
        # set index to startTrad + interval 5 minutes
        df_temp['index'] = df_temp['startTrad'] + pd.to_timedelta(df_temp.index * 5, unit='m')
        
        # set time = time + 7 hours
        df_temp['index'] = df_temp['index'] + pd.to_timedelta(7, unit='h')
        
        # when hour >= 12 hour = hour + 1
        df_temp['index'] = df_temp['index'].apply(lambda x: x + pd.to_timedelta(1, unit='h') if x.hour >= 12 else x)
        
        df_temp.set_index('index', inplace=True)

        stocks_dfs[ticker] = df_temp
        
    factor_dfs = {}
    for symbol in stocks_dfs:
        for column in stocks_dfs[symbol].columns:
            if column not in factor_dfs:
                factor_dfs[column] = pd.DataFrame()
            factor_dfs[column][symbol] = stocks_dfs[symbol][column]

    stocks_df = pd.concat(factor_dfs, axis=1)
    
    # drop date column
    # stocks_df = stocks_df.drop(columns='date')
        
    return stocks_df

def get_intraday_snapshots_all(tickers):
    print(f"Getting intraday snapshots for {len(tickers)} tickers")
    # break the tickers into chunks of 50
    chunk_size = 50
    chunks = [tickers[i:i + chunk_size] for i in range(0, len(tickers), chunk_size)]
    
    data = {}
    
    for chunk in chunks:
        res = get_intraday_snapshots(chunk)
        data.update(res)
    
    df = load_intraday_snapshots_to_dataframe(data)
    
    return df    


def get_all_industries():
    overview_df = pd.read_csv("data/stock_overview.csv", index_col=0)
        
    return overview_df['industryEn'].unique()

def filter_stocks_by_industry(stocks_df, industry):
    overview_df = pd.read_csv("data/stock_overview.csv", index_col=0)
    
    tickers = overview_df[overview_df['industryEn'] == industry].index
    
    # Filter only the tickers that exist in the stocks_df columns
    tickers = [ticker for ticker in tickers if ticker in stocks_df.columns]
    
    return stocks_df[tickers]

def construct_index_df(stocks_df):
    # create custom index for stocks_df price data
    index_df = stocks_df.mean(axis=1)
    
    return index_df
    
def construct_multi_index_df(stocks_df, industries):
    multi_index_dfs = {}
    
    for industry in industries:
        stocks_industry_df = filter_stocks_by_industry(stocks_df, industry)
        index_df = construct_index_df(stocks_industry_df)
        
        multi_index_dfs[industry] = index_df
        
    multi_index_df = pd.concat(multi_index_dfs, axis=1)
        
    return multi_index_df

def calculate_price_changes(stocks_df, news_df, lower_bound=-4, upper_bound=2):
    stocks_df = stocks_df.reindex(news_df.index, method='ffill')
    
    price_change_dfs = {}
    
    for i in range(lower_bound, upper_bound):
        price_change_df = (stocks_df.shift(i) - stocks_df) / stocks_df
        
        # set type to float
        price_change_df = price_change_df.astype(float)
        
        price_change_dfs[f"change_{i}"] = price_change_df
        
    price_changes_df = pd.concat(price_change_dfs, axis=1)
    price_changes_df.index = news_df.index
    
    price_changes_flat_df = price_changes_df.stack().reset_index()
    price_changes_flat_df = price_changes_flat_df.set_index('level_0')
    
    return price_changes_flat_df