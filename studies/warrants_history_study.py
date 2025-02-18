import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import utils.algotrade as at

from studies.stock_news_momentum_study import calculate_price_changes
from studies.stock_overnight_study import plot_scatter_2_sources
import utils.plot_utils as pu
from utils.processing import get_stocks, get_stocks_foregin_flow
import utils.stock_utils as su

import plotly.graph_objects as go
import streamlit as st
import requests
import pandas as pd
from bs4 import BeautifulSoup
import numpy as np
from studies.stock_gaps_recover_study import run as stock_gaps_recover_study

import numpy as np
import pandas as pd

import numpy as np
import pandas as pd

def parse_warrant_news(text):
    try:
        dfs = pd.read_html(text, header=0)
        
        if len(dfs) < 5:
            return None
        
        df = dfs[4]
        
        # trim all columns
        df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
        # remove ":" from columns
        df.columns = df.columns.str.replace(':', '')

        # convert to dict, column 1 is the header, column 2 is the value
        dct = dict(zip(df.iloc[:, 0].str.replace(':', ''), df.iloc[:, 1]))
        
        info = {}
        
        # rename columns
        #  CK cơ sở,Tổ chức phát hành CKCS,Tổ chức phát hành CW,Loại chứng quyền,Kiểu thực hiện,Phương thức thực hiện quyền,Thời hạn,Ngày phát hành,Ngày niêm yết,Ngày giao dịch đầu tiên,Ngày giao dịch cuối cùng,Ngày đáo hạn,Tỷ lệ chuyển đổi,TLCĐ điều chỉnh,Giá phát hành,Giá thực hiện,Giá TH điều chỉnh,Khối lượng Niêm yết,Khối lượng lưu hành
        info['stock'] = dct.get('CK cơ sở', None)
        info['issuer'] = dct.get('Tổ chức phát hành CKCS', None)
        info['cw_issuer'] = dct.get('Tổ chức phát hành CW', None)
        info['type'] = dct.get('Loại chứng quyền', None)
        info['exercise_type'] = dct.get('Kiểu thực hiện', None)
        info['exercise_method'] = dct.get('Phương thức thực hiện quyền', None)
        info['term'] = dct.get('Thời hạn', None)
        info['issue_date'] = dct.get('Ngày phát hành', None)
        info['listing_date'] = dct.get('Ngày niêm yết', None)
        info['first_trade_date'] = dct.get('Ngày giao dịch đầu tiên', None)
        info['last_trade_date'] = dct.get('Ngày giao dịch cuối cùng', None)
        info['maturity_date'] = dct.get('Ngày đáo hạn', None)
        info['conversion_ratio'] = dct.get('Tỷ lệ chuyển đổi', None)
        info['adjusted_conversion_ratio'] = dct.get('TLCĐ điều chỉnh', None)
        info['issue_price'] = dct.get('Giá phát hành', None)
        info['exercise_price'] = dct.get('Giá thực hiện', None)
        info['adjusted_exercise_price'] = dct.get('Giá TH điều chỉnh', None)
        info['listed_volume'] = dct.get('Khối lượng Niêm yết', None)
        info['circulating_volume'] = dct.get('Khối lượng lưu hành', None)
        
        
        # conversion_ratio = adjusted_conversion_ratio if not null else conversion_ratio
        if 'adjusted_conversion_ratio' in info and not pd.isnull(info['adjusted_conversion_ratio']):
            info['conversion_ratio'] = info['adjusted_conversion_ratio']
        
        # exercise_price = adjusted_exercise_price if not null else exercise_price
        if 'adjusted_exercise_price' in info and not pd.isnull(info['adjusted_exercise_price']):
            info['exercise_price'] = info['adjusted_exercise_price']
            
        # issue_price = adjusted_issue_price if not null else issue_price
        if 'adjusted_issue_price' in info and not pd.isnull(info['adjusted_issue_price']):
            info['exercise_price'] = info['adjusted_issue_price']
        
                
        df = pd.DataFrame([info])
        
        # convert to datetime
        for col in ['issue_date', 'listing_date', 'first_trade_date', 'last_trade_date', 'maturity_date']:
            df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # if all empty
        if df.isnull().values.all():
            return None
        
        return df
    except Exception as e:
        st.write(e)
        return None


def fetch_warrant_exta_info(ticker='ACB'):
    # https://finance.vietstock.vn/chung-khoan-phai-sinh/CFPT2001/cw-thong-ke-giao-dich.htm
    
    # fetch the data
    url = f'https://finance.vietstock.vn/chung-khoan-phai-sinh/{ticker}/cw-thong-ke-giao-dich.htm'
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }
    
    res = requests.get(url, headers=headers)
    
    if res.status_code != 200:
        return None
    
    soup = BeautifulSoup(res.text, 'html.parser')
   
    # find the all p.i-b
    # <p class="i-b">+/- Niêm yết<span class="pull-right txt-red">-95.19%</span></p>
    
    pibb = soup.find_all('p', class_='i-b')
    
    info = {}
    
    for p in pibb:
        # check if it contains +/- Niêm yết
        if '+/- Niêm yết' in p.text:
            listing_change = p.find('span').text
            # -95.19% -> -95.19
            listing_change = float(listing_change.replace('%', ''))
            info['listing_change'] = listing_change
    
    return info            

def fetch_warrant_price_history(ticker='ACB'):
    # https://finance.vietstock.vn/chung-khoan-phai-sinh/CFPT2403/cw-blackschole.htm
    
    # fetch the data
    url = f'https://finance.vietstock.vn/chung-khoan-phai-sinh/{ticker}/cw-blackschole.htm'
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }
    
    res = requests.get(url, headers=headers)
    
    if res.status_code != 200:
        return None
    
    # pandas read_html
    dfs = pd.read_html(res.text)
    
    st.write(dfs)
    
    if len(dfs) < 4:
        return None
    
    df = dfs[3]
    
    # trim all columns
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    
    return df

def fetch_warrant_news(ticker='ACB'):
    # https://finance.vietstock.vn/chung-khoan-phai-sinh/CFPT2001/cw-tong-quan.htm
    
    # fetch the data
    url = f'https://finance.vietstock.vn/chung-khoan-phai-sinh/{ticker}/cw-tong-quan.htm'
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }
    
    res = requests.get(url, headers=headers)
        
    if res.status_code != 200:
        return None
    
    df = parse_warrant_news(res.text)
    
    if df is None:
        return None

    # add index column as ticker
    df['ticker'] = ticker
    
    # set index
    df.set_index('ticker', inplace=True)
    
    return df


@st.cache_data
def reload_warrant_news(ticker='ACB'):
    # first we contruct a posible list of tickers
    # C + ticker + YY + MM
    tickers = []
    start_year = 20
    end_year = 25
    for y in range(start_year, end_year):
        for m in range(1, 13):
            tickers.append(f'C{ticker}{y}{m:02d}')
            
    # trying to fetch
    infos = pd.DataFrame()
    
    for t in tickers:
        news = fetch_warrant_news(t)
        ex_info = fetch_warrant_exta_info(t)
        if news is not None:
            # if infos is empty
            # use first row as header
            new_row = news
            new_row['listing_change'] = ex_info['listing_change'] if ex_info is not None and 'listing_change' in ex_info else np.nan
            infos = pd.concat([infos, new_row], axis=0)
        
    # save to csv
    infos.to_csv(f'data/warrant_news_{ticker}.csv')
    
    st.write('Done')
    
def run(symbol_benchmark, symbolsDate_dict):
    if len(symbolsDate_dict['symbols']) < 1:
        st.info("Please select symbols.")
        st.stop()
    
    ticker = symbolsDate_dict['symbols'][0]

    
    # button to reload
    if st.button('Reload'):
        reload_warrant_news(ticker)
        
    use_cache = st.checkbox('Use cache', value=True)
    
    t = fetch_warrant_price_history('CFPT2001')
    st.write(t)
    st.stop()

    
    df = None
    if use_cache:
        try:
            df = pd.read_csv(f'data/warrant_news_{ticker}.csv')
            df.set_index('ticker', inplace=True)
        except:
            st.write('Cache not found')
            st.stop()
            
    if df is None:
        st.write('Cache not found')
        st.stop()
                
    stocks_df = get_stocks(symbolsDate_dict, 'close')
    
    stock_df = stocks_df[ticker]
    
    # plot stock
    pu.plot_single_line(stock_df, title=f'{ticker} price')
    

    for row in df.iterrows():        
        if row[1]['issue_date'] is not pd.NaT:
            issue_stock_price = stock_df.loc[:row[1]['issue_date']].ffill().iloc[-1]
            df.loc[row[0], 'issue_stock_price'] = issue_stock_price
        
        if row[1]['maturity_date'] is not pd.NaT:
            exercise_stock_price = stock_df.loc[:row[1]['maturity_date']].ffill().iloc[-1]
            df.loc[row[0], 'exercise_stock_price'] = exercise_stock_price
        
        if row[1]['last_trade_date'] is not pd.NaT:
            last_stock_price = stock_df.loc[:row[1]['last_trade_date']].ffill().iloc[-1]
            df.loc[row[0], 'last_stock_price'] = last_stock_price
        
        if row[1]['first_trade_date'] is not pd.NaT:
            first_stock_price = stock_df.loc[:row[1]['first_trade_date']].ffill().iloc[-1]
            df.loc[row[0], 'first_stock_price'] = first_stock_price
    # caculate break even price
    df['out_of_money'] = df['exercise_stock_price'] - df['exercise_price']
    
    st.write(df)
    
    pu.plot_single_bar(df['out_of_money'], title='Out of money', x_title='Warrant', y_title='Out of money', legend_title='Out of money')
    
    pu.plot_single_bar(df['listing_change'], title='Listing change', x_title='Warrant', y_title='Listing change', legend_title='Listing change')
    
    
    listing_change_by_maturity = df.groupby('maturity_date')['listing_change'].mean()
     
  
    pu.plot_single_bar(listing_change_by_maturity, title='Listing change by maturity', x_title='Maturity date', y_title='Listing change', legend_title='Listing change')
    
    # price in range of listing change
    price_df = stock_df.loc[df['listing_date'].min():df['listing_date'].max()]
    
    pu.plot_single_line(price_df, title=f'{ticker} price', x_title='Date', y_title='Price', legend_title='Price')
  
  
    # listing change by cw_issuer
  
    listing_change_by_cw_issuer = df.groupby('cw_issuer')['listing_change'].mean()
    
    pu.plot_single_bar(listing_change_by_cw_issuer, title='Listing change by cw issuer', x_title='CW Issuer', y_title='Listing change', legend_title='Listing change')
    
    # listing change by term
    
    listing_change_by_term = df.groupby('term')['listing_change'].mean()
    
    pu.plot_single_bar(listing_change_by_term, title='Listing change by term', x_title='Term', y_title='Listing change', legend_title='Listing change')