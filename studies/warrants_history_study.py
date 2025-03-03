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
import re
from bs4 import BeautifulSoup
import numpy as np
from studies.stock_gaps_recover_study import run as stock_gaps_recover_study

import numpy as np
import pandas as pd

import numpy as np
import pandas as pd

from .warrants_exp_value_study import backtest_trade_cw_simulation, simulate_warrant, simulate_warrant_statical

def parse_warrant_news(text):
    try:
        dfs = pd.read_html(text, header=0)
        
        if len(dfs) < 5:
            return None
        
        df = None
        
        # find one that contains "CK cơ sở" in the first column
        for d in dfs:
            if any('CK cơ sở:' in col for col in d.columns):
                df = d
                break
        
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

def dict_to_form_data(data):
    form_data = ''
    for key, value in data.items():
        form_data += f'{key}={value}&'
        
    return form_data[:-1]

def get_vietstock_preface(ticker='ACB'):
    # https://finance.vietstock.vn/chung-khoan-phai-sinh/CFPT2001/cw-blackschole.htm
    
    # fetch the data
    url = f'https://finance.vietstock.vn/chung-khoan-phai-sinh/{ticker}/cw-blackschole.htm'
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }
    
    res = requests.get(url, headers=headers)
    
    if res.status_code != 200:
        return None
    
    soup = BeautifulSoup(res.text, 'html.parser')
    
    token = soup.find('input', {'name': '__RequestVerificationToken'})['value']
        
    # script contain "function callPrice() { ... }"
    form_script = soup.find('script', text=lambda x: x and 'function callPrice()' in x)
    
    if form_script is None:
        return None
    
    # use regex to extract infmation
    
    # extract the irate
    # if (irate == '' || irate == '0') irate = 5.5; => 5.5
    irate = re.search(r"if \(irate == '' \|\| irate == '0'\) irate = (\d+\.\d+);", form_script.text)
    irate = irate.group(1) if irate is not None else 4.5
    
    # extract the irate
    # if (price == '' || price == '0') price = 135000; => 135000
    
    # extract the exprice
    # if (exprice == '' || exprice == '0') exprice = 56000; => 56000
    exprice = re.search(r"if \(exprice == '' \|\| exprice == '0'\) exprice = (\d+);", form_script.text)
    exprice = exprice.group(1) if exprice is not None else 56000
    
    # extract the crate
    # if (crate == '' || crate == '0') crate = 5, 000; => 5
    crate = re.search(r"if \(crate == '' \|\| crate == '0'\) crate = (\d+),", form_script.text)
    crate = crate.group(1) if crate is not None else 5
    
    #  extract the tradedate
    # _tradedate='2020-06-18'; => 2020-06-18
    tradedate = re.search(r"_tradedate='(\d+-\d+-\d+)';", form_script.text)
    tradedate = tradedate.group(1) if tradedate is not None else '2020-06-18'
    
    # cookie
    cookie = res.headers['Set-Cookie']
    
    preface = {
        '__RequestVerificationToken': 'CTWbT4LpOnRJnHcSP3UojC9TKCTkOhEZp82XvTs07BGtLsVkVmqfTYFJ66bRFLEqDDUwk3z0UHjpcmffjRxMbLVu3Tt3eAEMZ-dO-vqNoDw1',
        'interestRate': irate,
        'tradeDate': tradedate,
        'price': exprice,
        'conversionRate': crate,
        'code': ticker,
        'cookie': '__gads=ID=423cc3950e82f39a:T=1726806150:RT=1726806150:S=ALNI_MbUOkE1uK9lBAAwopMVo-n-c24MIw; __gpi=UID=00000ef5802bdff8:T=1726806150:RT=1726806150:S=ALNI_MbbTkFr6YSzw6knlR2o8r1h4YqR5Q; __eoi=ID=ac4cf7895cd620ab:T=1726806150:RT=1726806150:S=AA-AfjYzuzpVkiByT_gFJk9wpfiF; language=vi-VN; Theme=Light; AnonymousNotification=; ASP.NET_SessionId=uifnmb2w2siidkionk3qe3zh; finance_viewedstock=CFPT2403,; __RequestVerificationToken=dqR2XrTNMaMG7ExBHlcQ8mjO19q0lLqmEiaFekMJKodMjMVolNYHh93yYHdz6b7bcCvlJtflD9xAZm2be4xjKHWthz8OcelQSgNQrsLfu1I1'
    }
    
    return preface

def fetch_warrant_price_history(ticker='ACB'):
    # https://finance.vietstock.vn/data/CallCWBlackSchole
    
    preface = get_vietstock_preface(ticker)
    
    if preface is None:
        return None
    
    url = "https://finance.vietstock.vn/data/CallCWBlackSchole"
    
    data = {
        'code': ticker,
        'interestRate': preface['interestRate'],
        'tradeDate': preface['tradeDate'],
        'price': preface['price'],
        'conversionRate': preface['conversionRate'],
        '__RequestVerificationToken': preface['__RequestVerificationToken']
    }
    

    # payload = "code=CFPT2403&interestRate=4.5&tradeDate=2025-02-18&price=135000&conversionRate=4&__RequestVerificationToken=CTWbT4LpOnRJnHcSP3UojC9TKCTkOhEZp82XvTs07BGtLsVkVmqfTYFJ66bRFLEqDDUwk3z0UHjpcmffjRxMbLVu3Tt3eAEMZ-dO-vqNoDw1"
    payload = dict_to_form_data(data)

    headers = {
        'Accept': '*/*',
        'Accept-Language': 'en-US,en;q=0.9,vi;q=0.8,vi-VN;q=0.7,fr-FR;q=0.6,fr;q=0.5,de;q=0.4',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
        'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
        'Cookie': preface['cookie'],
        'DNT': '1',
        'Origin': 'https://finance.vietstock.vn',
        'Pragma': 'no-cache',
        'Referer': 'https://finance.vietstock.vn/chung-khoan-phai-sinh/CFPT2403/cw-blackschole.htm',
        'Sec-Fetch-Dest': 'empty',
        'Sec-Fetch-Mode': 'cors',
        'Sec-Fetch-Site': 'same-origin',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36',
        'X-Requested-With': 'XMLHttpRequest',
        'sec-ch-ua': '"Not A(Brand";v="8", "Chromium";v="132", "Google Chrome";v="132"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"Windows"'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
        
    df = pd.read_json(response.text)
    
    if df.empty:
        return None
    
    df['TradingDate'] = df['TradingDate'].apply(lambda x: pd.to_datetime(int(re.search(r'/Date\((\d+)\)/', x).group(1)), unit='ms'))
    df['FirstTradingDate'] = df['FirstTradingDate'].apply(lambda x: pd.to_datetime(int(re.search(r'/Date\((\d+)\)/', x).group(1)), unit='ms'))
    df['DueDate'] = df['DueDate'].apply(lambda x: pd.to_datetime(int(re.search(r'/Date\((\d+)\)/', x).group(1)), unit='ms'))
    
    df['ticker'] = ticker
    
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
    start_year = 19
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

@st.cache_data
def reload_warrant_price_history(ticker='ACB'):
    # first we contruct a posible list of tickers
    # C + ticker + YY + MM
    tickers = []
    start_year = 19
    end_year = 25
    for y in range(start_year, end_year):
        for m in range(1, 13):
            tickers.append(f'C{ticker}{y}{m:02d}')
            
    # trying to fetch
    infos = pd.DataFrame()
    
    for t in tickers:
        price = fetch_warrant_price_history(t)
        if price is not None:
            infos = pd.concat([infos, price], axis=0)
        
    # save to csv
    infos.to_csv(f'data/warrant_price_{ticker}.csv')
    
    st.write('Done')
    
def load_full_warrants_history(ticker='ACB'):
    df = None

    try:
        df = pd.read_csv(f'data/warrant_news_{ticker}.csv')
        df.set_index('ticker', inplace=True)
    except:
        st.write('Cache not found')
        st.stop()
            
    if df is None:
        st.write('Cache not found')
        st.stop()
    
    price_df = None
    
    try:
        price_df = pd.read_csv(f'data/warrant_price_{ticker}.csv')
    except:
        st.write('Cache not found')
        st.stop()
        
    # st.write(df)
    
    # pu.plot_single_bar(df['out_of_money'], title='Out of money', x_title='Warrant', y_title='Out of money', legend_title='Out of money')
    
    # remove the first column unnamed 0
    price_df = price_df.loc[:, ~price_df.columns.str.contains('^Unnamed')]
    
   # merge the price_df with df on ticker

    full_df = pd.merge(df, price_df, left_index=True, right_on='ticker', how='left')
    
    # st.write(full_df)
    
    # convert TradingDate str '2020-06-17 17:00:00' -> '2020-06-17 00:00:00'
    # 17:00:00 -> 00:00:00
    # st.write(full_df)
    full_df['TradingDate'] = full_df['TradingDate'].apply(lambda x: x.split()[0] if isinstance(x, str) else x)
    full_df['TradingDate'] = pd.to_datetime(full_df['TradingDate'])
    full_df['stock'] = full_df['BaseStockCode']
    full_df['close_stock'] = full_df['BaseClosePrice']
    full_df['days_to_expired'] = full_df['RemainDays']
    full_df['listing_change'] = full_df['listing_change'].astype(float)
    # break_even_price = su.warrant_break_even_point(cw_price, df['Exercise_Price'][i], df['Exercise_Ratio'][i])
    # ticker,stock,issuer,cw_issuer,type,exercise_type,
    # exercise_method,term,issue_date,listing_date,first_trade_date,
    # last_trade_date,maturity_date,conversion_ratio,adjusted_conversion_ratio,issue_price,exercise_price,
    # adjusted_exercise_price,listed_volume,circulating_volume,listing_change
        # index as datetime
    full_df['listing_date'] = pd.to_datetime(full_df['listing_date'])
    full_df['maturity_date'] = pd.to_datetime(full_df['maturity_date'])
    full_df['first_trade_date'] = pd.to_datetime(full_df['FirstTradingDate'])
    
    full_df['issue_date'] = pd.to_datetime(full_df['issue_date'])
    full_df['days_to_listing'] = (full_df['TradingDate'] - full_df['first_trade_date']).dt.days
    full_df['Exercise_Price'] = full_df['exercise_price'].astype(float)
    full_df['close_cw'] = full_df['ClosePrice'].astype(float)
    full_df['price_to_issue'] = (full_df['close_cw'] - full_df['issue_price']) / full_df['issue_price']
    # 2.3:1 -> 2.3
    full_df['Exercise_Ratio'] = full_df['conversion_ratio'].apply(lambda x: float(x.split(':')[0]))
    full_df['break_even_price'] = su.warrant_break_even_point(full_df['close_cw'], full_df['Exercise_Price'], full_df['Exercise_Ratio'])
    
    # premium = (stock price - break_even_price) / break_even_price
    full_df['premium'] = (full_df['close_stock'] - full_df['break_even_price']) / full_df['break_even_price']
    
    # sort by TradingDate
    full_df.sort_values('TradingDate', inplace=True)
    
    return full_df
    
def run(symbol_benchmark, symbolsDate_dict):
    if len(symbolsDate_dict['symbols']) < 1:
        st.info("Please select symbols.")
        st.stop()
    
    ticker = symbolsDate_dict['symbols'][0]
    
    # button to reload
    if st.button('Reload Info'):
        reload_warrant_news(ticker)
        
    if st.button('Reload price'):
        reload_warrant_price_history(ticker)
        
    full_df = load_full_warrants_history(ticker)
    
    # drop ticker CFPT1908
    full_df = full_df[full_df['ticker'] != 'CFPT1908']
    
    full_df['listing_change_cal'] = full_df['close_cw'] / full_df['issue_price'] - 1
    
    listing_change_df = full_df[['listing_change_cal', 'TradingDate', 'ticker']]
    
    # reshape the data each column is a ticker
    listing_change_df = listing_change_df.pivot(index='TradingDate', columns='ticker', values='listing_change_cal')
    
    # convert to percentage acc return
    listing_change_acc_df = listing_change_df
    
    # set all nan of listing_change_acc_df to nan of listing_change_df nan
    listing_change_acc_df = listing_change_acc_df.where(~listing_change_df.isna())
    
    # index as datetime
    listing_change_acc_df.index = pd.to_datetime(listing_change_acc_df.index)
    
    # fill na with 0
    # listing_change_df.fillna(0, inplace=True)
    
    # st.write(listing_change_df)
    
    stocks_df = get_stocks(symbolsDate_dict, 'close')
        
    stock_df = stocks_df[[ticker]]
    stock_df.columns = ['close']
    
    # allign the stock_df with index > listing_change_acc_df.index < stock_df.index
    stock_align_df = stock_df[stock_df.index.isin(listing_change_acc_df.index)]
    stock_acc_return = stock_align_df.pct_change().cumsum()
    
    stock_acc_return.columns = ['close']
    
    # merge with listing_change_df
    listing_change_acc_with_stock_df = pd.merge(stock_acc_return, listing_change_acc_df, left_index=True, right_index=True, how='left')
    
    # plit multi line
    pu.plot_multi_line(listing_change_acc_with_stock_df, title='Listing Change', x_title='Date', y_title='Listing Change', legend_title='Ticker')
    
    # index to datetime
    stock_df.index = pd.to_datetime(stock_df.index)
    
    # full_df = full_df[full_df['ticker'] == 'CFPT2201']    
    
    listing_change_by_day_df = full_df[['listing_change_cal', 'days_to_listing', 'ticker']]
    
        
    # reshape the data each column is a ticker
    listing_change_by_day_df = listing_change_by_day_df.pivot(index='days_to_listing', columns='ticker', values='listing_change_cal')
    
    # sort by days_to_listing
    listing_change_by_day_df.sort_index(inplace=True)
            
    # convert to percentage acc return
    listing_change_by_day_acc_df = listing_change_by_day_df
    
    # pu.plot_multi_line(listing_change_by_day_df, title='Listing Change by Day', x_title='Days to Expired', y_title='Listing Change', legend_title='Ticker')
    
    # set all nan of listing_change_acc_df to nan of listing_change_df nan
    listing_change_by_day_acc_df = listing_change_by_day_acc_df.where(~listing_change_by_day_df.isna())
    
    pu.plot_multi_line(listing_change_by_day_acc_df, title='Listing Change by Day', x_title='Days to Expired', y_title='Listing Change', legend_title='Ticker')
    
    
    # plot stock
    pu.plot_single_line(stock_df['close'], title=f'{ticker} price')
    
    all_tickers = full_df['ticker'].unique()
    
    all_simulate_df = pd.DataFrame()
    if st.button('Run Simulation'):        
        for selected_ticker in all_tickers:
            selected_df = full_df[full_df['ticker'] == selected_ticker]
            
            # reindex to TradingDate
            selected_df.set_index('TradingDate', inplace=True)
            
            # result_df = simulate_warrant(stock_df, selected_df, 252)
            result_df = simulate_warrant_statical(stock_df, selected_df, 252)
            result_df['ticker'] = selected_ticker
            # cap = 100
            
            # cap expect_value to -cap +cap
            # result_df['expect_value'] = result_df['expect_value'].clip(-cap, cap)
            final_listing_change = result_df['listing_change'].iloc[-1]
            
            result_df['price_ratio'] = result_df['close_cw'] / result_df['close_stock']
            # pu.plot_single_bar(result_df['price_ratio'], title=f"Price Ratio {selected_ticker}")
            
            result_df['expect_value_change'] = result_df['expect_value'].pct_change()
            # pu.plot_single_bar(result_df['expect_value_change'], title=f"Expected Value Change {selected_ticker}")
            
            result_df['expect_value_change_acc'] = result_df['expect_value'].pct_change().cumsum()
            
            first_expect_value = result_df['expect_value'].iloc[0]
            
            result_df['expect_value_change_listing'] = result_df['expect_value'] / first_expect_value - 1
            
            all_simulate_df = pd.concat([all_simulate_df, result_df], axis=0)
                        
        all_simulate_df.to_csv(f'data/warrant_simulation_{ticker}.csv')
        st.success(f'Save to data/warrant_simulation_{ticker}.csv')

    load_simulate_df = st.checkbox('Load Simulation', value=False)
    
    if load_simulate_df:
        all_simulate_df = pd.read_csv(f'data/warrant_simulation_{ticker}.csv')

    
    if all_simulate_df.empty:
        st.stop()
        
    cap = 20
    all_simulate_df['expect_value'] = all_simulate_df['expect_value'].clip(-cap, cap)
    
    # plot all expect value
    fig = px.line(all_simulate_df, x='TradingDate', y='expect_value', color='ticker', title='Expected Value')
    st.plotly_chart(fig)
    
    # expect value by listing date
    all_simulate_df['days_to_listing'] = all_simulate_df['days_to_listing'].astype(int)
    
    listing_expect_value_df = all_simulate_df.pivot(index='days_to_listing', columns='ticker', values='expect_value')
    listing_expect_value_df.sort_index(inplace=True)
    
    pu.plot_multi_line(listing_expect_value_df, title='Expected Value by Days to Listing', x_title='Days to Listing', y_title='Expected Value', legend_title='Ticker')

    select_all = st.checkbox('Select all', value=False)
    default_tickers = all_tickers if select_all else []
    
    selected_tickers = st.multiselect('Select tickers', all_tickers, default=default_tickers)

    for selected_ticker in selected_tickers:
        selected_ticker = selected_tickers[0]
        result_df = all_simulate_df[all_simulate_df['ticker'] == selected_ticker]
        
        st.write(result_df)
                
        pu.plot_single_line(result_df['close_cw'], title=f"Close CW {selected_ticker}")
        # pu.plot_single_line(trade_df['PnL'], title=f"PnL {selected_ticker}")
        pu.plot_single_bar(result_df['expect_value'], title=f"Expected Value {selected_ticker}")
        pu.plot_single_line(result_df['close_stock'], title=f"Close Stock {selected_ticker}")
