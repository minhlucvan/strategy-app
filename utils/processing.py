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
from utils.stock_utils import get_stock_bars_very_long_term_cached, get_stock_balance_sheet, load_stock_balance_sheet_to_dataframe


@lru_cache
def get_us_symbol() -> dict:
    assets_df = ak.stock_us_spot_em()
    symbol_dict = {}
    for index, row in assets_df.iterrows():
        symbol = row['代码'].split('.')[1]
        symbol_dict[symbol] = row['代码']
    return symbol_dict


@lru_cache
def get_us_stock(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """get us stock data

    Args:
        ak_params symbol:str, start_date:str, end_date:str

    Returns:
        pd.DataFrame: _description_
    """
    return ak.stock_us_hist(symbol=symbol, start_date=start_date, end_date=end_date, adjust="qfq")


@lru_cache
def get_vn_stock(symbol: str, start_date: str, end_date: str, timeframe='1D') -> pd.DataFrame:
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
        stock_type='stock',
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
    # stock_df['date'] = stock_df['date'].dt.date
    
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
def stock_us_valuation_baidu(symbol: str = "AAPL", indicator: str = "总市值") -> pd.DataFrame:
    """
    百度股市通- 美股-财务报表-估值数据
    https://gushitong.baidu.com/stock/us-AAPL
    :param symbol: 股票代码
    :type symbol: str
    :param indicator: choice of {"总市值", "市盈率(TTM)", "市盈率(静)", "市净率", "市现率"}
    :type indicator: str
    :return: 估值数据
    :rtype: pandas.DataFrame
    """
    url = "https://finance.pae.baidu.com/selfselect/openapi"
    params = {
        "srcid": "51171",
        "code": symbol,
        "market": "us",
        "tag": f"{indicator}",
        "chart_select": "全部",
        "skip_industry": "0",
        "finClientType": "pc",
    }
    r = requests.get(url, params=params)
    data_json = r.json()
    if len(data_json["Result"]) == 0:
        temp_df = pd.DataFrame()
    else:
        temp_df = pd.DataFrame(data_json["Result"]["chartInfo"][0]["body"])
        temp_df.columns = ["date", "value"]
        temp_df["date"] = pd.to_datetime(temp_df["date"]).dt.date
        temp_df["value"] = pd.to_numeric(temp_df["value"])
    return temp_df

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
def get_us_valuation(symbol: st, indicator: str) -> pd.DataFrame:
    """get 百度股市通- 美股-财务报表-估值数据
        目标地址: https://gushitong.baidu.com/stock/us-AAPL
        限量: 单次获取指定 symbol 和 indicator 的所有历史数据
    Args:
        symbol      str symbol="AAPL"; 美股代码
        indicator	str	indicator="总市值"; choice of {"总市值", "市盈率(TTM)", "市盈率(静)", "市净率", "市现率"}
    return:
        date	object	-
        value	float64	-
    """
    result_df = stock_us_valuation_baidu(symbol, indicator)
    return result_df

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

class AKData(object):
    def __init__(self, market):
        self.market = market

    @vbt.cached_method
    def get_stock(self, symbol: str, start_date: datetime.datetime, end_date: datetime.datetime, timeframe='1D') -> pd.DataFrame:
        stock_df = pd.DataFrame()
        print(f"AKData-get_stock: {symbol}, {self.market}")
        symbol_df = load_symbol(symbol)
        
        # hot fix for vietnam stock
        if symbol == 'E1VFVN30':
            symbol_df = pd.DataFrame([{'category': 'index'}])
            
        if symbol == 'VN30':
            symbol_df = pd.DataFrame([{'category': 'index'}])

        if len(symbol_df) == 1:  # self.symbol_dict.keys():
            print(
                f"AKData-get_stock: {symbol}, {self.market}, {symbol_df.at[0, 'category']}")
            func = ('get_' + self.market + '_' +
                    symbol_df.at[0, 'category']).lower()
            symbol_full = symbol
            if self.market == 'US':
                symbol_full = symbol_df.at[0, 'exchange'] + '.' + symbol

            try:
                stock_df = eval(func)(symbol=symbol_full, start_date=start_date.strftime(
                    "%Y%m%d"), end_date=end_date.strftime("%Y%m%d"), timeframe=timeframe)
            except Exception as e:
                print(f"AKData-get_stock {func} error: {e}")

            if not stock_df.empty:
                if len(stock_df.columns) == 6:
                    stock_df.columns = ['date', 'open',
                                        'close', 'high', 'low', 'volume']
                elif len(stock_df.columns) == 7:
                    stock_df.columns = [
                        'date', 'open', 'close', 'high', 'low', 'volume', 'amount']
                else:
                    pass
                stock_df.index = pd.to_datetime(stock_df['date'], utc=True)
        return stock_df

    @vbt.cached_method
    def get_pettm(self, symbol: str) -> pd.DataFrame:
        print(f"AKData-get_pettm: {symbol}, {self.market}")
        stock_df = pd.DataFrame()
        symbol_df = load_symbol(symbol)

        if len(symbol_df) == 1:  # self.symbol_dict.keys():
            func = ('get_' + self.market + '_valuation').lower()
            try:
                stock_df = eval(func)(symbol=symbol, indicator='市盈率(TTM)')
            except Exception as e:
                print("get_pettm()---", e)

            if not stock_df.empty:
                stock_df.index = pd.to_datetime(stock_df['date'], utc=True)
                stock_df = stock_df['value']
        return stock_df
    
    @vbt.cached_method
    def get_valuation(self, symbol: str, indicator='pe') -> pd.DataFrame:
        print(f"AKData-get_valuation: {symbol}, {self.market}")
        stock_df = pd.DataFrame()
        symbol_df = load_symbol(symbol)

        if len(symbol_df) == 1:  # self.symbol_dict.keys():
            func = ('get_' + self.market + '_valuation').lower()
            try:
                stock_df = eval(func)(symbol=symbol, indicator=indicator)
            except Exception as e:
                print("get_pettm()---", e)

            if not stock_df.empty:
                stock_df.index = pd.to_datetime(stock_df['date'], utc=True)
                
        return stock_df
        
    @vbt.cached_method
    def get_fundamental(self, symbol: str) -> pd.DataFrame:
        print(f"AKData-get_fundamental: {symbol}, {self.market}")
        stock_df = pd.DataFrame()
        symbol_df = load_symbol(symbol)

        if len(symbol_df) == 1:
            func = ('get_' + self.market + '_fundamental').lower()
            try:
                stock_df = eval(func)(symbol=symbol)
            except Exception as e:
                print(e)

        return stock_df    

    @vbt.cached_method
    def get_financial(self, symbol: str) -> pd.DataFrame:
        print(f"AKData-get_financial: {symbol}, {self.market}")
        stock_df = pd.DataFrame()
        symbol_df = load_symbol(symbol)

        if len(symbol_df) == 1:
            func = ('get_' + self.market + '_financial').lower()
            try:
                stock_df = eval(func)(symbol=symbol)
            except Exception as e:
                print(e)

        return stock_df
    
    @vbt.cached_method
    def get_events(self, symbol: str, start_date: datetime.datetime, end_date: datetime.datetime) -> pd.DataFrame:
        print(f"AKData-get_events: {symbol}, {self.market}")
        stock_df = pd.DataFrame()
        symbol_df = load_symbol(symbol)

        if len(symbol_df) == 1:
            func = ('get_' + self.market + '_events').lower()
            try:
                stock_df = eval(func)(symbol=symbol, start_date=start_date, end_date=end_date)
            except Exception as e:
                print(e)

        return stock_df
    
    @vbt.cached_method
    def get_stock_foregin_flow(self, symbol: str, start_date: datetime.datetime, end_date: datetime.datetime) -> pd.DataFrame:
        print(f"AKData-get_stock_foregin_flow: {symbol}, {self.market}")
        stock_df = pd.DataFrame()
        symbol_df = load_symbol(symbol)

        if len(symbol_df) == 1:
            func = ('get_' + self.market + '_foregin_flow').lower()
            try:
                stock_df = eval(func)(symbol=symbol, start_date=start_date, end_date=end_date)
            except Exception as e:
                print(e)

        return stock_df
    
    @vbt.cached_method
    def get_pegttm(self, symbol: str) -> pd.DataFrame:
        stock_df = pd.DataFrame()
        symbol_df = load_symbol(symbol)
        mv_df = pd.DataFrame()
        pettm_df = pd.DataFrame()

        if len(symbol_df) == 1:  # self.symbol_dict.keys():
            func = ('get_' + self.market + '_valuation').lower()
            try:
                pettm_df = eval(func)(symbol=symbol, indicator='市盈率(TTM)')
                mv_df = eval(func)(symbol=symbol, indicator='总市值')
            except Exception as e:
                print("get_pettm()---", e)

            if not mv_df.empty and not pettm_df.empty:
                pettm_df.index = pd.to_datetime(pettm_df['date'], utc=True)
                mv_df.index = pd.to_datetime(mv_df['date'], utc=True)
                stock_df = pd.DataFrame()
                stock_df['pettm'] = pettm_df['value']
                stock_df['mv'] = mv_df['value']
                stock_df['earning'] = stock_df['mv']/stock_df['pettm']
                stock_df['cagr'] = stock_df['earning'].pct_change(periods=252)
                stock_df['pegttm'] = stock_df['pettm'] / stock_df['cagr']/100
                stock_df = stock_df['pegttm']
        return stock_df

# def get_stocks(symbolsDate_dict:dict):
#     datas = AKData(symbolsDate_dict['market'])
#     stock_dfs = []
#     for symbol in symbolsDate_dict['symbols']:
#         if symbol!='':
#                 stock_df = datas.get_stock(symbol, symbolsDate_dict['start_date'], symbolsDate_dict['end_date'])
#                 if stock_df.empty:
#                     st.warning(f"Warning: stock '{symbol}' is invalid or missing. Ignore it", icon= "⚠️")
#                 else:
#                     stock_dfs.append((symbol, stock_df))
#     return stock_dfs

@st.cache_data
def get_stocks(symbolsDate_dict: dict, column='close', stack=False, stack_level='factor', timeframe='D'):
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
                stock_df['value'] = stock_df['close'] * stock_df['volume']
                stock_df['price_change'] = stock_df['close'].pct_change()
                stock_df['volume_change'] = stock_df['volume'].pct_change()
                
                stock_df['value_change_weighted'] = stock_df['price_change'] * stock_df['volume_change']
                stocks_dfs[symbol] = stock_df if stack else stock_df[column]
    
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
def get_stocks_financial(symbolsDate_dict: dict, column='close',  stack=False, stack_level='factor'):
    datas = AKData(symbolsDate_dict['market'])
    stocks_dfs = {}
    
    for symbol in symbolsDate_dict['symbols']:
        if symbol != '':
            stock_df = datas.get_financial(symbol)
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

@lru_cache
def get_arkholdings(fund: str, end_date: str) -> pd.DataFrame:
    """get ARK fund holding companies's weight
    Args:
        ak_params symbol:str, start_date:str, end_date:str

    Returns:
        pd.DataFrame: _description_
    """

    r = requests.get(
        f"https://arkfunds.io/api/v2/etf/holdings?symbol={fund}&date_to={end_date}")
    data = r.json()
    holdings_df = pd.json_normalize(data, record_path=['holdings'])
    return holdings_df[['date', 'ticker', 'company', 'market_value', 'share_price', 'weight']]


@lru_cache
def get_feargreed(start_date: str) -> pd.DataFrame:
    BASE_URL = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
    r = requests.get("{}/{}".format(BASE_URL, start_date), headers=headers)
    data = r.json()
    fg_data = data['fear_and_greed_historical']['data']
    fg_df = pd.DataFrame(columns=['date', 'fear_greed'])
    for data in fg_data:
        dt = datetime.datetime.fromtimestamp(data['x'] / 1000, tz=pytz.utc)
        # fg_df = fg_df.append({'date': dt, 'fear_greed': int(data['y'])}, ignore_index=True)
        fg_df.loc[len(fg_df.index)] = [dt, int(data['y'])]
    fg_df.set_index('date', inplace=True)
    return fg_df
