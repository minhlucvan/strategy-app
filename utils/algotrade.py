import json
import pandas as pd

import requests
import json
import pandas as pd
import datetime as dt

import requests_cache

CACHE_TTL = 60 * 60 * 24  # 1 day
requests = requests_cache.CachedSession('cache/algotrade', expire_after=CACHE_TTL, allowable_codes=[200])

MAX_RETRIES = 5
RETRY_WAIT_TIME = 30


def get_foreign_sell_value(from_date = None, to_date = None):
    
    if isinstance(from_date, dt.datetime):
        from_date = from_date.strftime("%Y-%m-%d")
        
    if isinstance(to_date, dt.datetime):
        to_date = to_date.strftime("%Y-%m-%d")
    
    if from_date is None:
        from_str = "2021-07-01"
    
    if to_date is None:
        to_date = dt.datetime.now().strftime("%Y-%m-%d")
        
    url = "https://webapi.algotrade.vn/gql/v1/"

    query = """
    query GET_FOREIGN_SELL_VALUE($fromStr: String!, $toStr: String!) {
      foreignSell: vn30MarketValueDaily(
        valueType: "FOREIGN_SELL_VALUE"
        fromStr: $fromStr
        toStr: $toStr
      ) {
        id
        value
        dateMark
        year
        month
        day
        symbol
        expiryDateMark
        __typename
      }
    }
    """

    variables = {
        "fromStr": from_str,
        "toStr": to_date
    }

    payload = {
        "query": query,
        "variables": variables
    }

    headers = {
        'Accept-Language': 'en-US,en;q=0.9,vi;q=0.8,vi-VN;q=0.7,fr-FR;q=0.6,fr;q=0.5,de;q=0.4',
        'Connection': 'keep-alive',
        'DNT': '1',
        'Origin': 'https://live.algotrade.vn',
        'Referer': 'https://live.algotrade.vn/',
        'Sec-Fetch-Dest': 'empty',
        'Sec-Fetch-Mode': 'cors',
        'Sec-Fetch-Site': 'same-site',
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36',
        'accept': '*/*',
        'content-type': 'application/json',
        'sec-ch-ua': '"Google Chrome";v="125", "Chromium";v="125", "Not.A/Brand";v="24"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"macOS"',
        'Cookie': 'csrftoken=RXTVrg117JWoYqwi23rKlg74d6fnHO2p4ArPJOVEcWMOfJlWuxVf2ZWHDqpxZ3OR'
    }

    response = requests.post(url, headers=headers, json=payload)

    return response.json()


def load_foregin_sell_value_to_dataframe(data):
    # {
    #     "data": {
    #         "foreignSell": [
    #             {
    #                 "id": "FOREIGN_SELL_VALUE_2021-07-01",
    #                 "value": "1398006810.00",
    #                 "dateMark": "2021-07-01",
    #                 "year": 2021,
    #                 "month": 7,
    #                 "day": 1,
    #                 "symbol": "VN30B",
    #                 "expiryDateMark": null,
    #                 "__typename": "CustomMarketDailyType"
    #             }
    
    foreign_sell = data['data']['foreignSell']
    foreign_sell_df = pd.DataFrame(foreign_sell)
    
    foreign_sell_df['value'] = foreign_sell_df['value'].astype(float)
    foreign_sell_df['dateMark'] = pd.to_datetime(foreign_sell_df['dateMark'])
    
    return foreign_sell_df



def get_foreign_buy_volume(from_date = None, to_date = None):
    
    if isinstance(from_date, dt.datetime):
        from_date = from_date.strftime("%Y-%m-%d")
        
    if isinstance(to_date, dt.datetime):
        to_date = to_date.strftime("%Y-%m-%d")
    
    if from_date is None:
        from_date = "2021-07-01"
    
    if to_date is None:
        to_date = dt.datetime.now().strftime("%Y-%m-%d")
        
    url = "https://webapi.algotrade.vn/gql/v1/"
    
    query = """
    query GET_FOREIGN_BUY_VOLUME($fromStr: String!, $toStr: String!) {
      foreignBuy: vn30MarketDaily(
        valueType: "FOREIGN_BUY_VOL"
        fromStr: $fromStr
        toStr: $toStr
      ) {
        id
        value
        dateMark
        year
        month
        day
        symbol
        expiryDateMark
        __typename
      }
    }
    """
    
    variables = {
        "fromStr": from_date,
        "toStr": to_date
    }
    
    payload = json.dumps({
        "query": query,
        "variables": variables
    })
    
    headers = {
        'Accept-Language': 'en-US,en;q=0.9,vi;q=0.8,vi-VN;q=0.7,fr-FR;q=0.6,fr;q=0.5,de;q=0.4',
        'Connection': 'keep-alive',
        'DNT': '1',
        'Origin': 'https://live.algotrade.vn',
        'Referer': 'https://live.algotrade.vn/',
        'Sec-Fetch-Dest': 'empty',
        'Sec-Fetch-Mode': 'cors',
        'Sec-Fetch-Site': 'same-site',
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36',
        'accept': '*/*',
        'content-type': 'application/json',
        'sec-ch-ua': '"Google Chrome";v="125", "Chromium";v="125", "Not.A/Brand";v="24"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"macOS"',
        'Cookie': 'csrftoken=RXTVrg117JWoYqwi23rKlg74d6fnHO2p4ArPJOVEcWMOfJlWuxVf2ZWHDqpxZ3OR'
    }
    
    response = requests.post(url, headers=headers, data=payload)
    
    return response.json()

def load_foreign_buy_volume_to_dataframe(data):
    # {
    #     "data": {
    #         "foreignBuy": [
    #             {
    #                 "id": "FOREIGN_BUY_VOL_2021-07-01",
    #                 "value": "1398006810.00",
    #                 "dateMark": "2021-07-01",
    #                 "year": 2021,
    #                 "month": 7,
    #                 "day": 1,
    #                 "symbol": "VN30B",
    #                 "expiryDateMark": null,
    #                 "__typename": "CustomMarketDailyType"
    #             }
    
    foreign_buy = data['data']['foreignBuy']
    foreign_buy_df = pd.DataFrame(foreign_buy)
    
    foreign_buy_df['value'] = foreign_buy_df['value'].astype(float)
    foreign_buy_df['dateMark'] = pd.to_datetime(foreign_buy_df['dateMark'])
    
    # set index
    foreign_buy_df.set_index('dateMark', inplace=True)
    
    return foreign_buy_df

import requests
import json

def get_foreign_sell_volume(from_date = None, to_date = None):
    
    if isinstance(from_date, dt.datetime):
        from_date = from_date.strftime("%Y-%m-%d")
        
    if isinstance(to_date, dt.datetime):
        to_date = to_date.strftime("%Y-%m-%d")
    
    if from_date is None:
        from_date = "2021-07-01"
    
    if to_date is None:
        to_date = dt.datetime.now().strftime("%Y-%m-%d")
        
    url = "https://webapi.algotrade.vn/gql/v1/"

    query = """
    query GET_FOREIGN_SELL_VOLUME($fromStr: String!, $toStr: String!) {
      foreignSell: vn30MarketDaily(
        valueType: "FOREIGN_SELL_VOL"
        fromStr: $fromStr
        toStr: $toStr
      ) {
        id
        value
        dateMark
        year
        month
        day
        symbol
        expiryDateMark
        __typename
      }
    }
    """

    variables = {
        "fromStr": from_date,
        "toStr": to_date
    }

    payload = json.dumps({
        "query": query,
        "variables": variables
    })

    headers = {
        'Accept-Language': 'en-US,en;q=0.9,vi;q=0.8,vi-VN;q=0.7,fr-FR;q=0.6,fr;q=0.5,de;q=0.4',
        'Connection': 'keep-alive',
        'DNT': '1',
        'Origin': 'https://live.algotrade.vn',
        'Referer': 'https://live.algotrade.vn/',
        'Sec-Fetch-Dest': 'empty',
        'Sec-Fetch-Mode': 'cors',
        'Sec-Fetch-Site': 'same-site',
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36',
        'accept': '*/*',
        'content-type': 'application/json',
        'sec-ch-ua': '"Google Chrome";v="125", "Chromium";v="125", "Not.A/Brand";v="24"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"macOS"',
        'Cookie': 'csrftoken=RXTVrg117JWoYqwi23rKlg74d6fnHO2p4ArPJOVEcWMOfJlWuxVf2ZWHDqpxZ3OR'
    }

    response = requests.post(url, headers=headers, data=payload)
    return response.json()

def load_foreign_sell_volume_to_dataframe(data):
    # {
    #     "data": {
    #         "foreignSell": [
    #             {
    #                 "id": "FOREIGN_SELL_VOL_2021-07-01",
    #                 "value": "1398006810.00",
    #                 "dateMark": "2021-07-01",
    #                 "year": 2021,
    #                 "month": 7,
    #                 "day": 1,
    #                 "symbol": "VN30B",
    #                 "expiryDateMark": null,
    #                 "__typename": "CustomMarketDailyType"
    #             }
    
    foreign_sell = data['data']['foreignSell']
    foreign_sell_df = pd.DataFrame(foreign_sell)
    
    foreign_sell_df['value'] = foreign_sell_df['value'].astype(float)
    foreign_sell_df['dateMark'] = pd.to_datetime(foreign_sell_df['dateMark'])
    
    # set index
    foreign_sell_df.set_index('dateMark', inplace=True)
    
    return foreign_sell_df

def get_foreign_buy_value(from_date = None, to_date = None):
    
    if isinstance(from_date, dt.datetime):
        from_date = from_date.strftime("%Y-%m-%d")
        
    if isinstance(to_date, dt.datetime):
        to_date = to_date.strftime("%Y-%m-%d")
    
    if from_date is None:
        from_date = "2021-07-01"
    
    if to_date is None:
        to_date = dt.datetime.now().strftime("%Y-%m-%d")
        
    url = "https://webapi.algotrade.vn/gql/v1/"
    
    query = """
    query GET_FOREIGN_BUY_VALUE($fromStr: String!, $toStr: String!) {
      foreignBuy: vn30MarketValueDaily(
        valueType: "FOREIGN_BUY_VALUE"
        fromStr: $fromStr
        toStr: $toStr
      ) {
        id
        value
        dateMark
        year
        month
        day
        symbol
        expiryDateMark
        __typename
      }
    }
    """
    
    variables = {
        "fromStr": from_date,
        "toStr": to_date
    }
    
    payload = {
        "query": query,
        "variables": variables
    }
    
    headers = {
        'Accept-Language': 'en-US,en;q=0.9,vi;q=0.8,vi-VN;q=0.7,fr-FR;q=0.6,fr;q=0.5,de;q=0.4',
        'Connection': 'keep-alive',
        'DNT': '1',
        'Origin': 'https://live.algotrade.vn',
        'Referer': 'https://live.algotrade.vn/',
        'Sec-Fetch-Dest': 'empty',
        'Sec-Fetch-Mode': 'cors',
        'Sec-Fetch-Site': 'same-site',
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36',
        'accept': '*/*',
        'content-type': 'application/json',
        'sec-ch-ua': '"Google Chrome";v="125", "Chromium";v="125", "Not.A/Brand";v="24"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"macOS"',
        'Cookie': 'csrftoken=RXTVrg117JWoYqwi23rKlg74d6fnHO2p4ArPJOVEcWMOfJlWuxVf2ZWHDqpxZ3OR'
    }
    
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    return response.json()

def load_foreign_buy_value_to_dataframe(data):
    # {
    #     "data": {
    #         "foreignBuy": [
    #             {
    #                 "id": "FOREIGN_BUY_VALUE_2021-07-01",
    #                 "value": "1398006810.00",
    #                 "dateMark": "2021-07-01",
    #                 "year": 2021,
    #                 "month": 7,
    #                 "day": 1,
    #                 "symbol": "VN30B",
    #                 "expiryDateMark": null,
    #                 "__typename": "CustomMarketDailyType"
    #             }
    
    foreign_buy = data['data']['foreignBuy']
    foreign_buy_df = pd.DataFrame(foreign_buy)
    
    foreign_buy_df['value'] = foreign_buy_df['value'].astype(float)
    foreign_buy_df['dateMark'] = pd.to_datetime(foreign_buy_df['dateMark'])
    
    # set index
    foreign_buy_df.set_index('dateMark', inplace=True)
    
    return foreign_buy_df

def get_vn30_foreign_trade_values(last_id=None):
    url = "https://webapi.algotrade.vn/gql/v1/"
    
    query = """
    query GET_VN30_FOREIGN_TRADE_VALUES($lastId: String) {
      vn30ForeignTradeValues(lastId: $lastId) {
        id
        datetime
        intradayAccValue
        __typename
      }
    }
    """
    
    variables = {
        "lastId": last_id
    }
    
    payload = {
        "query": query,
        "variables": variables
    }
    
    headers = {
        'Accept-Language': 'en-US,en;q=0.9,vi;q=0.8,vi-VN;q=0.7,fr-FR;q=0.6,fr;q=0.5,de;q=0.4',
        'Connection': 'keep-alive',
        'DNT': '1',
        'Origin': 'https://live.algotrade.vn',
        'Referer': 'https://live.algotrade.vn/',
        'Sec-Fetch-Dest': 'empty',
        'Sec-Fetch-Mode': 'cors',
        'Sec-Fetch-Site': 'same-site',
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36',
        'accept': '*/*',
        'content-type': 'application/json',
        'sec-ch-ua': '"Google Chrome";v="125", "Chromium";v="125", "Not.A/Brand";v="24"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"macOS"',
        'Cookie': 'csrftoken=RXTVrg117JWoYqwi23rKlg74d6fnHO2p4ArPJOVEcWMOfJlWuxVf2ZWHDqpxZ3OR'
    }
    
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    
    return response.json()

def load_vn30_foreign_trade_values_to_dataframe(data):
    # {
    #     "data": {
    #         "vn30ForeignTradeValues": [
    #             {
    #                 "id": "2022-10-20T00:00:00.000Z",
    #                 "datetime": "2022-10-20T00:00:00.000Z",
    #                 "intradayAccValue": "0.00",
    #                 "__typename": "VN30ForeignTradeValueType"
    #             }
    
    vn30_foreign_trade_values = data['data']['vn30ForeignTradeValues']
    vn30_foreign_trade_values_df = pd.DataFrame(vn30_foreign_trade_values)
    
    vn30_foreign_trade_values_df['intradayAccValue'] = vn30_foreign_trade_values_df['intradayAccValue'].astype(float)
    vn30_foreign_trade_values_df['datetime'] = pd.to_datetime(vn30_foreign_trade_values_df['datetime'])
    
    # set index
    vn30_foreign_trade_values_df.set_index('datetime', inplace=True)
    
    return vn30_foreign_trade_values_df