import requests
import json
import subprocess
import datetime
import pandas as pd
from utils.config import deep_value
from utils.tcbs_api import TCBSAPI

class TCBSAgent:
    config = None
    auth_token = None
    tcbs_id = None
    sub_account_id = None
    account_id = None
    custodyId = None
    account_type = None
    
    # store data
    assets_info = None
    balance_info = None
    
    active_account = None
    
    def __init__(self):
        self.auth_token = None
        
    def get_config(self, key):
        return deep_value(self.config, key)

    def get_account_list(self):
        if self.balance_info is None:
            self.get_balance_info()
            
        accounts = deep_value(self.balance_info, 'bankSubAccounts')
        
        for account in accounts:
            account['id'] = account.get('accountNo')
            account['name'] = f"{account.get('accountName')}({account.get('accountTypeName')})"
        
        return accounts

    def get_total_cash(self,):
        if self.assets_info is None:
            self.get_assets_info()
            
        return deep_value(self.assets_info, 'assets.cash.totalCash')
    
    def get_stocks(self):
        if self.assets_info is None:
            self.get_assets_info()
        
        stock_data = deep_value(self.assets_info, 'assets.stock.data')
        stock_assets = []
        
        for data in stock_data:
            accountNo = data.get('accountNo')
            tickers = data.get('tickers')
            
            for ticker in tickers:
                # "volume":0
                # "symbol":"CMG"
                # "costPrice":0
                # "currentPrice":68400
                stock_assets.append({
                    'accountNo': accountNo,
                    'symbol': ticker.get('symbol'),
                    'volume': ticker.get('volume'),
                    'costPrice': ticker.get('costPrice'),
                    'currentPrice': ticker.get('currentPrice')
                })
        stock_assets_df = pd.DataFrame(stock_assets)
        return stock_assets_df
    
    def get_assets_allocation(self):
        total_cash = self.get_total_cash()
        total_stock = self.get_total_stocks_value()
        
        assets_df = pd.DataFrame([
            {'symbol': 'cash', 'value': total_cash},
            {'symbol': 'stock', 'value': total_stock}
        ])
        
        return assets_df
        
    def get_total_stocks_value(self):
        stocks_df = self.get_stocks()
        stocks_df['value'] = stocks_df['volume'] * stocks_df['currentPrice']
        return stocks_df['value'].sum()
        
    def get_total_stocks(self):
        stocks_df = self.get_stocks()
        stocks_grouped_df = stocks_df.groupby('symbol').agg({'volume': 'sum', 'costPrice': 'mean', 'currentPrice': 'mean'}).reset_index()
    
        stocks_grouped_df['value'] = stocks_grouped_df['volume'] * stocks_grouped_df['currentPrice']

        stocks_grouped_df['return'] = (stocks_grouped_df['currentPrice'] - stocks_grouped_df['costPrice']) / stocks_grouped_df['costPrice']

        stocks_grouped_df['weight'] = stocks_grouped_df['value'] / stocks_grouped_df['value'].sum()
        
        return stocks_grouped_df

    def get_account_normal(self):
        return self.get_account_info('NORMAL')
    
    def get_account_margin(self):
        return self.get_account_info('MAGIN')
    
    def get_account_by_id(self, account_id):
        account_list = self.get_account_list()
        for account in account_list:
            if account.get('id') == account_id:
                return account
        
        return None
        
    def use_account(self, account):
        if isinstance(account, str):
            account = self.get_account_by_id(account)
            
        self.active_account = account
        self.account_id = account.get('accountNo')
        self.account_type = account.get('accountType')

    def configure(self, config):
        self.config = config
        self.auth_token = config.get('authToken')
        self.tcbs_id = config.get('TCBSId')
        self.custodyId = config.get('custodyId')
        self.api = TCBSAPI(self.auth_token)

    def place_order(self, side, symbol, ref_id, price, volume, order_type, pin):
        return self.api.place_order(self.sub_account_id, self.account_id, side, symbol, ref_id, price, volume, order_type, pin)

    def request_otp(self, tcbs_id, iotp_ticket, session, browser_info, duration, type_otp, device_info):
        return self.api.request_otp(tcbs_id, iotp_ticket, session, browser_info, duration, type_otp, device_info)

    def need_otp(self):
        return self.api.need_otp()
    
    def get_total_stock_of_account(self, symbol, account_id):
        stocks_df = self.get_stocks()
        stocks_df = stocks_df[stocks_df['accountNo'] == account_id]
        stocks_df = stocks_df[stocks_df['symbol'] == symbol]
        return stocks_df['volume'].sum()

    def place_stock_order(self, account_id, side, symbol, price, volume, order_type):
        return self.api.place_stock_order(account_id, side, symbol, price, volume, order_type)

    def preorder_stock(self, type, symbol, price, price_type, volume, start_date=None, end_date=None, full_amount=False):
        
        if full_amount:
            volume =  self.get_total_stock_of_account(symbol, self.account_id)
            
        return self.api.preorder_stock(
            tcbs_id=self.tcbs_id, 
            custodyId=self.custodyId, 
            account_id=self.account_id, 
            type=type,
            symbol=symbol,
            price=price,
            price_type=price_type,
            volume=volume,
            account_type=self.account_type,
            start_date=start_date,
            end_date=end_date)

    def get_balance_info(self):
        self.balance_info = self.api.get_balance_info(self.custodyId)
        return self.balance_info

    def get_hawkeye_balance(self, tcbs_id):
        return self.api.get_hawkeye_balance(tcbs_id)

    def get_assets_info(self):
        self.assets_info = self.api.get_assets_info()
        return self.assets_info

    def get_account_info(self):
        return self.api.get_account_info(self.tcbs_id)

    def get_market_calendar(self, from_date=None, to_date=None):
        from_date = from_date or datetime.datetime.now().strftime('%Y-%m-%d')
        to_date = to_date or (datetime.datetime.now() + datetime.timedelta(days=30)).strftime('%Y-%m-%d')
        
        data = self.api.get_market_calendar(self.tcbs_id, from_date, to_date)
        
        # List to store parsed data
        parsed_data = []

        # Parse the JSON data
        for timeline_entry in data['timeline']:
            date = timeline_entry['date']
            for event in timeline_entry['events']:
                event_data = {
                    'date': date,
                    'defType': event['defType'],
                    'evName': event['evName'],
                    'mkCode': event['mkCode'],
                }
                
                # Add stockInfo if it exists
                if 'stockInfo' in event:
                    stock_info = event['stockInfo']
                    event_data.update(stock_info)
                
                parsed_data.append(event_data)

        # Convert to DataFrame
        df = pd.DataFrame(parsed_data)
        
        return df