import pandas as pd
import numpy as np
import json
import datetime

import streamlit as st
import vectorbt as vbt
import akshare as ak


from utils.fe_base_fund_engine import fundEngine
from utils.dataroma import *
from utils import vndirect
from utils.portfolio import Portfolio
from utils.riskfolio import get_pfOpMS
import json
import os

class fe_local_fund(fundEngine):
    name = "Local"
    market = "VN"
    file_path = 'data/portfolio.json'
    portfolio = None
    value_df = pd.DataFrame()
    position_df = pd.DataFrame()
    symbols = []

    def __init__(self, is_live=False):
        try:
            self.funds_name = ['Local Fund']
            self.is_live = is_live
            self.load()
        except ValueError as ve:
            st.write(f"Get {self.name} data error: {ve}")
            
    def readPortfolio(self):
        st.write('Loading Portfolio...')
        mode = 'live' if self.is_live else 'backtest'
        st.write(f'Mode: {mode}')
        self.portfolio = Portfolio(is_live=self.is_live)
        
        for index in self.portfolio.df.index:
            pf = vbt.Portfolio.loads(self.portfolio.df.loc[index, 'vbtpf'])
            self.value_df[self.portfolio.df.loc[index, 'name']] = pf.value()
            self.position_df[self.portfolio.df.loc[index, 'name']] = pf.position_mask()
                
        if self.fund_df.empty:
            # optimize portfolio
            self.optimize_pf()
        self.symbols = self.fund_df['Ticker'].tolist()
        
        self.setPortfolioAllocation()
    
    def setPortfolioAllocation(self):
        
        allocation = {
            "assets": {},
            "capital": self.capital,
        }
        
        for index in self.symbols:
            allocation['assets'][index] = self.fund_df[self.fund_df['Ticker'] == index]['Portfolio (%)'].values[0]
            
        self.portfolio.allocate(allocation)
    
    def optimize_pf(self):
        st.write('Optimizing Portfolio...')

        _, weights_df = get_pfOpMS(self.value_df, return_w=True, show_report=False, plot=False)
        
        df = pd.DataFrame(index=weights_df.index, columns=['Ticker', 'Stock', 'Portfolio (%)'])
        df['Ticker'] = weights_df['Ticker']
        df['Stock'] = weights_df['Ticker']
        df['Portfolio (%)'] = weights_df['weights'] * 100
        
        self.fund_df = df

    
    @vbt.cached_method
    def readStocks(self, fund_name: str = 'Local Fund'):
        try:
            self.fund_ticker = 'Local'            
            self.fund_name = fund_name
            self.fund_update_date = datetime.date.today().strftime('%Y-%m-%d')
                        
            self.readPortfolio()
                        
        except ValueError as ve:
            st.write(f"Get {self.name}-{self.fund_ticker} data error: {ve}")
        return
    
    def getSotcks(self):
        return self.value_df
    
    def load(self):
        if not os.path.exists(self.file_path):
            return
        
        with open(self.file_path, 'r') as f:
            data = json.load(f)
            self.fund_name = data["name"]
            self.fund_ticker = data["ticker"]
            self.fund_update_date = data["update_date"]
            self.fund_info = data["info"]
            self.fund_period = data["period"]
            self.fund_df = pd.read_json(data["df"])
            self.symbolsDate_dict = data["symbolsDate_dict"]
            self.capital = data["capital"]
            
            self.fund_update_date = datetime.datetime.strptime(self.fund_update_date, '%Y-%m-%d')
    
    def update_fund_df(self, fund_df):
        self.fund_df = fund_df
    
    def save(self):
        if not os.path.exists('data'):
            os.makedirs('data')

        data = {
            "name": self.fund_name,
            "ticker": self.fund_ticker,
            "update_date": self.fund_update_date,
            "info": self.fund_info,
            "period": self.fund_period,
            "df": self.fund_df.to_json(),
            "symbolsDate_dict": self.symbolsDate_dict,
            "capital": self.capital
        }
        data['symbolsDate_dict']['start_date'] = data['symbolsDate_dict']['start_date'].strftime('%Y-%m-%d')
        data['symbolsDate_dict']['end_date'] = data['symbolsDate_dict']['end_date'].strftime('%Y-%m-%d')
        
        with open(self.file_path, 'w') as f:
            json.dump(data, f)
            
    def reset(self):
        if os.path.exists(self.file_path):
            os.remove(self.file_path)
        else:
            st.write("The file does not exist")