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
from utils.tcbs_agent import TCBSAgent
import utils.config as cfg

class fe_broker_tcbs(fundEngine):
    name = "TCBS Broker"
    market = "VN"

    def __init__(self):
        try:
            self.funds_name = ['TCBS Fund']
        except ValueError as ve:
            st.write(f"Get {self.name} data error: {ve}")
        
    @vbt.cached_method
    def readStocks(self, fund_name: str):
        try:
            self.fund_ticker = 'TCBS'
            tcbs_config = cfg.get_config('tcbs.info')
            agent = TCBSAgent()
            agent.configure(tcbs_config)
            
            df = agent.get_total_stocks()
            
            df['name'] = df['symbol']
            
            df = df[['symbol', 'name', 'weight']]
            
            df.columns = ['Ticker', 'Stock', 'Portfolio (%)']
            
            df.index = df['Ticker']
            # self.fund_name = self.fund_rating_all_df.loc[self.fund_rating_all_df['代码']==fund_ticker, '简称'].values[0]
            self.fund_name = fund_name
            self.fund_update_date = datetime.date.today().strftime('%Y-%m-%d')
            self.fund_df = df
        except ValueError as ve:
            st.write(f"Get {self.name}-{self.fund_ticker} data error: {ve}")
        return

