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
from utils.fe_local_fund import fe_local_fund

class fe_vndirect(fundEngine):
    name = "Vndirect"
    market = "VN"

    def __init__(self):
        try:
            self.funds_name = ['VND']
        except ValueError as ve:
            st.write(f"Get {self.name} data error: {ve}")
        
    @vbt.cached_method
    def readStocks(self, fund_name: str):
        try:
            self.fund_ticker = 'VND'
            data = vndirect.get_fund_ratios()
            df = vndirect.load_fund_ratios_to_df(data)
            # {"code":"ACB","type":"IFC","period":"1M","ratioCode":"IFC_HOLDING_COUNT_CR","reportDate":"2024-04-26","value":1.0},
            
            # Column name corrections.
            df = df[(df["ratioCode"] == 'IFC_HOLDING_COUNT_CR')]
            
            # filter ticker with len(code) == 3
            df = df[df['code'].apply(lambda x: len(x) == 3)]
            
            max_report_date = df['reportDate'].max()
            
            df = df[(df["reportDate"] == max_report_date)]
            
            self.fund_period = max_report_date
            df = df[['code', 'type', 'value']]
            df = df.rename(columns={"code": "Ticker"})
            df = df.rename(columns={"type": "Stock"})
            df = df.rename(columns={"value": "Portfolio (%)"})
            
            df.index = df['Ticker']
            # self.fund_name = self.fund_rating_all_df.loc[self.fund_rating_all_df['代码']==fund_ticker, '简称'].values[0]
            self.fund_name = fund_name
            self.fund_update_date = datetime.date.today().strftime('%Y-%m-%d')
            self.fund_df = df
        except ValueError as ve:
            st.write(f"Get {self.name}-{self.fund_ticker} data error: {ve}")
        return

