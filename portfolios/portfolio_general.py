
from portfolios.portfolio_base import PortfolioBase
from utils.processing import get_stocks
import streamlit as st
import yfinance as yf
import pandas as pd

# General Portfolio
# for optmizing general assets classes including
# US Stocks, Cryptocurrencies, Metals, Cash
class PortfolioGeneral(PortfolioBase):
    name = None
    is_stock = False
    symbolDate_dict = {}
    symbols = [
        # 'BTC-USD',
        'GC=F',
        'CL=F',
        'SPY',
        ]
    
    def __init__(self, symbolDate_dict) -> None:
        super().__init__()
        self.symbolDate_dict = symbolDate_dict
        
    def get_assets(self):
        stocks_df = yf.download(self.symbols)
        symbolDate_dict_copy = self.symbolDate_dict.copy()
        symbolDate_dict_copy['symbols'] = ['VN30']
        stocks_df = stocks_df['Adj Close']
        
        vn30 = get_stocks(symbolDate_dict_copy, 'close')
        
        # convert vn30 index to the same timezone as stocks_df
        vn30.index = vn30.index.tz_convert(stocks_df.index.tz)
        stocks_df = stocks_df.reindex(vn30.index, method='ffill')
        
                
        stocks_df  = pd.concat([stocks_df, vn30], axis=1)
                
        self.stocks_df = stocks_df
        return self.stocks_df

    def is_ready(self):
        return len(self.symbols) > 0