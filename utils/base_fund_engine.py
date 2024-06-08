
import pandas as pd

from utils.processing import get_stocks

class fundEngine(object):
    '''base class of fund engine'''
    name = "base"
    market = ""
    funds = []
    fund_name = ""
    fund_ticker = ""
    fund_update_date = ""
    fund_info = ""
    fund_period = ""
    fund_df = pd.DataFrame()
    symbolsDate_dict = {}
    capital = 1000000
    is_live = False

    def __init__(self):
         pass
    
    def readStocks(self, fund_ticker:str):
        return

    def getSotcks(self):
        return get_stocks(self.symbolsDate_dict,'close')
    
    def setSymbolsDate(self, symbolsDate_dict):
        self.symbolsDate_dict = symbolsDate_dict
        return
    
    def update_capital(self, capital):
        self.capital = capital
        return
    
    def save(self):
        return