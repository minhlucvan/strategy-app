import pandas as pd
import numpy as np
import streamlit as st
import datetime

from utils.processing import get_stocks  

import pandas as pd
import numpy as np
import datetime
from utils.processing import get_stocks

class PaperStrategy:
    def __init__(self, symbolsDate_dict, capital=100000, holding_period=2):
        self.symbolsDate_dict = symbolsDate_dict
        self.symbols = symbolsDate_dict['symbols']
        self.start_date = symbolsDate_dict['start_date']
        self.end_date = symbolsDate_dict['end_date']
        self.current_date = self.start_date
        self.holding_period = holding_period
        self.capital = capital
        self.transaction_cost = 0.0016
        self.trade_log = []
        self.portfolio = {}
        self.nav = [capital]
        self.daily_returns = []
        
    def setup(self):
        # get historical data
        self.stocks_df = self.get_stock_data()
        
    def get_live_data(self):
        raise NotImplementedError("Subclasses should implement this method")
    
    def get_stock_data(self, is_live=False):
        if is_live:
            return self.get_live_data()
        
        symbolsDate_dict = self.symbolsDate_dict.copy()
        return get_stocks(symbolsDate_dict, stack=True)
    
    def generate_signals(self, stocks_df, volume_df):
        raise NotImplementedError("Subclasses should implement this method")
    
    def update_portfolio(self, latest_prices):
        closed_trades = []
        daily_pnl = 0
        
        for symbol, (entry_price, entry_date) in list(self.portfolio.items()):
            if (self.current_date - entry_date).days >= self.holding_period:
                exit_price = latest_prices.get(symbol, np.nan)
                if not np.isnan(exit_price):
                    ret = (exit_price - entry_price) / entry_price - self.transaction_cost
                    daily_pnl += ret * (self.capital / max(1, len(self.portfolio)))
                    closed_trades.append(symbol)
                    self.trade_log.append([entry_date, self.current_date, symbol, entry_price, exit_price, ret])
        
        for symbol in closed_trades:
            del self.portfolio[symbol]
        
        return daily_pnl
    
    def get_stock_ddata_slice(self, date):
        # all data before the current date
        return self.stocks_df.loc[:date]
    
    def run_day(self, is_live=False):
        current_date = self.current_date
        stocks_df = self.get_stock_ddata_slice(current_date)
        close_df = stocks_df['close']
        volume_df = stocks_df['volume']
        
        if close_df.empty or volume_df.empty:
            return
        
        latest_prices = close_df.loc[self.current_date].to_dict()
        daily_pnl = self.update_portfolio(latest_prices)
        signals = self.generate_signals(close_df, volume_df)
        new_positions = {s: (latest_prices[s], self.current_date) for s in signals.index if signals[s]}
        self.portfolio.update(new_positions)
        
        self.capital *= (1 + daily_pnl)
        self.nav.append(self.capital)
        self.daily_returns.append(daily_pnl)
    
    def backtest(self):
        date_range = pd.date_range(self.start_date, self.end_date, freq='B')
        for date in date_range:
            self.current_date = date
            self.run_day()
        
        return pd.DataFrame(self.trade_log, columns=['Entry Date', 'Exit Date', 'Symbol', 'Entry Price', 'Exit Price', 'Net Return'])
    
    def run_live(self):
        self.current_date = datetime.date.today()
        self.run_day(is_live=True)
    
    def get_trade_log(self):
        return pd.DataFrame(self.trade_log, columns=['Entry Date', 'Exit Date', 'Symbol', 'Entry Price', 'Exit Price', 'Net Return'])

class LLRPaperStrategy(PaperStrategy):
    def __init__(self, symbolsDate_dict, capital=100000, lookback=400, llr_threshold=20.0, holding_period=2):
        super().__init__(symbolsDate_dict, capital, holding_period)
        self.lookback = lookback
        self.llr_threshold = llr_threshold
    
    def calculate_llr(self, stocks_df, volume_df):
        avg_volume = volume_df.rolling(window=self.lookback, min_periods=1).mean()
        price_change_sign = np.sign(stocks_df.pct_change())
        return (volume_df * price_change_sign) / avg_volume
    
    def generate_signals(self, stocks_df, volume_df):
        llr = self.calculate_llr(stocks_df, volume_df)
        return llr.iloc[-1] > self.llr_threshold

# Example of running daily update
def run(symbol_benchmark, symbolsDate_dict): 
    st.header("LLR Live Test")
    if len(symbolsDate_dict['symbols']) < 1:
        st.info("Please select symbols.")
        st.stop()

    papper = LLRPaperStrategy(symbolsDate_dict)
    papper.setup()
    
    res = papper.backtest()
    st.write(res)
