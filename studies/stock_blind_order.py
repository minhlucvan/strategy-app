from time import sleep
import pandas as pd
import numpy as np
import json

import streamlit as st
import vectorbt as vbt

import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots

from pages.LiveTrade import execute_trade, show_trade_form
from utils.component import check_password, input_dates, input_SymbolsDate
import matplotlib.pyplot as plt

from utils.processing import get_stocks
from studies.rrg import plot_RRG, rs_ratio, RRG_Strategy
from utils.stock_utils import get_last_trading_history
from utils.trader import get_trader, get_trader_list
from utils.vbt import plot_pf
from vbt_strategy.MOM_D import get_MomDInd
from studies.magic_fomula_study import run as run_magic_fomula

import numpy as np
import pandas as pd
import utils.vnstock as vnstock

def calculate_order_price(price, gap_pct=0.09):
    price = price - price * gap_pct
    return price


def run(symbol_benchmark, symbolsDate_dict):
    trader_list =  get_trader_list()

    
    # select broker
    broker = st.sidebar.selectbox("Broker", trader_list)
    
    trader = get_trader(broker)
    
    account_list = trader.get_account_list()
    
    # select account
    account = st.sidebar.selectbox("Account", account_list, format_func=lambda x: x['name'], index=0)
    
    if account is not None:
        trader.use_account(account['id'])
    
    symbols = symbolsDate_dict['symbols']
    
    st.write(f"Symbols: {symbols}")
    
    last_prices = get_last_trading_history(symbols, stock_type='stock')
    
    last_closes = last_prices['close'].to_dict()
    
    run_trade = st.button("Run Trade")
    
    if run_trade:
        for symbol in symbols:
            
            price = last_closes[symbol]
            
            order_price = calculate_order_price(price, gap_pct=0.09)
            value = 5_000_000
            size = 100

            row = {
                "Price": order_price,
                "Size": size,
                "Side": "Buy",
                "Symbol": symbol,
            }
            if pd.isna(row['Price']):
                st.write(f"Price is NaN for {row['Symbol']}")
                continue
            
            try:
                execute_trade(trader, row['Side'], row['Symbol'], row['Price'], row['Size'], price_type="LO")
            except Exception as e:
                st.error(f"Error: {e}")
            
            sleep(1)
        