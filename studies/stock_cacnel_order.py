from time import sleep
import pandas as pd
import numpy as np
import json

import streamlit as st
import vectorbt as vbt

import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots

from utils.st import execute_trade, show_trade_form
from utils.component import check_password, input_dates, input_SymbolsDate
import matplotlib.pyplot as plt

from utils.processing import get_stocks
from utils.stock_utils import get_last_trading_history
from utils.trader import get_trader, get_trader_list
from utils.vbt import plot_pf
from vbt_strategy.MOM_D import get_MomDInd
from studies.magic_fomula_study import run as run_magic_fomula

import numpy as np
import pandas as pd
import utils.vnstock as vnstock

def cancel_order(trader, order_id):
    st.write(f"Cancelling order {order_id}...")
    trader.cancel_preorder(order_id)
    st.write(f"Order {order_id} cancelled.")


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
    
    orders = trader.get_pending_orders()
    
    orders_df = pd.DataFrame(orders)
    
    st.write(orders_df)
    
    cancel_all = st.button("Cancel All")
    
    if cancel_all:
        for index, row in orders_df.iterrows():
            cancel_order(trader, row['id'])