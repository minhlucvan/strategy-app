

from utils.tcbs_agent import TCBSAgent
from utils.config import get_config
import os
import streamlit as st

from utils.trader_base import TraderBase

class TraderTCBS(TraderBase):    
    def __init__(self):
        self.agent = self.get_agent()
    
    def use_account(self, account_id):
        self.agent.use_account(account_id)
    
    def get_agent(self):
        tcbs_config = get_config('tcbs.info')
        agent = TCBSAgent()
        agent.configure(tcbs_config)
        
        return agent
    
    def place_preorder(self, type='NB', symbol=None, price='', price_type='ATO', volume='0', start_date=None, end_date=None):
        st.write(f"Placing preorder {type} {symbol}, {price}, {price_type}, {volume}...")
        self.agent.preorder_stock(type, symbol, price, price_type, volume, start_date, end_date)
    
    def place_order(self, side, symbol, ref_id, price, volume, order_type, pin):
        st.write(f"Placing order {side} {symbol}, {price}, {volume}...")
        return self.agent.place_order(side, symbol, ref_id, price, volume, order_type, pin)
    
    def get_account_list(self):
        return self.agent.get_account_list()