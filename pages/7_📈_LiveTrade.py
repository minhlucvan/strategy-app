import pandas as pd
import numpy as np
import pytz
from datetime import datetime, date
import json
import plotly.graph_objects as go

import streamlit as st

from utils.fe_local_fund import fe_local_fund
from utils.riskfolio import get_pfOpMS
from utils.stock_utils import get_first_trade_date_of_week, get_last_trading_date
from utils.trader import get_trader, get_trader_list

import vectorbt as vbt

from utils.vbt import plot_pf
from utils.component import check_password, params_selector
from utils.portfolio import Portfolio, selectpf_bySymbols
from vbt_strategy.PairTrade import pairtrade_pfs
from utils.st import check_params, show_PortfolioTable, show_PortforlioDetail, show_trade_form
from utils.db import get_SymbolName, get_SymbolsNames
from utils.vbt import display_pfbrief

import config
import utils.vnstock as vnstock


def main():
    
    trader_list =  get_trader_list()
    
    # select broker
    broker = st.sidebar.selectbox("Broker", trader_list)
    
    trader = get_trader(broker)

    account_list = trader.get_account_list()
    
    # select account
    account = st.sidebar.selectbox("Account", account_list, format_func=lambda x: x['name'], index=0)
    
    if account is not None:
        trader.use_account(account['id'])
        
    # select date on the sidebar
    today = st.sidebar.date_input("Current Date", datetime.now().date())
    
    st.header("Live Trading")
    selected_pfs = []
    fund = fe_local_fund(is_live=True)
    
    fund.readStocks()
    
    portfolio = fund.portfolio
    
    selected_pfs = show_PortfolioTable(portfolio.df)

    ##多portfolio比较
    value_df = pd.DataFrame()
    position_df = pd.DataFrame()
    
    if len(selected_pfs) == 0:
        st.info("Please select portfolios.")
        st.stop()
        
    # plot pie chart for selected portfolios
    st.markdown("#### Portfolio Allocation")
    fund_df = fund.fund_df
    fig = go.Figure(data=[go.Pie(labels=fund_df['Ticker'], values=fund_df['Portfolio (%)'])])
    st.plotly_chart(fig, use_container_width=True)
    
    for index in selected_pfs:
        pf = vbt.Portfolio.loads(portfolio.df.loc[index, 'vbtpf'])
        value_df[portfolio.df.loc[index, 'name']] = pf.value()
        position_df[portfolio.df.loc[index, 'name']] = pf.position_mask()
        # show_PortforlioDetail(portfolio.df, index)

    ##无选择portforlio
    for index in selected_pfs:
        st.write(f"updating portfolio('{portfolio.df.loc[index]['name']}')...")
        if not portfolio.update(portfolio.df.loc[index]['id']):
            st.error(f"Fail to update portfolio('{portfolio.df.loc[index]['name']}').")

    last_trading_date = get_last_trading_date()
    first_trade_date_of_week =  get_first_trade_date_of_week()
    check_df = portfolio.check_records(dt=today, last_trading_date=last_trading_date, first_trade_date_of_week=first_trade_date_of_week, selected_pfs=selected_pfs)
    
    check_df = portfolio.ajust_positions(check_df)
    
    # send the notification to telegram users
    if len(check_df) == 0:
        st.success("No signal found.")
    else:
        st.markdown("### Trading Signals 📈")

        signals_by_name = check_df.groupby('Name')
        for name, signals in signals_by_name:
            st.write(f"##### {name} 📊")
            st.dataframe(signals[['Side', 'Symbol', 'Price', 'Size', 'Timestamp']], use_container_width=True)
            
            # execute the trades
            for i, row in signals.iterrows():
                with st.expander(f"{row['Side']} {row['Symbol']}"):
                    show_trade_form(name, row, trader)
if __name__ == "__main__":
    if check_password():
        main()
