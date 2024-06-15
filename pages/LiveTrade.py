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
from pages.Strategy import check_params
from utils.db import get_SymbolName, get_SymbolsNames
from utils.vbt import display_pfbrief

import config

def select_portfolios(portfolio_df, default_selected=False):
        df_with_selections = portfolio_df.copy()
        df_with_selections.set_index('id', inplace=True)
        df_with_selections.insert(0, "Select", False)
        # display in 100% percentage format
        df_with_selections['annual_return'] *= 100
        df_with_selections['lastday_return'] *= 100
        df_with_selections['total_return'] *= 100
        df_with_selections['maxdrawdown'] *= 100
        df_with_selections['Select'] = default_selected

        edited_df = st.data_editor(
                        df_with_selections,
                        hide_index=True,
                        use_container_width=True,
                        column_order=['Select','name', 'annual_return','lastday_return', 'sharpe_ratio', 'total_return', 'maxdrawdown', 'symbols', 'end_date'],
                        column_config={
                                "Select":           st.column_config.CheckboxColumn(required=True, width='small'),
                                "sharpe_ratio":     st.column_config.Column(width='small'),
                                "annual_return":    st.column_config.NumberColumn(required=True, format='%i%%', width='small'),
                                "lastday_return":    st.column_config.NumberColumn(required=True, format='%.1f%%', width='small'),    
                                "total_return":    st.column_config.NumberColumn(required=True, format='%i%%', width='small'),    
                                "maxdrawdown":    st.column_config.NumberColumn(required=True, format='%i%%', width='small'),        
                            },
                        disabled=['name', 'annual_return','lastday_return', 'sharpe_ratio', 'total_return', 'maxdrawdown', 'symbols', 'end_date'],
                    )
        selected_ids = list(edited_df[edited_df.Select].index)
        return selected_ids

def show_PortfolioTable(portfolio_df):
    ## using new st.data_editor
    def stringlist_to_set(strlist: list):
        slist = []
        for sstr in strlist:
            # for s in sstr.split(','):
            slist.append(sstr)
            
        slist = list(dict.fromkeys(slist))
        slist.sort()
        return(slist)
        
    symbols = stringlist_to_set(portfolio_df['symbols'].values)
    if 'symbolsSel' not in st.session_state:
        st.session_state['symbolsSel'] = symbols

    run_all = st.checkbox("Run All", key='run_all')

    df = selectpf_bySymbols(portfolio_df, st.session_state['symbolsSel'])
    selectpf = select_portfolios(df, default_selected=run_all)
    return(selectpf)

def show_PortforlioDetail(portfolio_df, index):
    if index > -1 and (index in portfolio_df.index):
        st.info('Selected portfolio:    ' + portfolio_df.at[index, 'name'])
        param_dict = json.loads(portfolio_df.at[index, 'param_dict'])
        pf = vbt.Portfolio.loads(portfolio_df.at[index, 'vbtpf'])
        display_pfbrief(pf=pf, param_dict=param_dict)
        st.markdown("**Description**")
        st.markdown(portfolio_df.at[index, 'description'], unsafe_allow_html=True)
        return True
    else:
        return False

def execute_trade(trader, side, symbol, price, volume, price_type="ATO"):
    st.write(f"Executing trade for {volume} shares of {symbol} at {price}...")
    # place_preorder(self, type='NB', symbol=None, price='', price_type='ATO', volume='0', start_date=None, end_date=None):
    type = 'NB' if side == 'Buy' else 'NS'
    price = int(price)
    volume = int(volume)
    price_type = price_type
    trader.place_preorder(
        type=type,
        symbol=symbol,
        price=price,
        volume=volume,
        price_type=price_type,
    )
    st.success("Trade executed successfully.")

def show_trade_form(prefix, row, trader):
    id = prefix + row['Side'] + row['Symbol']
    side = st.selectbox("Side", ["Buy", "Sell"], index=0 if row['Side'] == "Buy" else 1, key=f'{id}_side')
    price = st.number_input("Price", value=row['Price'], key=f'{id}_price')
    volume = st.number_input("Volume", value=row['Size'], key=f'{id}_volume')
    price_type = st.selectbox("Price Type", ["LO", "ATO", "ATC"], index=0, key=f'{id}_price_type')
    
    full_amount = st.checkbox("Full Amount", key=f'{id}_full_amount')
    
    amount = int(price * volume)
    st.write(f"**Total**: {amount:,} VND")
    
    if st.button("Execute", key=f'{id}_execute'):
        execute_trade(trader, side, row['Symbol'], price, volume, price_type)

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
        # st.write(f"updating portfolio('{portfolio.df.iloc[i]['name']}')")
        if not portfolio.update(portfolio.df.loc[index]['id']):
            st.error(f"Fail to update portfolio('{portfolio.df.iloc[index]['name']}')")

    last_trading_date = get_last_trading_date()
    first_trade_date_of_week =  get_first_trade_date_of_week()
    check_df = portfolio.check_records(dt=today, last_trading_date=last_trading_date, first_trade_date_of_week=first_trade_date_of_week)
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
