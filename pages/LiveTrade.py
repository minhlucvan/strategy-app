import pandas as pd
import numpy as np
import pytz
from datetime import datetime, date
import json

import streamlit as st

from utils.riskfolio import get_pfOpMS

st.set_page_config(page_title="BForecast Strategy App")

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

def show_PortfolioTable(portfolio_df, default_selected=True):
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

    sSel = st.multiselect("Please select symbols:", symbols, 
                                format_func=lambda x: x,
                                help='empty means all')

    df = selectpf_bySymbols(portfolio_df, st.session_state['symbolsSel'])
    selectpf = select_portfolios(df, default_selected=default_selected)
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

def main():
    # select date on the sidebar
    today = st.sidebar.date_input("Current Date", datetime.now().date())
    
    st.header("Portfolio Board")
    selected_pfs = []
    portfolio = Portfolio()
    selected_pfs = show_PortfolioTable(portfolio.df)

    ##多portfolio比较
    value_df = pd.DataFrame()
    position_df = pd.DataFrame()
    
    if len(selected_pfs) == 0:
        st.info("Please select portfolios.")
        st.stop()
    
    for index in selected_pfs:
        pf = vbt.Portfolio.loads(portfolio.df.loc[index, 'vbtpf'])
        value_df[portfolio.df.loc[index, 'name']] = pf.value()
        position_df[portfolio.df.loc[index, 'name']] = pf.position_mask()
        # show_PortforlioDetail(portfolio.df, index)

    ##无选择portforlio
    num_portfolio = len(portfolio.df)
    for i in range(num_portfolio):
        # st.write(f"updating portfolio('{portfolio.df.iloc[i]['name']}')")
        if not portfolio.update(portfolio.df.iloc[i]['id']):
            st.error(f"Fail to update portfolio('{portfolio.df.iloc[i]['name']}')")
            
    check_df = portfolio.check_records(dt=today)

    # send the notification to telegram users
    if len(check_df) == 0:
        st.success("No signal found.")
    else:
        st.markdown("### Trading Signals 📈")

        signals_by_name = check_df.groupby('Name')
        for name, signals in signals_by_name:
            st.write(f"##### {name} 📊")
            for i, row in signals.iterrows():
                # Formatted message with Markdown and emojis
                message = f"""
                        - ↔️ **Side:** {row['Side']}
                        - 💹 **Symbol:** {row['Symbol']}
                        - 💲 **Price:** {row['Price']}
                        - 📏 **Size:** {row['Size']}
                        - 🕒 **Timestamp:** {row['Timestamp']}
                        """
                st.markdown(message)

if __name__ == "__main__":
    if check_password():
        main()
