import streamlit as st
import pandas as pd
import json
import vectorbt as vbt
from utils.component import check_password, params_selector
from utils.vbt import plot_pf
from utils.portfolio import selectpf_bySymbols
from utils.vbt import display_pfbrief
from vbt_strategy.PairTrade import pairtrade_pfs
from utils.db import get_SymbolName, get_SymbolsNames

def check_params(params):
    # for key, value in params.items():
    #     if len(params[key]) < 2:
    #         st.error(f"{key} 's numbers are not enough. ")
    #         return False
    return True


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