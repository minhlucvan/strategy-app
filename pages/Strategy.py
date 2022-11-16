import streamlit as st

from utils.component import input_SymbolsDate, check_password, params_selector, form_SavePortfolio
from utils.db import get_symbolname

def check_params(params):
    # for key, value in params.items():
    #     if len(params[key]) < 2:
    #         st.error(f"{key} 's numbers are not enough. ")
    #         return False
    return True

def get_symbolsname(symbols):
    names = set()
    for symbol in symbols:
        names.add(get_symbolname(symbol))
    return names


if check_password():
    strategy_list = getattr(__import__(f"vbt_strategy"), 'strategy_list')
    strategyname = st.sidebar.selectbox("Please select strategy", strategy_list)
    if strategyname:
        symbolsDate_dict = input_SymbolsDate()
        if len(symbolsDate_dict['symbols']) > 0:
            st.header(strategyname)
            strategy_cls = getattr(__import__(f"vbt_strategy"), strategyname + 'Strategy')
            strategy = strategy_cls(symbolsDate_dict)
            st.markdown(strategy.desc, unsafe_allow_html= True)
            if len(strategy.stock_dfs) > 0:
                st.subheader("Stocks:    " + ' , '.join(get_symbolsname(symbolsDate_dict['symbols'])))
                params = params_selector(strategy.param_def)
                if check_params(params):
                    if strategy.maxSR(params, output_bool=True):
                        st.text("Max Sharpe_Ratio's parameters:    " + str(strategy.param_dict))
                        form_SavePortfolio(symbolsDate_dict, strategyname, strategy.param_dict, strategy.pf)
                    else:
                        st.error("Stocks don't match the Strategy.")
            else:
                st.error("None of stocks is valid.")
