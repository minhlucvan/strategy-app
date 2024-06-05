import streamlit as st

from utils.component import input_SymbolsDate, check_password, form_SavePortfolio, params_selector
from utils.db import get_SymbolName
from utils.vbt import display_pfbrief

if check_password():
    symbolsDate_dict = input_SymbolsDate()
    
    if len(symbolsDate_dict['symbols']) < 1:
        st.info("Please select symbols.")
        st.stop()
    
    if len(symbolsDate_dict['symbols']) > 0:
        st.header(f"{get_SymbolName(symbolsDate_dict['symbols'][0])} Strategies' comparision board")
        strategy_list = getattr(__import__(f"vbt_strategy"), 'strategy_list')
        
        defalut_strategies = []
        selected_strategies = st.multiselect("Please select strategies", strategy_list, defalut_strategies)
        
        params = params_selector({})
        for strategyname in selected_strategies:
            strategy_cls = getattr(__import__(f"vbt_strategy"), strategyname + 'Strategy')
            strategy = strategy_cls(symbolsDate_dict)
            if len(strategy.stock_dfs) > 0:
                if strategy.maxRARM(params, output_bool=False):
                    st.info(f"Strategy '{strategyname}' Max {strategy.param_dict['RARM']} Result")
                    display_pfbrief(strategy.pf, strategy.param_dict)
                    form_SavePortfolio(symbolsDate_dict, strategyname, strategy.param_dict, strategy.pf)
            else:
                st.error("None of stocks is valid.")

