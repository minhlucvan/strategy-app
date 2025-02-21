import streamlit as st
import pandas as pd
import json

from utils.component import input_SymbolsDate, check_password, params_selector, form_SavePortfolio
from utils.db import get_SymbolsName, get_SymbolsNames
from utils.st import check_params


if check_password():
    strategy_list = getattr(__import__(f"vbt_strategy"), 'strategy_list')
    strategyName = st.sidebar.selectbox("Please select strategy", strategy_list)
    
    if strategyName:
        symbolsDate_dict = input_SymbolsDate()
        
        # if len(symbolsDate_dict['symbols']) < 1:
        #     st.info("Please select symbols.")
        #     st.stop()

        if len(symbolsDate_dict['symbols']) > 0:
            st.header(strategyName)
            strategy_cls = getattr(__import__(f"vbt_strategy"), strategyName + 'Strategy')
            strategy = strategy_cls(symbolsDate_dict)
            with st.expander("Description:"):
                st.markdown(strategy.desc, unsafe_allow_html= True)
            if strategy.validate():
                st.subheader("Stocks:    " + ' , '.join(get_SymbolsNames(symbolsDate_dict['symbols'])))
                params = params_selector(strategy.param_def)
                if check_params(params):
                    if strategy.maxRARM(params, output_bool=True):
                        st.text(f"Maximize Target's Parameters:    ")
                        st.write(json.dumps(strategy.param_dict, indent=4))
                        asset_name = strategy.get_assets_identifier()
                        form_SavePortfolio(symbolsDate_dict, strategyName, strategy.param_dict, strategy.pf, asset_name)
                    else:
                        st.error("Strategy failed to maximize the Target's Parameters.")
            else:
                st.error("Stocks don't match the Strategy.")

