import streamlit as st
import pandas as pd
import json

from utils.component import input_SymbolsDate, check_password, params_selector, form_SavePortfolio
from utils.db import get_SymbolsName

def check_params(params):
    # for key, value in params.items():
    #     if len(params[key]) < 2:
    #         st.error(f"{key} 's numbers are not enough. ")
    #         return False
    return True

if check_password(): 
    study_list = getattr(__import__(f"studies"), 'study_list')
    studyNames = [study['name'] for study in study_list]
    studyName = st.sidebar.selectbox("Please select Study", studyNames)
    study = [study['module'] for study in study_list if study['name'] == studyName][0]
    
    symbolsDate_dict = input_SymbolsDate(group=True)

    symbol_benchmark = symbolsDate_dict['benchmark']
    
    if len(symbolsDate_dict['symbols']) < 1:
        st.info("Please select symbols.")
        st.stop()
    
    if studyName and study is not None:
        st.write(f"### Study: {studyName}")        
        try:
            study(symbol_benchmark, symbolsDate_dict)
        except Exception as e:
            print(e)
            st.error(e)
    else:
        st.write("Please select a study.")
        