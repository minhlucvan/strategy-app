from datetime  import datetime, date, timedelta
import pytz
import numpy as np

import streamlit as st
from streamlit_quill import st_quill
import utils.data_vn as data_vn
import utils.data_bin as data_bin

from utils.market_utils import get_maket_groups
from utils.portfolio import Portfolio

def input_dates(by='unique'):
    start_date = st.sidebar.date_input("Start date?", date(2018, 1, 1), key=by+'_start_date')
    end_date = st.sidebar.date_input("End date?", date.today(),  key=by+'_end_date')
    start_date = datetime(year=start_date.year, month=start_date.month, day=start_date.day, tzinfo=pytz.utc)
    end_date = datetime(year=end_date.year, month=end_date.month, day=end_date.day, tzinfo=pytz.utc)
    return start_date, end_date

def form_SavePortfolio(symbolsDate_dict, strategyname:str, strategy_param:dict, pf, assets_name:str=None, key=None):
    with st.expander("Edit description and Save"):
        with st.form("form_" + strategyname if key is None else key):
            tittle = st.text_input("Portfolio Name", value= f"{strategyname}_{assets_name}")
            # desc_str = st_quill(value= f"{strategyname},  Param_dict: {strategy_param}", html= True)
            desc_str = st.text_area("Description", value= f"{strategyname},  Param_dict: {strategy_param}")
            submitted = st.form_submit_button("Save")
            if submitted:
                portfolio = Portfolio()
                if portfolio.add(symbolsDate_dict, strategyname, strategy_param, pf, desc_str, name=tittle):
                    st.success("Save the portfolio sucessfully.")
                else:
                    st.error('Fail to save the portfolio.')

def check_password():
    # hide_bar()
    """Returns `True` if the user had the correct password."""
    hide_streamlit_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
        """
    # st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if "password" not in st.session_state:
            st.session_state["password"] = st.secrets["password"]
            
        if st.session_state["password"] == st.secrets["password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    if st.secrets["password"]=="":
        return True
        
    if "password_correct" not in st.session_state:
        if "password" not in st.session_state:
            # First run, show input for password.
            st.text_input(
                "Password", type="password", on_change=password_entered, key="password"
            )
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 Password incorrect")
        return False
    else:
        # Password correct.
        # show_bar()
        return True

def hide_bar():
    bar= """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
            visibility:hidden;
            width: 0px;
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
            visibility:hidden;
        }
        </style>
    """
    st.markdown(bar, unsafe_allow_html=True)

def show_bar():
    bar= """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
            visibility:visible;
            width: 0px;
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
            visibility:visible;
        }
        </style>
    """
    st.markdown(bar, unsafe_allow_html=True)

def input_SymbolsGroup(groups_dict):        
    group_options = groups_dict['group'].keys()
    group_options = ['None'] + list(group_options)
    group_sel = st.sidebar.selectbox("Symbols' group", group_options)
    
    if group_sel == 'None':
        return {
            "name": None,
            "benchmark": groups_dict['benchmark'],
            "symbols": []
        }
    
    return {
        "benchmark": groups_dict['benchmark'],
        "symbols": groups_dict['group'][group_sel],
        "name": group_sel
    }

def input_Symbols_wildcard(market):
    symbols_string = st.sidebar.text_input("Enter Tickers", '', key="textinput" + "_symbols")
    symbols = []
    if len(symbols_string) > 0:
        symbols = symbols_string.strip().split(',')
        
    return symbols

def input_symbols_group(group_dict):
    select_all = st.sidebar.checkbox("Select all symbols", value= False)
    default_symbols = group_dict['symbols'] if select_all else []
    # multi select symbols from group's symbols
    group_symbols = st.sidebar.multiselect("Select symbols from group", group_dict['symbols'], default_symbols)
    return group_symbols

def input_SymbolsDate(group=True) -> dict:
    # market = st.sidebar.radio("Select market", ("VN",), horizontal= True)
    market = st.sidebar.selectbox("Select market", ("VN", "BIN"))
    
    groups_data = get_maket_groups(market)
    
    group_dict = {
        "benchmark": groups_data['benchmark'],
        "symbols": []
    }
    
    symbols = []
    
    if group:
        group_dict = input_SymbolsGroup(groups_data)
    
    if group and len(group_dict['symbols']) > 0:
        symbols = input_symbols_group(group_dict)
    else:
        symbols = input_Symbols_wildcard(market)

    start_date = st.sidebar.date_input("Start date?", date(2018, 1, 1))
    end_date = st.sidebar.date_input("End date?", date.today()- timedelta(days = 1))
    start_date = datetime(year=start_date.year, month=start_date.month, day=start_date.day, tzinfo=pytz.utc)
    end_date = datetime(year=end_date.year, month=end_date.month, day=end_date.day, tzinfo=pytz.utc)
    
    group_symbols = group_dict['symbols']
    benmark_symbols = group_dict['benchmark']
    group_name = group_dict['name'] if 'name' in group_dict else None
    
    # upper case
    symbols = [symbol.upper() for symbol in symbols]
    
    # timeframe
    timeframes = ['1D', '1W', '1M', '4h', '1h', '15m', '5m']
    
    if market == 'VN':
        timeframes = data_vn.get_intervals()
    elif market == 'BIN':
        timeframes = data_bin.get_intervals()
    
    timeframe = st.sidebar.selectbox('Timeframe', timeframes, index=0)
    
    return {
            "market":   market,
            "symbols":  symbols,
            "start_date": start_date,
            "end_date": end_date,
            "goup_symbols": group_symbols,
            "benchmark": benmark_symbols,
            "group_name": group_name,
            "timeframe": timeframe
        }

def params_selector(params):
    params_parse = dict()
    st.write("**Optimization Parameters:**")

    col1, col2 = st.columns([3, 1])
    with col1:
        params_parse['RARM'] = st.selectbox('Risk Adjusted Return Method', 
                        ['sharpe_ratio', 'annualized_return', 'deflated_sharpe_ratio', 'calmar_ratio', 'sortino_ratio', 
                         'omega_ratio', 'information_ratio', 'tail_ratio'])
    with col2:
        params_parse['WFO'] = st.selectbox("Walk Forward Optimization",
                        ['None', 'Non-anchored', 'Anchored'])

    for param in params:
        col1, col2 = st.columns([3, 1])
        with col1:
            if param.get("control", "range") == "single":
                value = st.slider(f"Select {param['name']}", min_value=param["min"], max_value=param["max"], step=param["step"], value=param["value"])
                params_parse[param["name"]] = [value]
            else:
                gap = (param["max"] - param["min"]) * 0.5
                if param["type"] == 'int':
                    gap = int(gap)
                    bottom = max(0, param["min"] - gap)
                else:
                    bottom = max(0.0, param["min"] - gap)
                
                values = st.slider(f"Select a range of {param['name']}", bottom, param['max'] + gap, (param["min"], param["max"]))
                step_number = st.number_input(f"Step of {param['name']}", value=param["step"])
                
                if step_number == 0:
                    params_parse[param["name"]] = [values[0]]
                else:
                    params_parse[param["name"]] = np.arange(values[0], values[1], step_number).tolist()
    
    return params_parse
