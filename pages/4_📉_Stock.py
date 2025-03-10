import streamlit as st

from utils.component import input_SymbolsDate, check_password, form_SavePortfolio, params_selector
from utils.db import get_SymbolName
from utils.stock_utils import load_stock_overview_cached, get_stock_ratio
from utils.vbt import display_pfbrief
import pandas as pd
from studies.magic_fomula_study import run as run_magic_fomula
import utils.db as db
import utils.vnstock as vn

def show_stock(symbol):
    st.write("## Stocks insignt board")
        
    st.write("Company profile")
    symbol = symbolsDate_dict['symbols'][0]
    stock_overview = load_stock_overview_cached(symbol)
    stock_info = get_stock_ratio(symbol)
    
    st.write("Overview")
    st.table(stock_overview)
    
    # display dictionary as table
    st.write("Ratio")
    st.table(stock_info)
    
    st.write("Historical data")
    symbol_benchmark = 'VN30'
    run_magic_fomula(symbol_benchmark=symbol_benchmark, symbolsDate_dict=symbolsDate_dict)

def show_stock_compare(symbols):
    st.write("## Stocks insignt board")
    st.write("Coming soon...")
    
    st.write("Company profile")
    overview_dict = {}
    
    for symbol in symbols:
        stock_overview = load_stock_overview_cached(symbol)
        overview_dict[symbol] = stock_overview
    
    overview_df = pd.DataFrame(overview_dict)
    
    st.write("Overview")
    st.table(overview_df)

    st.write("Financials")
    
    ratio_dict = {}
    
    for symbol in symbols:
        stock_info = get_stock_ratio(symbol)
        ratio_dict[symbol] = stock_info
        
    ratio_df = pd.DataFrame(ratio_dict)
    
    st.write("Ratio")
    st.table(ratio_df)
    
    st.write("Historical data")
    symbol_benchmark = 'VN30'
    run_magic_fomula(symbol_benchmark=symbol_benchmark, symbolsDate_dict=symbolsDate_dict)
    
def show_stocks(symbolsDate_dict: dict):
    st.write("## Stocks insignt board")

    df = db.load_symbols()   
    
    exchanges = df['exchange'].unique()
    symbols = df['symbol'].unique()
    
    selected_exchanges = st.multiselect("Select exchange", exchanges)
    selected_symbols = st.multiselect("Select symbols", symbols)
    
    if len(selected_symbols) > 0:
        df = df[df['symbol'].isin(selected_symbols)]
        
    if len(selected_exchanges) > 0:
        df = df[df['exchange'].isin(selected_exchanges)]
    
    st.write("Stocks")
    st.dataframe(df)

    
if check_password():
    symbolsDate_dict = input_SymbolsDate()
    
    if len(symbolsDate_dict['symbols']) < 1:
        show_stocks(symbolsDate_dict)
        st.stop()
    
    if len(symbolsDate_dict['symbols']) == 1:
        show_stock(symbolsDate_dict['symbols'][0])
        
    if len(symbolsDate_dict['symbols']) > 1:
        show_stock_compare(symbolsDate_dict['symbols'])