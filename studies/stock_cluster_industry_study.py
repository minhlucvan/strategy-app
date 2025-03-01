from utils.processing import get_stocks
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from sklearn.preprocessing import StandardScaler
import utils.plot_utils as pu
import utils.stock_utils as su

def fetch_stock_overview(symbolsDate_dict):
    overview_dict = {}
    
    symbols = symbolsDate_dict['symbols']
    
    for symbol in symbols:
        stock_overview = su.load_stock_overview_cached(symbol)
        overview_dict[symbol] = stock_overview
    
    overview_df = pd.DataFrame(overview_dict)
    
    # transform overview_df T
    overview_df = overview_df.T
    
    return overview_df

def run(symbol_benchmark, symbolsDate_dict):
    if st.button("Fetch Data"):
        overview_df = fetch_stock_overview(symbolsDate_dict)
        overview_df.to_csv("data/stock_overview.csv")
        st.success("Data fetched and saved successfully")
    else:
        try:
            overview_df = pd.read_csv("data/stock_overview.csv", index_col=0)
            st.success("Data loaded from CSV")
        except FileNotFoundError:
            st.error("CSV file not found. Please fetch data first.")
            st.stop()
    
    if st.button("Save Overview"):
        overview_df.to_csv("data/stock_overview.csv")
        st.success("Overview saved successfully")

    st.write("Overview")
    
    
    stocks_df = get_stocks(symbolsDate_dict, 'close') if len(symbolsDate_dict['symbols']) > 0 else None
    
       
    all_industries = su.get_all_industries()
    
    
    indices_df = su.construct_multi_index_df(stocks_df, all_industries)
    
    pu.plot_multi_line(indices_df, title="All Industries")
    
    
    selected_industry = st.selectbox("Select industry", all_industries)
    
    st.write(f"Industry: {selected_industry}")
    
    if stocks_df is not None:
        stocks_industry_df = su.filter_stocks_by_industry(stocks_df, selected_industry)
        
        pu.plot_multi_line(stocks_industry_df, title=f"Stocks in {selected_industry}")
        
        index_df = su.construct_index_df(stocks_industry_df)
        
        pu.plot_single_line(index_df, title=f"{selected_industry} Index")