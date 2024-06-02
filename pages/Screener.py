import streamlit as st
import pandas as pd
import json
import plotly.graph_objects as go

from indicators.RSC import get_RSCInd
from indicators.ReturnRSC import get_ReturnRSCInd
from utils.component import input_SymbolsDate, check_password, params_selector, form_SavePortfolio
from utils.db import get_SymbolsName
from utils.processing import get_stocks


def check_params(params):
    # for key, value in params.items():
    #     if len(params[key]) < 2:
    #         st.error(f"{key} 's numbers are not enough. ")
    #         return False
    return True

def plot_RSC(rs_df):
    fig = go.Figure()
    for symbol in rs_df.columns:
        fig.add_trace(go.Scatter(x=rs_df.index, y=rs_df[symbol], mode='lines', name=symbol))
    st.plotly_chart(fig)

def plot_price(price_df):
    fig = go.Figure()
    for symbol in price_df.columns:
        fig.add_trace(go.Scatter(x=price_df.index, y=price_df[symbol], mode='lines', name=symbol))
    st.plotly_chart(fig)

if check_password():
    symbolsDate_dict = input_SymbolsDate(group=True)
    
    if len(symbolsDate_dict['symbols']) < 1:
        st.info("Please select symbols.")
        st.stop()
        
    st.write(f"### Study: Relative Strength Comparison")
        
    symbolsDate_dict['symbols'] = symbolsDate_dict['symbols'] + [symbolsDate_dict['benchmark']]
    stocks_df = get_stocks(symbolsDate_dict, 'close')
    benchmark_df = stocks_df[symbolsDate_dict['benchmark']]
    
    # drop benchmark from stocks_df
    stocks_df = stocks_df.drop(columns=[symbolsDate_dict['benchmark']])

    rs_indicator = get_RSCInd().run(stocks_df, benchmark_df)
    
    # plot price
    plot_price(stocks_df)
    
    # plot RSC
    plot_RSC(rs_indicator.rs)
    
    # top 10
    rs_mean = rs_indicator.rs.mean()
    st.write("### Rolling RS")
    
    rs_smooth = rs_indicator.rs.rolling(window=20).mean()
    rs_pct_change = rs_smooth.pct_change()
    rs_cumsum = rs_pct_change.cumsum()
    
    plot_RSC(rs_cumsum)
    
    final_rs = rs_cumsum.iloc[-1]
    
    final_rs = final_rs.sort_values(ascending=False)
    
    # plot final RS
    
    fig = go.Figure()
    fig.add_trace(go.Bar(x=final_rs.index, y=final_rs, name='RS'))
    st.plotly_chart(fig)