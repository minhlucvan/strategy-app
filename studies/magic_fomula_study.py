import pandas as pd
import numpy as np
import json

import streamlit as st
import vectorbt as vbt

import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots # creating subplots


from utils.component import  check_password, input_dates, input_SymbolsDate
import matplotlib.pyplot as plt

from utils.processing import get_stocks, get_stocks_financial
from studies.rrg import plot_RRG, rs_ratio, RRG_Strategy
from utils.vbt import plot_pf
from vbt_strategy.MOM_D import get_MomDInd

from studies.market_wide import MarketWide_Strategy

def plot_multi_line(df, title, x_title, y_title, legend_title):
    fig = go.Figure()
    for col in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df[col], mode='lines', name=col))
    fig.update_layout(title=title, xaxis_title=x_title, yaxis_title=y_title, legend_title=legend_title)
    st.plotly_chart(fig, use_container_width=True)


def run(symbol_benchmark, symbolsDate_dict):
    
    with st.expander("Market Pricing Study"):
        st.markdown("""Market Pricing Study is a study that compares the market price of a bunch of stocks with their valuation. The valuation is calculated by the average of the PE and PB of the stocks. The market price is the close price of the stocks. The study is useful to identify the overvalued and undervalued stocks in the market
                    """)
        
    symbolsDate_dict['symbols'] =  symbolsDate_dict['symbols']
    
    if len(symbolsDate_dict['symbols']) < 1:
        st.info("Please select symbols.")
        st.stop()
    
    # copy the symbolsDate_dict
    benchmark_dict = symbolsDate_dict.copy()
    benchmark_dict['symbols'] = [symbol_benchmark]
    
    # Assuming get_stocks and get_stocks_funamental are defined elsewhere
    stocks_df = get_stocks(symbolsDate_dict, stack=True)
    financial_df = get_stocks_financial(symbolsDate_dict, stack=True)
            
    # Ensure both indices are timezone-naive
    stocks_df.index = pd.to_datetime(stocks_df.index).tz_localize(None)
    financial_df.index = pd.to_datetime(financial_df.index).tz_localize(None)
    
    # filter the stocks_df and financial_df to the same date range
    start_date = pd.to_datetime(symbolsDate_dict['start_date']).tz_localize(None)
    
    # Create a union of the indices
    union_index = stocks_df.index.union(financial_df.index)
    
    # Reindex both DataFrames to the union index
    reindexed_stocks_df = stocks_df.reindex(union_index)
    reindexed_fundamental_df = financial_df.reindex(union_index)
    
    union_df = pd.concat([reindexed_stocks_df, reindexed_fundamental_df], axis=1)
    #shortAsset,cash,shortInvest,shortReceivable,inventory,longAsset,fixedAsset,asset,debt,shortDebt,longDebt,equity,capital,centralBankDeposit,otherBankDeposit,otherBankLoan,stockInvest,customerLoan,badLoan,provision,netCustomerLoan,otherAsset,otherBankCredit,oweOtherBank,oweCentralBank,valuablePaper,payableInterest,receivableInterest,deposit,otherDebt,fund,unDistributedIncome,minorShareHolderProfit,payable,close
    
    # filter date > start_date
    union_df = union_df.loc[start_date:]
    
    # reverse engineering the financials
    # caculate equity = priceToEquity / PE
    
    
    # fill missing values with the last available value
    union_df = union_df.fillna(method='ffill')
    
    # index (code, factor)
    metrics = union_df.columns.get_level_values(0).unique()
    selected_metrics = st.selectbox('Select Metrics to Plot', metrics)
    
    plot_multi_line(union_df[selected_metrics], f'{selected_metrics} of Stocks and Financials', 'Date', selected_metrics, 'Stocks and Financials')