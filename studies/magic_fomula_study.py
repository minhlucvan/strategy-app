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

from utils.processing import get_stocks, get_stocks_funamental
from studies.rrg import plot_RRG, rs_ratio, RRG_Strategy
from utils.vbt import plot_pf
from vbt_strategy.MOM_D import get_MomDInd

from studies.market_wide import MarketWide_Strategy


def run(symbol_benchmark, symbolsDate_dict):
    
    with st.expander("Market Pricing Study"):
        st.markdown("""Market Pricing Study is a study that compares the market price of a bunch of stocks with their valuation. The valuation is calculated by the average of the PE and PB of the stocks. The market price is the close price of the stocks. The study is useful to identify the overvalued and undervalued stocks in the market
                    """)
        
    symbolsDate_dict['symbols'] =  symbolsDate_dict['symbols']
    # copy the symbolsDate_dict
    benchmark_dict = symbolsDate_dict.copy()
    benchmark_dict['symbols'] = [symbol_benchmark]
    
    # Assuming get_stocks and get_stocks_funamental are defined elsewhere
    stocks_df = get_stocks(symbolsDate_dict, stack=True)
    fundamentals_df = get_stocks_funamental(symbolsDate_dict, stack=True)
            
    # Ensure both indices are timezone-naive
    stocks_df.index = pd.to_datetime(stocks_df.index).tz_localize(None)
    fundamentals_df.index = pd.to_datetime(fundamentals_df.index).tz_localize(None)
    
    # filter the stocks_df and fundamentals_df to the same date range
    start_date = max(stocks_df.index[0], fundamentals_df.index[0])
    
    fundamentals_df = fundamentals_df.loc[start_date:]
    
    # Create a union of the indices
    union_index = stocks_df.index.union(fundamentals_df.index)
    
    # Reindex both DataFrames to the union index
    reindexed_stocks_df = stocks_df.reindex(union_index)
    reindexed_fundamental_df = fundamentals_df.reindex(union_index)
    
    union_df = pd.concat([reindexed_stocks_df, reindexed_fundamental_df], axis=1)
    #shortAsset,cash,shortInvest,shortReceivable,inventory,longAsset,fixedAsset,asset,debt,shortDebt,longDebt,equity,capital,centralBankDeposit,otherBankDeposit,otherBankLoan,stockInvest,customerLoan,badLoan,provision,netCustomerLoan,otherAsset,otherBankCredit,oweOtherBank,oweCentralBank,valuablePaper,payableInterest,receivableInterest,deposit,otherDebt,fund,unDistributedIncome,minorShareHolderProfit,payable,close

    # # Calculate Liquidity Ratios
    # union_df['Current Ratio'] = union_df['asset'] / (union_df['debt'] + union_df['shortDebt'])
    # union_df['Quick Ratio'] = (union_df['asset'] - union_df['inventory']) / (union_df['debt'] + union_df['shortDebt'])

    # # Calculate Solvency Ratios
    # union_df['Debt-to-Equity Ratio'] = (union_df['debt'] + union_df['shortDebt']) / union_df['equity']
    # union_df['Debt Ratio'] = (union_df['debt'] + union_df['shortDebt']) / union_df['asset']
    # union_df['Equity Ratio'] = union_df['equity'] / union_df['asset']

    # # Calculate Efficiency Ratios (assuming 'Cost of Goods Sold' and 'Net Credit Sales' are available)
    # union_df['Inventory Turnover Ratio'] = union_df['Cost of Goods Sold'] / union_df['inventory']
    # union_df['Receivables Turnover Ratio'] = union_df['Net Credit Sales'] / union_df['shortReceivable']

    # # Calculate Profitability Ratios
    # union_df['Gross Profit Margin'] = (union_df['Revenue'] - union_df['Cost of Goods Sold']) / union_df['Revenue']
    # union_df['Net Profit Margin'] = union_df['close'] / union_df['Revenue']  # Assuming 'close' represents net income
    # union_df['ROA'] = union_df['close'] / union_df['asset']
    # union_df['ROE'] = union_df['close'] / union_df['equity']

    # # Calculate Coverage Ratios (assuming 'EBIT' and 'Interest Expense' are available)
    # union_df['Interest Coverage Ratio'] = union_df['EBIT'] / union_df['payableInterest']
    # union_df['DSCR'] = union_df['netCustomerLoan'] / (union_df['debt'] + union_df['shortDebt'])

    # # Additional calculations can be performed based on available data and specific requirements.

    # # Print or return the DataFrame with calculated metrics
    # print(union_df[['Current Ratio', 'Quick Ratio', 'Debt-to-Equity Ratio', 'Debt Ratio', 'Equity Ratio', 'Inventory Turnover Ratio', 'Receivables Turnover Ratio', 'Gross Profit Margin', 'Net Profit Margin', 'ROA', 'ROE', 'Interest Coverage Ratio', 'DSCR']])

    current_ratio = union_df['asset'] / (union_df['debt'] + union_df['shortDebt'])
    
    st.write(current_ratio)
    
    roe = union_df['close'] / union_df['equity']
    
    st.write(roe)