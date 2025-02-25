import pandas as pd
import numpy as np
import json

import streamlit as st
import vectorbt as vbt

import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots # creating subplots

import utils.plot_utils as pu
from utils.component import  check_password, input_dates, input_SymbolsDate
import matplotlib.pyplot as plt

from utils.processing import get_stocks, get_stocks_income_statement

def run(symbol_benchmark, symbolsDate_dict):
        
    if len(symbolsDate_dict['symbols']) == 0:
        st.info("No symbols selected.")
        return
    
    symbolsDate_dict['symbols'] =  symbolsDate_dict['symbols']
    # copy the symbolsDate_dict
    benchmark_dict = symbolsDate_dict.copy()
    benchmark_dict['symbols'] = [symbol_benchmark]
    
    benchmark_df = get_stocks(benchmark_dict,'close')
    stocks_df = get_stocks(symbolsDate_dict,'close')
   

    income_statement_df = get_stocks_income_statement(symbolsDate_dict, stack=True)
    
    #     "ticker": "VND",
    #     "quarter": 4,
    #     "year": 2024,
    #     "revenue": 1212,
    #     "yearRevenueGrowth": -0.373,
    #     "quarterRevenueGrowth": -0.046,
    #     "costOfGoodSold": -630,
    #     "grossProfit": 582,
    #     "operationExpense": -124,
    #     "operationProfit": 458,
    #     "yearOperationProfitGrowth": -0.637,
    #     "quarterOperationProfitGrowth": -0.423,
    #     "interestExpense": -182,
    #     "preTaxProfit": 275,
    #     "postTaxProfit": 251,
    #     "shareHolderIncome": 251,
    #     "yearShareHolderIncomeGrowth": -0.694,
    #     "quarterShareHolderIncomeGrowth": -0.502,
    #     "investProfit": null,
    #     "serviceProfit": null,
    #     "otherProfit": null,
    #     "provisionExpense": null,
    #     "operationIncome": null,
    #     "ebitda": 475
    
    quatertly_revenue = income_statement_df['revenue']
    quatertly_revenue_growth = income_statement_df['quarterRevenueGrowth']
    quatertly_operation_profit = income_statement_df['operationProfit']
    quatertly_operation_profit_growth = income_statement_df['quarterOperationProfitGrowth']
    quatertly_shareholder_income = income_statement_df['shareHolderIncome']
    quatertly_shareholder_income_growth = income_statement_df['quarterShareHolderIncomeGrowth']
    quatertly_ebitda = income_statement_df['ebitda']
    quatertly_ebitda_growth = quatertly_ebitda.pct_change()
    
    # index reset timezone
    # income_statement_df.index = income_statement_df.index.tz_localize(None)
    
    stock_end_df = stocks_df.reindex(income_statement_df.index, method='ffill')
    # stock_end_df is price of previous statement
    stock_start_df = stock_end_df.shift(1)

    stock_return_df = (stock_end_df - stock_start_df) / stock_start_df
    
    pu.plot_multi_line(stock_end_df, title='Stocks Close Price', x_title='Date', y_title='Close Price')
    
    pu.plot_multi_line(quatertly_revenue, title='Quatertly Revenue', x_title='Date', y_title='Revenue')
   
    st.write("### Stock Return")
    pu.plot_multi_line(stock_return_df, title='Stocks Return', x_title='Date', y_title='Return')
    
    
    # quatertly_revenue_growth
    st.write("### Quatertly Revenue Growth")
    pu.plot_multi_line(quatertly_revenue_growth, title='Quatertly Revenue Growth', x_title='Date', y_title='Growth')
    
    # ---
    return_long_df = pd.melt(stock_return_df, ignore_index=False, var_name='symbol', value_name='return')

    ebitda_long_df = pd.melt(quatertly_ebitda, ignore_index=False, var_name='symbol', value_name='ebitda')
    ebitda_merged_flat_df = pd.merge(ebitda_long_df, return_long_df, left_index=True, right_index=True)
    ebitda_merged_flat_df = ebitda_merged_flat_df.reset_index()
    ebitda_merged_flat_df['date'] = ebitda_merged_flat_df.index
    ebitda_merged_flat_df = ebitda_merged_flat_df.dropna()
    
    # plot the scatter plot
    # show the relationship between ebitda and return
    fig = px.scatter(ebitda_merged_flat_df, x='ebitda', y='return', color='symbol_x', trendline='ols')
    fig.update_layout(title='EBITDA vs Return', xaxis_title='EBITDA', yaxis_title='Return')
    st.plotly_chart(fig)
    
    # ---
    
    ebitda_growth_long_df = pd.melt(quatertly_ebitda_growth, ignore_index=False, var_name='symbol', value_name='ebitda_growth')
    ebitda_growth_merged_flat_df = pd.merge(ebitda_growth_long_df, return_long_df, left_index=True, right_index=True)
    ebitda_growth_merged_flat_df = ebitda_growth_merged_flat_df.reset_index()
    ebitda_growth_merged_flat_df['date'] = ebitda_growth_merged_flat_df.index
    ebitda_growth_merged_flat_df = ebitda_growth_merged_flat_df.dropna()
    
    # plot the scatter plot
    # show the relationship between ebitda growth and return
    fig = px.scatter(ebitda_growth_merged_flat_df, x='ebitda_growth', y='return', color='symbol_x', trendline='ols')
    fig.update_layout(title='EBITDA Growth vs Return', xaxis_title='EBITDA Growth', yaxis_title='Return')
    st.plotly_chart(fig)