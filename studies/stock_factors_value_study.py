import pandas as pd
import numpy as np
import json

import streamlit as st
import vectorbt as vbt

import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots

from utils.component import check_password, input_dates, input_SymbolsDate
import matplotlib.pyplot as plt

from utils.plot_utils import plot_multi_line, plot_snapshot
from utils.processing import get_stocks, get_stocks_financial
from studies.rrg import plot_RRG, rs_ratio, RRG_Strategy
from utils.vbt import plot_pf
from vbt_strategy.MOM_D import get_MomDInd
from studies.magic_fomula_study import calculate_realtime_metrics, run as run_magic_fomula

import numpy as np
import pandas as pd


# # Function to calculate value score
# def calculate_value_score(metrics, means, stds):
#     pe = metrics['priceToEarning']
#     pb = metrics['priceToBook']
#     ev_ebitda = metrics['valueBeforeEbitda']
#     dividend_yield = metrics['dividend'] / metrics['earningPerShare']
    
#     # Normalize metrics using z-scores
#     z_pe = (pe - means['priceToEarning']) / stds['priceToEarning']
#     z_pb = (pb - means['priceToBook']) / stds['priceToBook']
#     z_ev_ebitda = (ev_ebitda - means['valueBeforeEbitda']) / stds['valueBeforeEbitda']
#     z_dividend_yield = (dividend_yield - means['dividend']) / stds['dividend']
    
#     # Composite value score
#     value_score = -z_pe - z_pb - z_ev_ebitda + z_dividend_yield
    
#     return value_score

# Define the magic formula function
def magic_formula(metrics):
    value_score = metrics['priceToEarning']
    return value_score

def run(symbol_benchmark, symbolsDate_dict):
    st.write("## Value Factor Study")
    
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
    # close,priceToEarning,priceToBook,valueBeforeEbitda,dividend,roe,roa,daysReceivable,daysInventory,daysPayable,ebitOnInterest,earningPerShare,bookValuePerShare,interestMargin,nonInterestOnToi,badDebtPercentage,provisionOnBadDebt,costOfFinancing,equityOnTotalAsset,equityOnLoan,costToIncome,equityOnLiability,currentPayment,quickPayment,epsChange,ebitdaOnStock,grossProfitMargin,operatingProfitMargin,postTaxMargin,debtOnEquity,debtOnAsset,debtOnEbitda,shortOnLongDebt,assetOnEquity,capitalBalance,cashOnEquity,cashOnCapitalize,cashCirculation,revenueOnWorkCapital,capexOnFixedAsset,revenueOnAsset,postTaxOnPreTax,ebitOnRevenue,preTaxOnEbit,preProvisionOnToi,postTaxOnToi,loanOnEarnAsset,loanOnAsset,loanOnDeposit,depositOnEarnAsset,badDebtOnAsset,liquidityOnLiability,payableOnEquity,cancelDebt,ebitdaOnStockChange,bookValuePerShareChange,creditGrowth
        
    # filter date > start_date
    union_df = union_df.loc[start_date:]
    
    # ffill
    union_df = union_df.fillna(method='ffill')
    
    union_df = calculate_realtime_metrics(union_df) 
    
    realtime_pe = pd.DataFrame(union_df['close'] / union_df['earningPerShare'])
    
    st.write("Realtime PE")    
    plot_multi_line(realtime_pe, title='Realtime PE', x_title='Date', y_title='PE', legend_title='Stocks')
    
    st.write("Realtime PB")
    realtime_pb = pd.DataFrame(union_df['close'] / union_df['bookValuePerShare'])
    
    plot_multi_line(realtime_pb, title='Realtime PB', x_title='Date', y_title='PB', legend_title='Stocks')
    
    plot_snapshot(realtime_pe, title='Realtime PE', x_title='Stocks', y_title='PE', legend_title='Stocks', sorted=False)