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

from utils.processing import get_stocks
from utils.vbt import plot_pf
from vbt_strategy.MOM_D import get_MomDInd
from studies.magic_fomula_study import run as run_magic_fomula

import numpy as np
import pandas as pd

def calculate_rotce(metrics):
    # Assuming Net PPE (Net Fixed Assets) = Gross PPE - Accumulated Depreciation
    net_ppe = metrics['capitalBalance']  # Assuming this corresponds to Net Fixed Assets
    
    # Assuming Net Working Capital = Current Assets - Current Liabilities
    # Current Assets and Current Liabilities are not directly provided, using Current and Quick Payments as proxies
    # quickPayment = (Current Assets - Inventory) / Current Liabilities
    # currentPayment = Current Assets / Current Liabilities
    # Assume Inventory can be derived from daysInventory
    current_assets = metrics['quickPayment'] * metrics['currentPayment']  # Simplified assumption
    current_liabilities = current_assets / metrics['currentPayment']
    net_working_capital = current_assets - current_liabilities
    
    # Tangible Capital Employed
    tangible_capital_employed = net_ppe + net_working_capital
    
    # ROTCE
    ebit = metrics['ebitOnRevenue'] * metrics['revenueOnAsset'] * tangible_capital_employed  # Assuming some derived EBIT
    rotce = ebit / tangible_capital_employed
    return rotce

def calculate_earnings_yield(metrics):
    # Assuming Enterprise Value = Market Cap + Total Debt - Cash and Cash Equivalents
    market_cap = metrics['priceToEarning'] * metrics['earningPerShare']  # Simplified assumption
    total_debt = metrics['debtOnEquity'] * (market_cap * metrics['equityOnTotalAsset'])  # Simplified assumption
    cash_and_equivalents = metrics['cashOnCapitalize'] * market_cap  # Simplified assumption
    
    enterprise_value = market_cap + total_debt - cash_and_equivalents
    
    # Earnings Yield
    ebit = metrics['ebitOnRevenue'] * metrics['revenueOnAsset'] * (market_cap + total_debt)  # Assuming some derived EBIT
    earnings_yield = ebit / enterprise_value
    return earnings_yield

def magic_formula(metrics):
    rotce = calculate_rotce(metrics)
    earnings_yield = calculate_earnings_yield(metrics)
    
    magic_score = (rotce + earnings_yield) / 2
    return magic_score

def run(symbol_benchmark, symbolsDate_dict):
    st.write("## Steven's Magic Formula")
    
    with st.expander("Methodology and Approach"):
        st.markdown("""
        ### Methodology and Approach
        https://medium.com/@steffenjanbrouwer/magic-formula-stock-tracker-app-functional-description-9b18f6f2698e
        
Return on Tangible Capital Employed (ROTCE)
ROTCE = EBIT / (Net Fixed Assets + Net Working Capital)
ROTCE is a financial metric used to measure a company’s efficiency and it says something about a company’s growth opportunities. If return on capital is low, the company will not be able to grow quickly based on retained earnings or other sources of capital. The following calculations are made in the app:

Tangible Capital Employed = Net Working Capital + Net Fixed Assets
where,

Net Fixed Assets = Net PPE = Gross PPE — Accumulated Depreciation
Net Working Capital = Current Assets — Current Liabilities
Earnings Yield
Earnings Yield = EBIT / Enterprise value
Earnings yield is a financial metric that represents the earnings generated for each dollar invested in a company. It indicates whether the company is relatively expensive for its level of earnings. The following calculation is made in the app:

Enterprise Value = Market Cap + Total Debt — Cash and Cash equivalents
Stock prices in the app are updated intraday. For fundamentals (balance sheet and income statement) annual or, where available, trailing twelve months (TTM) is used.

Ranking stocks versus Magic score
Next the ROTCE and Earnings Yield of all stocks are ranked from highest to lowest, where a higher metric value means a better ranking. Thus, you will have two rankings, one ranking of ROTCE and one ranking of Earnings Yield. Next you add the ranking of ROTCE to the ranking of Earnings Yield for a certain stock. The better the combined ranking, the better the buy!

However within the app you will see a Magic score. This is a personal deviation from the Book by Joel Greenblatt. There are three reasons for this. In the original Magic Formula I didn't like that ROTCE and Earnings yield where both equally important for the total ranking. Secondly the ranking from first to last stock made the relative difference between two given stocks dependent of (the spread of ) the other stocks in the sample. Lastly company size was not considered. This seemed somewhat arbitrary to me. To counter this I made a “Magic score” based on the scoring of the two financial metrics in normal distributed samples and a little extra.

Exact numbers for the Magic score calculation (such as mean, variance and sample) are secret for now though ;) Life needs some mystery to keep it interesting, don’t you think?
        """)
    
    run_magic_fomula(
        symbol_benchmark=symbol_benchmark,
        symbolsDate_dict=symbolsDate_dict,
        default_use_saved_benchmark=True,
        use_benchmark=False,
        default_metrics=[],
        magic_func=magic_formula
    )
