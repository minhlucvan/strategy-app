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
from studies.rrg import plot_RRG, rs_ratio, RRG_Strategy
from utils.vbt import plot_pf
from vbt_strategy.MOM_D import get_MomDInd
from studies.magic_fomula_study import run as run_magic_fomula

def run(symbol_benchmark, symbolsDate_dict):
    st.write("## Banking Metrics Analysis")
    
    with st.expander("Methodology and Approach"):
        st.markdown("""
        ### Methodology and Approach
        This analysis aims to identify good bank stocks by evaluating them based on several key metrics. We categorize the metrics into four main groups:
        
        1. **Valuation**: Helps determine if a stock is fairly priced.
        2. **Operating Efficiency**: Gauges how well a bank is utilizing its assets and managing its operations.
        3. **Financial Stability**: Assesses the bank's financial health and risk.
        4. **Profitability**: Evaluates the bank's ability to generate profit.
        """)

    method = st.radio("Select analysis method", ['Valuation (Lower is Better)', 'Operating Efficiency (Higher is Better)', 'Financial Stability (Higher is Better)', 'Profitability (Higher is Better)', 'Banking Specific (Higher is Better)'])
    
    default_metrics = []
    if method == 'Valuation (Lower is Better)':
        st.markdown("## Valuation")
        st.markdown("""
        Valuation metrics help determine if a stock is fairly priced.
        - **Price to Earnings (P/E) Ratio**: Lower is better, suggesting the stock is undervalued relative to its earnings.
        - **Price to Book (P/B) Ratio**: Lower is better, indicating the stock is undervalued relative to its book value.
        """)
        default_metrics = ['priceToEarning', 'priceToBook']
    elif method == 'Operating Efficiency (Higher is Better)':
        st.markdown("## Operating Efficiency")
        st.markdown("""
        Operating efficiency metrics gauge how well a bank is utilizing its assets and managing its operations.
        - **Return on Assets (ROA)**: Higher is better, indicating efficient use of assets.
        - **Return on Equity (ROE)**: Higher is better, showing effective use of equity to generate profits.
        - **Cost to Income**: Lower is better, indicating efficient cost management relative to income.
        """)
        default_metrics = ['roa', 'roe', 'costToIncome']
    elif method == 'Financial Stability (Higher is Better)':
        st.markdown("## Financial Stability")
        st.markdown("""
        Financial stability metrics assess the bank's financial health and risk.
        - **Equity to Total Assets**: Higher is better, indicating a strong equity base.
        - **Equity on Liability**: Higher is better, indicating a greater proportion of equity to total liabilities.
        - **Liquidity on Liability**: Higher is better, showing the ability to meet short-term obligations.
        """)
        default_metrics = ['equityOnTotalAsset', 'equityOnLiability', 'liquidityOnLiability']
    elif method == 'Profitability (Higher is Better)':
        st.markdown("## Profitability")
        st.markdown("""
        Profitability metrics evaluate the bank's ability to generate profit.
        - **Earnings Per Share (EPS)**: Higher is better, indicating greater profitability on a per-share basis.
        - **Interest Margin**: Higher is better, indicating efficient interest income generation.
        - **Non-Interest Income to Operating Income**: Higher is better, indicating diversified income sources.
        """)
        default_metrics = ['earningPerShare', 'interestMargin', 'nonInterestOnToi']
    elif method == 'Banking Specific (Higher is Better)':
        st.markdown("## Banking Specific Metrics")
        st.markdown("""
        These metrics are specific to the banking sector and provide deeper insights into the performance and health of bank stocks.
        - **Credit Growth**: Higher is generally better, indicating expansion in the bank's loan portfolio.
        - **Loan on Earning Assets**: Higher is better, indicating effective utilization of earning assets for lending.
        - **Equity on Loan**: Higher is better, indicating a strong equity base relative to loans.
        """)
        default_metrics = ['creditGrowth', 'loanOnEarnAsset', 'equityOnLoan']

    run_magic_fomula(
        symbol_benchmark=symbol_benchmark,
        symbolsDate_dict=symbolsDate_dict,
        default_metrics=default_metrics,
        default_use_saved_benchmark=True,
        use_benchmark=False
    )
