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

def fill_na(val, default):
    if val is None or pd.isnull(val):
        return default
    return val

def magic_formula(metrics):
    # Define the weights for each category
    weights = {
        'Valuation': 0.28,
        'Operating Efficiency': 0.24,
        'Financial Stability': 0.24,
        'Profitability': 0.24
    }

    costToIncome = fill_na(metrics['costToIncome'], 0)
    nonInterestOnToi = fill_na(metrics['nonInterestOnToi'], 0)
    loanOnEarnAsset = fill_na(metrics['loanOnEarnAsset'], 0)
    equityOnLoan = fill_na(metrics['equityOnLoan'], 0)
    creditGrowth = fill_na(metrics['creditGrowth'], 0)
    equityOnTotalAsset = fill_na(metrics['equityOnTotalAsset'], 0)
    equityOnLiability = fill_na(metrics['equityOnLiability'], 0)
    liquidityOnLiability = fill_na(metrics['liquidityOnLiability'], 0)
    interestMargin = fill_na(metrics['interestMargin'], 0)
    earningPerShare = fill_na(metrics['earningPerShare'], 0)
    roe = fill_na(metrics['roe'], 0)
    priceToEarning = fill_na(metrics['priceToEarning'], 0)
    priceToBook = fill_na(metrics['priceToBook'], 0)
    

    # Calculate the score for each category
    valuation_score = (1 / priceToEarning) + (1 / priceToBook) if priceToEarning > 0 and priceToBook > 0 else 0
    operating_efficiency_score = roe - costToIncome
    financial_stability_score = equityOnTotalAsset + equityOnLiability + liquidityOnLiability
    profitability_score = earningPerShare + interestMargin + nonInterestOnToi
    banking_specific_score = creditGrowth + loanOnEarnAsset + equityOnLoan

    # Combine the scores using the weights
    total_score = (weights['Valuation'] * valuation_score +
                   weights['Operating Efficiency'] * operating_efficiency_score +
                   weights['Financial Stability'] * financial_stability_score +
                   weights['Profitability'] * profitability_score +
                   banking_specific_score) / (sum(weights.values()) + 1)

    return total_score

def run(symbol_benchmark, symbolsDate_dict):
    st.write("## Relative comparison of bank stocks")
    
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

    use_magic_formula = st.checkbox("Use Magic Formula", value=True)
    
    if use_magic_formula:
        st.markdown("## Magic Formula")
        st.markdown("""
This formula provides investors with a comprehensive assessment of a stock's investment potential by integrating multiple key metrics across four critical dimensions: Valuation, Operating Efficiency, Financial Stability, Profitability, and Banking Specifics.

**Valuation (Lower is Better):**
- Considers the stock's Price-to-Earnings (P/E) and Price-to-Book (P/B) ratios, aiming to identify undervalued stocks relative to their earnings and book value.

**Operating Efficiency (Higher is Better):**
- Evaluates the efficient use of assets and operations management through metrics like Return on Assets (ROA), Return on Equity (ROE), and Cost-to-Income ratio.

**Financial Stability (Higher is Better):**
- Assesses the bank's financial health and risk management by analyzing metrics such as Equity to Total Assets, Equity on Liability, and Liquidity on Liability.

**Profitability (Higher is Better):**
- Measures the bank's ability to generate profit through metrics like Earnings Per Share (EPS), Interest Margin, and Non-Interest Income to Operating Income ratio.

**Banking Specifics (Higher is Better):**
- Focuses on banking-specific metrics such as Credit Growth, Loan on Earning Assets, and Equity on Loan, providing insights into the bank's performance within its industry.

By combining these dimensions into a single score, investors gain a holistic view of the stock's attractiveness for investment, where a higher score indicates a more favorable investment opportunity.

--- 

This description highlights the comprehensive nature of the formula and its ability to capture various aspects of a stock's performance, aiding investors in making informed investment decisions.
""")

    run_magic_fomula(
        symbol_benchmark=symbol_benchmark,
        symbolsDate_dict=symbolsDate_dict,
        default_metrics=default_metrics,
        default_use_saved_benchmark=True,
        use_benchmark=False,
        magic_formula_method='relatice',
        magic_func=magic_formula if use_magic_formula else None
    )
