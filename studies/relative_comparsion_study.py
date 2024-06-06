import pandas as pd

import streamlit as st



from studies.magic_fomula_study import run as run_magic_fomula

def fill_na(val, default):
    if val is None or pd.isnull(val):
        return default
    return val


def fill_na(val, default):
    if val is None or pd.isnull(val):
        return default
    return val

def calculate_mean(metrics, factors):
    total = 0
    count = 0
    for factor in factors:
        if factor in metrics and not pd.isnull(metrics[factor]):
            total += metrics[factor]
            count += 1
    return total / count if count > 0 else 0

# valuation metrics (lower is better)
# = priceToEarning + priceToBook + valueBeforeEbitda + dividend
def calculate_valuation(metrics):
    score = calculate_mean(metrics, ['priceToEarning', 'priceToBook', 'valueBeforeEbitda', 'dividend'])
    score = 1 / score if score > 0 else 0
    return score

# operating efficiency metrics (higher is better)
# = roa + roe + daysPayable + daysReceivable
def calculate_operating_efficiency(metrics):
    score = calculate_mean(metrics, ['roa', 'roe', 'daysPayable', 'daysReceivable'])
    return score 

# financial stability metrics (higher is better)
# = equityOnTotalAsset + currentPayment + quickPayment
def calculate_financial_stability(metrics):
    score = calculate_mean(metrics, ['equityOnTotalAsset', 'currentPayment', 'quickPayment'])
    return score

# profitability metrics (higher is better)
# = grossProfitMargin + operatingProfitMargin + postTaxMargin
def calculate_profitability(metrics):
    score = calculate_mean(metrics, ['grossProfitMargin', 'operatingProfitMargin', 'postTaxMargin'])
    return score

# one metric uniformly applied to all stocks
# combine other metrics to form a score
# the score is used to rank the stocks
# the score is linear combination of the metrics
# parameter is a np.series of metrics
# index: ticker,quarter,year,priceToEarning,priceToBook,valueBeforeEbitda,dividend,roe,roa,
# daysReceivable,daysInventory,daysPayable,ebitOnInterest,earningPerShare,bookValuePerShare,
# interestMargin,nonInterestOnToi,badDebtPercentage,provisionOnBadDebt,costOfFinancing,equityOnTotalAsset,
# equityOnLoan,costToIncome,equityOnLiability,currentPayment,quickPayment,epsChange,ebitdaOnStock,
# grossProfitMargin,operatingProfitMargin,postTaxMargin,debtOnEquity,debtOnAsset,debtOnEbitda,
# shortOnLongDebt,assetOnEquity,capitalBalance,cashOnEquity,cashOnCapitalize,cashCirculation,
# revenueOnWorkCapital,capexOnFixedAsset,revenueOnAsset,postTaxOnPreTax,ebitOnRevenue,preTaxOnEbit,
# preProvisionOnToi,postTaxOnToi,loanOnEarnAsset,loanOnAsset,loanOnDeposit,depositOnEarnAsset,
# badDebtOnAsset,liquidityOnLiability,payableOnEquity,cancelDebt,ebitdaOnStockChange,
# bookValuePerShareChange,creditGrowth
# return: magic score
def magic_formula(metrics):
    valuation_score = calculate_valuation(metrics)
    operating_efficiency_score = calculate_operating_efficiency(metrics)
    financial_stability_score = calculate_financial_stability(metrics)
    profitability_score = calculate_profitability(metrics)
    sum_score = valuation_score + operating_efficiency_score + financial_stability_score + profitability_score
    total_score = sum_score / 4
    return total_score

def run(symbol_benchmark, symbolsDate_dict):
    st.write("## Relative comparison of stocks")

    with st.expander("Methodology and Approach"):
        st.markdown("""
        ### Methodology and Approach
        This analysis aims to identify good stocks by evaluating them based on several key metrics. We categorize the metrics into four main groups:
        
        1. **Valuation**: Helps determine if a stock is fairly priced.
        2. **Operating Efficiency**: Gauges how well a company is utilizing its assets and managing its operations.
        3. **Financial Stability**: Assesses the company's financial health and risk.
        4. **Profitability**: Evaluates the company's ability to generate profit.
        """)

    method = st.radio("Select analysis method", ['Valuation (Lower is Better)', 'Operating Efficiency (Higher is Better)', 'Financial Stability (Higher is Better)', 'Profitability (Higher is Better)'])
    
    default_metrics = []
    if method == 'Valuation (Lower is Better)':
        st.markdown("## Valuation")
        st.markdown("""
        Valuation metrics help determine if a stock is fairly priced.
        - **Price to Earnings (P/E) Ratio**: Lower is better, suggesting the stock is undervalued relative to its earnings.
        - **Price to Book (P/B) Ratio**: Lower is better, indicating the stock is undervalued relative to its book value.
        - **EBITDA (Earnings Before Interest, Taxes, Depreciation, and Amortization)**: Higher is better, indicating stronger operational earnings.
        - **Dividend Yield**: Higher is better, showing potential returns to investors.
        """)
        default_metrics = ['priceToEarning', 'priceToBook', 'valueBeforeEbitda', 'dividend']
    elif method == 'Operating Efficiency (Higher is Better)':
        st.markdown("## Operating Efficiency")
        st.markdown("""
        Operating efficiency metrics gauge how well a company is utilizing its assets and managing its operations.
        - **Return on Assets (ROA)**: Higher is better, indicating efficient use of assets.
        - **Return on Equity (ROE)**: Higher is better, showing effective use of equity to generate profits.
        - **Days Payable Outstanding**: Lower is better, indicating efficient management of accounts payable.
        - **Days Receivable Outstanding**: Lower is better, indicating efficient management of accounts receivable.
        """)
        default_metrics = ['roa', 'roe', 'daysPayable', 'daysReceivable']
    elif method == 'Financial Stability (Higher is Better)':
        st.markdown("## Financial Stability")
        st.markdown("""
        Financial stability metrics assess the company's financial health and risk.
        - **Equity on Total Assets**: Higher is better, indicating a stronger financial position.
        - **Current Ratio**: Higher is better, indicating better short-term liquidity.
        - **Quick Ratio**: Higher is better, indicating even better short-term liquidity.
        """)
        default_metrics = ['equityOnTotalAsset', 'currentPayment', 'quickPayment']
    elif method == 'Profitability (Higher is Better)':
        st.markdown("## Profitability")
        st.markdown("""
        Profitability metrics evaluate the company's ability to generate profit.
        - **Net Profit Margin**: Higher is better, indicating better profitability relative to revenue.
        - **Gross Profit Margin**: Higher is better, indicating better efficiency in production.
        - **Operating Profit Margin**: Higher is better, indicating better operational efficiency.
        """)
        default_metrics = ['grossProfitMargin', 'operatingProfitMargin']

    use_magic_formula = st.checkbox("Use Magic Formula", value=True)
    
    if use_magic_formula:
        st.markdown("## Magic Formula")
        st.markdown("""
        This formula provides investors with a comprehensive assessment of a stock's investment potential by integrating multiple key metrics across four critical dimensions: Valuation, Operating Efficiency, Financial Stability, and Profitability.

        **Valuation (Lower is Better):**
        - Considers the stock's Price-to-Earnings (P/E) and Price-to-Book (P/B) ratios, aiming to identify undervalued stocks relative to their earnings and book value.
        - Also includes EBITDA and Dividend Yield.

        **Operating Efficiency (Higher is Better):**
        - Evaluates the efficient use of assets and operations management through metrics like Return on Assets (ROA) and Return on Equity (ROE).
        - Includes Days Payable Outstanding and Days Receivable Outstanding.

        **Financial Stability (Higher is Better):**
        - Assesses the company's financial health and risk management by analyzing metrics such as Equity on Total Assets, Current Ratio, and Quick Ratio.

        **Profitability (Higher is Better):**
        - Measures the company's ability to generate profit through metrics like Net Profit Margin, Gross Profit Margin, and Operating Profit Margin.

        By combining these dimensions into a single score, investors gain a holistic view of the stock's attractiveness for investment, where a higher score indicates a more favorable investment opportunity.
        """)
        
    run_magic_fomula(
        symbol_benchmark=symbol_benchmark,
        symbolsDate_dict=symbolsDate_dict,
        default_metrics=default_metrics,
        default_use_saved_benchmark=True,
        use_benchmark=False,
        magic_formula_method='relative',
        magic_func=magic_formula if use_magic_formula else None
    )
