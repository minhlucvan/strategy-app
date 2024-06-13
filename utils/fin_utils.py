import streamlit as st
import pandas as pd

def score_financial_report(prev_report_df, curr_report_df):
    # Define the metrics and their corresponding weights
    metrics_weights = {
        "priceToEarning": -1,
        # "priceToBook": -1.2,
        # "valueBeforeEbitda": -1.5,
        # "roe": 1.2,
        # "roa": 1.0,
        # "daysReceivable": -0.8,
        # "daysInventory": -0.8,
        # "daysPayable": 0.8,
        # "ebitOnInterest": -1.5,
        # "earningPerShare": 1.2,
        # "bookValuePerShare": -1.0,
        # "equityOnTotalAsset": 1.0,
        # "equityOnLiability": 1.0,
        # "currentPayment": 0.7,
        # "quickPayment": 0.7,
        # "epsChange": 1.2,
        # "ebitdaOnStock": 1.0,
        # "grossProfitMargin": 1.0,
        # "operatingProfitMargin": 1.0,
        # "postTaxMargin": 1.0,
        # "debtOnEquity": -1.5,
        # "debtOnAsset": -1.5,
        # "debtOnEbitda": -1.5,
        # "shortOnLongDebt": -0.8,
        # "assetOnEquity": 0.8,
        # "capitalBalance": 0.8,
        # "cashOnEquity": 0.8,
        # "cashOnCapitalize": 0.8,
        # "cashCirculation": 0.8,
        # "revenueOnWorkCapital": 1.0,
        # "capexOnFixedAsset": 0.8,
        # "revenueOnAsset": 1.0,
        # "postTaxOnPreTax": 1.2,
        # "ebitOnRevenue": 1.0,
        # "preTaxOnEbit": 1.2,
        # "payableOnEquity": -0.8,
        # "ebitdaOnStockChange": 1.0,
        # "bookValuePerShareChange": 1.0,
    }

    def calculate_score(prev_row, curr_row):
        score = 0
        for metric, weight in metrics_weights.items():
            prev_value = prev_row.get(metric)
            curr_value = curr_row.get(metric)
            if prev_value is None or curr_value is None:
                continue
            diff = curr_value - prev_value
            diff = 0 if pd.isna(diff) else diff
            score += weight * diff
        return score

    # Apply the score calculation to each row pair in the DataFrames
    scores = []
    for i in range(len(prev_report_df)):
        prev_row = prev_report_df.iloc[i]
        curr_row = curr_report_df.iloc[i]
        score = calculate_score(prev_row, curr_row)
        scores.append(score)
    
    return scores
