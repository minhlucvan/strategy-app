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

import numpy as np
import pandas as pd


def calculate_value_score(metrics):
    return 0

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
    value_score = calculate_value_score(metrics)
    return 0

def run(symbol_benchmark, symbolsDate_dict):
    st.write("## Value Factor Study")
    
    run_magic_fomula(
        symbol_benchmark=symbol_benchmark,
        symbolsDate_dict=symbolsDate_dict,
        default_use_saved_benchmark=True,
        use_benchmark=False,
        default_metrics=[],
        magic_func=magic_formula
    )
