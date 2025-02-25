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

def calculate_value_score(metrics):
    return 0

def calculate_size_score(metrics):
    return 0

def calculate_momentum_score(metrics):
    return 0

def calculate_quality_score(metrics):
    return 0

def calculate_volatility_score(metrics):
    return 0

def magic_formula(metrics):
    value = calculate_value_score(metrics)
    size = calculate_size_score(metrics)
    momentum = calculate_momentum_score(metrics)
    quality = calculate_quality_score(metrics)
    volatility = calculate_volatility_score(metrics)
    
    weights = {
        'value': [1],
        'size': [1],
        'momentum': [1],
        'quality': [1],
        'volatility': [1]
    }
    
    score_df = pd.DataFrame({
        'value': [value],
        'size': [size],
        'momentum': [momentum],
        'quality': [quality],
        'volatility': [volatility]
    })
    
    weights_df = pd.DataFrame(weights, index=[0])
    
    # weighted mean
    magic_score = (score_df * weights_df).sum(axis=1) / weights_df.sum(axis=1)
    return magic_score

def run(symbol_benchmark, symbolsDate_dict):
    st.write("## Steven's Magic Formula")
    
    with st.expander("Multi Factor Model"):
        st.markdown("""Multi Factor Model is a model that combines multiple factors to predict stock returns.
                    https://www.ishares.com/us/insights/what-is-factor-investing
                    
Value: Metrics like P/E, P/B, Dividend Yield.
Size: Market capitalization.
Momentum: Past returns over various periods.
Quality: Metrics like ROE, Debt-to-Equity ratio, Earnings Stability.
Volatility: Historical price volatility or Beta.""")
    
    run_magic_fomula(
        symbol_benchmark=symbol_benchmark,
        symbolsDate_dict=symbolsDate_dict,
        default_use_saved_benchmark=True,
        use_benchmark=False,
        default_metrics=[],
        magic_func=magic_formula
    )
