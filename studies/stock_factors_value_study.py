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
    value_score = (metrics['priceToEarning'] + metrics['priceToBook'] ) / 2
    return value_score

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
