import pandas as pd
import numpy as np
import json

import streamlit as st
import vectorbt as vbt

import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots # creating subplots


from utils.component import  check_password, input_dates, input_SymbolsDate
import matplotlib.pyplot as plt

from utils.processing import get_stocks, get_stocks_valuation
from studies.rrg import plot_RRG, rs_ratio, RRG_Strategy

def run(symbol_benchmark, symbolsDate_dict):
        
    symbolsDate_dict['symbols'] =  symbolsDate_dict['symbols']
    # copy the symbolsDate_dict
    benchmark_dict = symbolsDate_dict.copy()
    benchmark_dict['symbols'] = [symbol_benchmark]
    
    benchmark_df = get_stocks(benchmark_dict,'close')
    stocks_df = get_stocks(symbolsDate_dict,'close')
   
