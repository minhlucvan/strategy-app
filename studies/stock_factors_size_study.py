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

def magic_formula(metrics):

    return 0

def run(symbol_benchmark, symbolsDate_dict):
    st.write("## Size Factor Study")
    
    run_magic_fomula(
        symbol_benchmark=symbol_benchmark,
        symbolsDate_dict=symbolsDate_dict,
        default_use_saved_benchmark=True,
        use_benchmark=False,
        default_metrics=[],
        magic_func=magic_formula
    )
