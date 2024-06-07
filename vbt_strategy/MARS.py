import numpy as np
import pandas as pd

import streamlit as st
import vectorbt as vbt

from .MA import MAStrategy

class MARSStrategy(MAStrategy):
    '''MARS strategy'''
    _name = "MARS"
    desc = """This is a trend-following strategy that requires wrapping the MA strategy with RSC as indicator source instead of close price."""
    use_rsc = True
    bm_symbol = 'VN30'
    include_bm = True

    def get_ma_src(self):
        return self.rs_dfs[0][1]