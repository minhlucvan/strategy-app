import pandas as pd
import numpy as np
import json
import datetime

import streamlit as st
import vectorbt as vbt
import akshare as ak


from utils.fe_base_fund_engine import fundEngine
from utils.dataroma import *
from utils import vndirect
from utils.fe_local_fund import fe_local_fund
from utils.fe_vndirect import fe_vndirect
from utils.fe_broker_tcbs import fe_broker_tcbs

def get_fundSources():
    return ['Local', 'TCBS', 'Vndirect']

def get_fundEngine(fund_source:str):
    if (fund_source == 'Vndirect'):
        return fe_vndirect()
    if (fund_source == 'TCBS'):
        return fe_broker_tcbs()
    else:
        return fe_local_fund()
