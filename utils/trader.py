
from utils.dataroma import *
from utils.trader_tcbs import TraderTCBS
from utils.trader_local import TraderLocal

def get_trader_list():
    return ['Local', 'TCBS']

def get_trader(fund_source:str):
    if (fund_source == 'TCBS'):
        return TraderTCBS()

    if (fund_source == 'Local'):
        return TraderLocal()