from .MOM import MOMStrategy
from .PairTrade import PairTradeStrategy
from .MA import MAStrategy
from .RSI import RSIStrategy
from .MACD import MACDStrategy
from .MOM_RSI import MOM_RSIStrategy
from .SuperTrend import SuperTrendStrategy
from .CSPR import CSPRStrategy, CSPR5Strategy
from .RSI3 import RSI3Strategy
from .PETOR import PETORStrategy
from .PEGTOR import PEGTORStrategy
from .ADX_RSI import ADX_RSIStrategy
from .EMACloud import EMACloudStrategy
from .MOM_D import MOM_DStrategy
# from .HHT import HHTStrategy
from .DivArb import DivArbStrategy
from .IssArb import IssArbStrategy
from .MARS import MARSStrategy
from .RRG import RRGStrategy
from .MMA import MMAStrategy
from .MMARS import MMARSStrategy
from .MOMTOP import MOMTOPStrategy
from .LiqFlow import LiqFlowStrategy
from .FinReportArb import FinReportArbStrategy
from .GapsRecover import GapsRecoverStrategy

__all__ = [
    "MOMStrategy",
    "MOM_DStrategy"
    "PairTradeStrategy",
    "MAStrategy",
    "RSIStrategy",
    "MACDStrategy",
    "MOM_RSIStrategy",
    "SuperTrendStrategy",
    "CSPRStrategy",
    "CSPR5Strategy",
    "RSI3Strategy",
    "ADX_RSIStrategy",
    "PETORStrategy",
    "PEGTORStrategy",
    "EMACloudStrategy",
    "DivArbStrategy",
    "IssArbStrategy",
    "MARSStrategy",
    "RRGStrategy",
    "MMAStrategy",
    "MMARSStrategy",
    "MOMTOPStrategy",
    "LiqFlowStrategy",
    "FinReportArbStrategy",
    "GapsRecoverStrategy"
]

strategy_list = [
    "MA",
    "MACD",
    "MOM",
    "MOM_D",
    "PairTrade",
    "RSI",
    "MOM_RSI",
    "SuperTrend",
    "CSPR",
    "CSPR5",
    "RSI3",
    "ADX_RSI",
    "PETOR",
    "PEGTOR",
    "EMACloud",
    "DivArb",
    "IssArb",
    "MARS",
    "RRG",
    "MMA",
    "MMARS",
    "MOMTOP",
    "LiqFlow",
    "FinReportArb",
    "GapsRecover"
]