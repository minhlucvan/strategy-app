from .rrg_study import run as run_rotation
from .market_wide_study import run as run_market_wide
from .value_investment_study import run as run_value_investment
from .market_pricing_study import run as run_market_pricing
from .fundamental_pricing_study import run as run_fundamental_pricing
from .momentum_top_study import run as run_momentum_top_down
from .etf_flow_study import run as run_etf_flow
from .etf_structure_study import run as run_etf_structure
from .magic_fomula_study import run as run_magic_fomula

study_list = [
    {
        "name": "Select Study",
        "module": None
    },
    {
        "name": "Relative Rotation Graph",
        "module": run_rotation
    }, 
    {
        "name": "Market Wide",
        "module": run_market_wide
    },
    {
        "name": "Value Investment",
        "module": run_value_investment
    },
    {
        "name": "Market Pricing",
        "module": run_market_pricing
    },
    {
        "name": "Fundamental Pricing",
        "module": run_fundamental_pricing
    },
    {
        "name": "Momentum top down",
        "module": run_momentum_top_down
    },
    {
        "name": "ETF Flow",
        "module": run_etf_flow
    },
    {
        "name": "ETF Structure",
        "module": run_etf_structure
    },
    {
        "name": "Magic Fomula",
        "module": run_magic_fomula
    }       
]
