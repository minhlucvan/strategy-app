from .rrg_study import run as run_rotation
from .market_wide_study import run as run_market_wide
from .value_investment_study import run as run_value_investment
from .market_pricing_study import run as run_market_pricing
from .fundamental_pricing_study import run as run_fundamental_pricing
from .momentum_top_study import run as run_momentum_top_down
from .etf_flow_study import run as run_etf_flow
from .etf_structure_study import run as run_etf_structure
from .magic_fomula_study import run as run_magic_fomula
from .stock_events_study import run as run_stock_events
from .stock_dividend_study import run as run_stock_dividend
from .bank_pricing_study import run as run_bank_pricing
from .steven_fomula_study import run as run_steven_fomula
from .relative_comparsion_study import run as run_relative_comparsion
from .stock_issue_study import run as run_stock_issue
from .tcbs_agent_study import run as run_tcbs_agent
from .tcbs_market_calendar_study import run as run_tcbs_market_calendar

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
    },
    {
        "name": "Stock Events",
        "module": run_stock_events
    },
    {
        "name": "Stock Dividend",
        "module": run_stock_dividend
    },
    {
        "name": "Stock Issue",
        "module": run_stock_issue
    },
    {
        "name": "Bank Pricing",
        "module": run_bank_pricing
    },
    {
        "name": "Steven Fomula",
        "module": run_steven_fomula
    },
    {
        "name": "Relative Comparsion",
        "module": run_relative_comparsion
    },
    {
        "name": "TCBS Agent",
        "module": run_tcbs_agent
    },
    {
        "name": "TCBS Market Calendar",
        "module": run_tcbs_market_calendar
    }
]
