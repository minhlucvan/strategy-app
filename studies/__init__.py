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
from .stock_foregin_flow_study import run as run_stock_foregin_flow
from .stock_liqudity_flow_study import run as run_stock_liqudity_flow
from .stock_factors_models_study import run as run_stock_factors_models
from .stock_factors_value_study import run as run_stock_factors_value
from .stock_factors_size_study import run as run_stock_factors_size
from .stock_factors_quality_study import run as run_stock_factors_quality
from .stock_factors_momentum_study import run as run_stock_factors_momentum
from .stock_factors_volatility_study import run as run_stock_factors_volatility
from .stock_factor_base_study import run as run_stock_factor_base
from .stock_financial_report_study import run as run_stock_financial_report
from .stock_earning_suprise_study import run as run_stock_earning_suprise_study
from .stock_financial_momentum_study import run as run_stock_financial_momentum_study
from .cash_flow_study import run as run_cash_flow_study
from .stock_news_momentum_study import run as run_stock_news_momentum_study
from .stock_headline_momentum_study import run as run_stock_headline_momentum_study
from .relative_stregth_compasion_study import run as run_relative_stregth_compasion

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
        "name": "Stock Foregin Flow",
        "module": run_stock_foregin_flow
    },
    {
        "name": "Stock Liqudity Flow",
        "module": run_stock_liqudity_flow
    },
    {
        "name": "Stock Factors Models",
        "module": run_stock_factors_models
    },
    { 
        "name": "Stock Factor Base",
        "module": run_stock_factor_base
    },
    {
        "name": "Stock Factors Value",
        "module": run_stock_factors_value
    },
    {
        "name": "Stock Factors Size",
        "module": run_stock_factors_size
    },
    {
        "name": "Stock Factors Quality",
        "module": run_stock_factors_quality
    },
    {
        "name": "Stock Factors Momentum",
        "module": run_stock_factors_momentum
    },
    {
        "name": "Stock Factors Volatility",
        "module": run_stock_factors_volatility
    },
    {
        "name": "Stock Financial Report",
        "module": run_stock_financial_report
    },
    {
        "name": "Stock Earning Suprise",
        "module": run_stock_earning_suprise_study
    },
    {
        "name": "Stock Financial Momentum",
        "module": run_stock_financial_momentum_study
    },
    {
        "name": "Cash Flow",
        "module": run_cash_flow_study
    },
    {
        "name": "Stock News Momentum",
        "module": run_stock_news_momentum_study
    },
    {
        "name": "Stock Headline Momentum",
        "module": run_stock_headline_momentum_study  
    },
    {
        "name": "Relative Strength Compasion",
        "module": run_relative_stregth_compasion
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
