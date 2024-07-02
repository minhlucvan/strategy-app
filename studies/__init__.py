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
from .stock_split_study import run as run_stock_split
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
from .stock_overnight_study import run as run_stock_overnight_study
from .stock_gaps_recover_study import run as run_stock_gaps_recover_study
from .stock_intraday_study import run as run_stock_intraday_study
from .vn30f_gaps_recover_study import run as run_vn30f_gaps_recover_study
from .vn30f_foregin_flow import run as run_vn30f_foregin_flow
from .vn30_hedging_flow import run as run_vn30_hedging_flow
from .vn_30_cash_flow_study import run as run_vn_30_cash_flow_study
from .vn30_volality_study import run as run_vn30_volality_study
from .warrants_gap_recover_study import run as run_warrants_gap_recover_study
from .rs_momentum_study import run as run_rs_momentum_study
from .market_value_study import run as run_market_value_study
from .stock_blind_order import run as run_stock_blind_order
from .warrants_blind_order_study import run as run_warrants_blind_order_study
from .warrants_simmulation_study import run as run_warrants_simmulation_study
from .warrants_simmulation_2_study import run as run_warrants_simmulation_2_study
from .warrants_volatility_study import run as run_warrants_volatility_study
from .vn30f_gaps_dips_study import run as run_vn30f_gaps_dips_study
from .stock_cacnel_order import run as run_stock_cacnel_order
from .stock_low_flow_study import run as run_stock_low_flow_study
from .tcbs_stock_noti_study import run as run_tcbs_stock_noti_study
from .stock_smc_study import run as run_stock_smc_study
from .stock_123_pullback_study import run as run_stock_123_pullback_study
from .stock_darvas_box_study import run as run_stock_darvas_box_study

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
        "name": "Stock Split",
        "module": run_stock_split
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
        "name": "Stock Overnight",
        "module": run_stock_overnight_study  
    },
    {
        "name": "Stock Gaps Recover",
        "module": run_stock_gaps_recover_study
    },
    {
        "name": "VN30F Gaps Recover",
        "module": run_vn30f_gaps_recover_study
    },
    {
        "name": "VN30F Gaps Dips",
        "module": run_vn30f_gaps_dips_study
    },
    {
        "name": "VN30F Foregin Flow",
        "module": run_vn30f_foregin_flow
    },
    {
        "name": "VN30 Hedging Flow",
        "module": run_vn30_hedging_flow
    },
    {
        "name": "VN30 Cash Flow",
        "module": run_vn_30_cash_flow_study
    },
    {
        "name": "VN30 Volality",
        "module": run_vn30_volality_study
    },
    {
        "name": "Stock Intraday",
        "module": run_stock_intraday_study  
    },
    {
        "name": "RS Momentum",
        "module": run_rs_momentum_study
    },
    {
        "name": "Market Value",
        "module": run_market_value_study
    },
    {
        "name": "Stock Blind Order",
        "module": run_stock_blind_order
    },
    {
        "name": "Stock Cancel Order",
        "module": run_stock_cacnel_order
    },
    {
        "name": "Warrants Blind Order",
        "module": run_warrants_blind_order_study
    },
    {
        "name": "Warrants Simmulation",
        "module": run_warrants_simmulation_study  
    },
    {
        "name": "Warrants Simmulation 2",
        "module": run_warrants_simmulation_2_study
    },
    {
        "name": "Warrants Volatility",
        "module": run_warrants_volatility_study
    },
    {
        "name": "Stock Low Flow",
        "module": run_stock_low_flow_study
    },
    {
        "name": "Stock 123 Pullback",
        "module": run_stock_123_pullback_study
    },
    {
        "name": "Stock Darvas Box",
        "module": run_stock_darvas_box_study
    },
    {
        "name": "TCBS Agent",
        "module": run_tcbs_agent
    },
    {
        "name": "Stock SMC",
        "module": run_stock_smc_study
    },
    {
        "name": "TCBS Market Calendar",
        "module": run_tcbs_market_calendar
    },
    {
        "name": "TCBS Stock Noti",
        "module": run_tcbs_stock_noti_study
    },
    {
        "name": "Warrants Gap Recover",
        "module": run_warrants_gap_recover_study  
    }
]
