import pandas as pd
import numpy as np
import json
import datetime

import streamlit as st
import vectorbt as vbt
import akshare as ak

import plotly.express as px
import humanize
import matplotlib.cm

from utils.component import check_password, input_dates
from utils.dataroma import *

from utils.riskfolio import get_pfOpMS, FactorExposure, plot_AssetsClusters
from utils.portfolio import Portfolio
from utils.vbt import get_pfByWeight, get_pfByMaxReturn, plot_pf
from utils.processing import get_stocks
from studies.rrg import plot_RRG, RRG_Strategy
from utils.fundEngine import get_fundSources, get_fundEngine

# @st.cache_data()


def get_bobmaxsr(_symbolsDate_dict: dict, fund_desc: str = ""):
    '''
    get the best of best max sharpe ratio solution
    '''
    strategy_list = getattr(__import__(f"vbt_strategy"), 'strategy_list')
    max_dict = {
        "symbol": _symbolsDate_dict['symbols'][0],
        "sharpe ratio": 0,
        "pf":   None,
        "strategy name": '',
        "param_dict": {},
        "savetodb": False
    }
    for strategyname in strategy_list:
        strategy_cls = getattr(__import__(
            f"vbt_strategy"), strategyname + 'Strategy')
        strategy = strategy_cls(_symbolsDate_dict)
        if len(strategy.stock_dfs) > 0:
            strategy.param_dict['RARM'] = 'sharpe_ratio'
            strategy.param_dict['WFO'] = 'None'
            if strategy.maxRARM(strategy.param_dict, output_bool=False):
                sharpe_ratio = round(strategy.pf.stats('sharpe_ratio')[0], 2)
                if sharpe_ratio > max_dict['sharpe ratio']:
                    max_dict['sharpe ratio'] = sharpe_ratio
                    max_dict['pf'] = strategy.pf
                    max_dict['strategy name'] = strategyname
                    max_dict['param_dict'] = strategy.param_dict.copy()
    return max_dict


def cal_beststrategy(symbolsDate_dict, fund_desc):
    bobs = []
    info_holder = st.empty()
    expander_holder = st.expander(
        "Best Strategy of Top 10 Stocks", expanded=True)
    sd_dict = symbolsDate_dict.copy()
    portfolio = Portfolio()
    with expander_holder:
        col1, col2, col3, col4, col5 = st.columns((1, 1, 1, 6, 1))
        col1.text('Symbol')
        col2.text('Sharpe Ratio')
        col3.text('Strategy')
        col4.text('Parameters')
    for symbol in symbolsDate_dict['symbols']:
        info_holder.write(f"Calculate symbol('{symbol}')")
        sd_dict['symbols'] = [symbol]
        bob_dict = get_bobmaxsr(sd_dict, fund_desc)
        bobs.append(bob_dict)
        with expander_holder:
            col1, col2, col3, col4, col5 = st.columns((1, 1, 1, 6, 1))
            col1.text(bob_dict['symbol'])
            col2.text(bob_dict['sharpe ratio'])
            col3.text(bob_dict['strategy name'])
            print(bob_dict['param_dict'])
            col4.text(json.dumps(bob_dict['param_dict']))
            button_type = "Save"
            button_phold = col5.empty()  # create a placeholder
            do_action = button_phold.button(
                button_type, key='btn_save_'+symbol)
            if do_action:
                if portfolio.add(sd_dict, bob_dict['strategy name'], bob_dict['param_dict'], bob_dict['pf'], fund_desc):
                    button_phold.write("Saved")
                else:
                    button_phold.write('Fail')

    info_holder.empty()
    # expander_holder.expander = False
    return bobs


def show_FactorExposure(symbolsDate_dict, pf, stocks_df):
    factors_dict = {
        "None": "Select the Factors Exposure",
        "iShares 5 factors": {
            'MTUM': "Momentum",
            'QUAL': "Quality",
            'SIZE': "Size",
            'USMV': "Low Volatility",
            'VLUE': "Value"
        },
        "All Sector factors": {
            "XLB": "Materials",
            "XLC": "Communication",
            "XLE": "Energy",
            "XLF": "Financials",
            "XLI": "Industrials",
            "XLK": "Technology",
            "XLP": "Consumer Staples",
            "XLRE": "Real Estate",
            "XLU": "Utilities",
            "XLV": "Healthcare",
            "XLY": "Consumer Discretionary"
        }
    }
    factors_sel = st.selectbox(
        "**Factor Exposures**", factors_dict.keys(), label_visibility='collapsed')
    factors_sel = factors_dict[factors_sel]
    if isinstance(factors_sel, dict):
        st.write('、'.join(k+'('+v+')' for v, k in factors_sel.items()))
        sd_dict = symbolsDate_dict.copy()
        sd_dict['symbols'] = factors_sel.keys()
        factors_df = get_stocks(sd_dict, 'close')
        portfolio_df = pf.asset_value().to_frame("Portfolio")
        main_df = pd.concat([portfolio_df, stocks_df], axis=1)
        st.table(FactorExposure(main_df, factors_df).style.format(
            "{:.4f}").bar(color=['#ef553b', '#00cc96'], align='mid', axis=1))
        st.line_chart(pd.concat([portfolio_df, factors_df], axis=1))


def run():
    fund_sources = get_fundSources()
    fund_source = st.sidebar.selectbox("Select Funds' Source", fund_sources)
    fund_engine = get_fundEngine(fund_source)
    fund_name = st.sidebar.selectbox(
        f"Select fund from {fund_source}", fund_engine.funds_name)
    fund_engine.readStocks(fund_name)

    # 1. display selected fund's information
    st.subheader(fund_engine.fund_name + " - " + fund_engine.fund_ticker)
    df = fund_engine.fund_df

    market = fund_engine.market

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.write('**Period:**')
        st.write('**Portfolio_date:**')
    with col2:
        st.write(fund_engine.fund_period)
        st.write(fund_engine.fund_update_date)
    with col3:
        st.write('**Num_of_positions:**')
        st.write('**Top 10 holdings(%):**')
    with col4:
        st.write(df["Stock"].count())
        st.write("{}%".format(round(df["Portfolio (%)"].iloc[0:10].sum(), 2)))

    with st.expander("Portfolio Table"):
        st.dataframe(df[['Stock', 'Portfolio (%)']]
                     .style.format({'Portfolio (%)': '{:.2f}%'}),
                     use_container_width=True
                     )

    # 2.select optimized portfolio strategies.
    start_date, end_date = input_dates()
    symbolsDate_dict = {
        'market':       market,
        'symbols':      df.iloc[0:10]['Ticker'].tolist(),
        'weights':      df.iloc[0:10]['Portfolio (%)'].tolist(),
        'start_date':   start_date,
        'end_date':     end_date,
    }
    fund_engine.setSymbolsDate(symbolsDate_dict)
    if market == 'US':
        symbol_benchmark = 'SPY'
    elif market == 'VN':
        symbol_benchmark = 'VN30'
    else:
        symbol_benchmark = 'sh000001'

    subpage = st.sidebar.radio(
        "Select Optimized Methods:",
        ('Original Weights',
         'Fund Allocation',
         'Max Sharpe Weights',
         'Relative Rotation Graph Strategy (RRG)'), 
        horizontal=True)
    
    st.subheader(subpage)
    if subpage == 'Original Weights':
        # 2.1.1 plot Pie chart of Orginial fund porforlio.
        fig = px.pie(df.iloc[0:10], values='Portfolio (%)',
                     names='Ticker', title='Top 10 holdings')
        st.plotly_chart(fig)

        # 2.1.2 plot pf chart of Orginial fund porforlio.
        stocks_df = fund_engine.getSotcks()
        
        weights = []
        for symbol in stocks_df.columns:
            if symbol in df.index:
                weights.append(df.loc[symbol, 'Portfolio (%)'])
            else:
                weights.append(0)
                
        weights = np.array(weights) / sum(weights)
        pf = get_pfByWeight(stocks_df, weights)
        
        st.write('----')
        st.write("**Porfolio's Performance**")
        plot_pf(pf, select=False,
                name=f"{fund_name}-Original Weights", show_recents=False)

        # 2.1.3 calculate the factors effect of Original fund portfolio.
        st.write('----')
        # st.write("**Factor Exposures**")

        # show_FactorExposure(symbolsDate_dict, pf, stocks_df)
        # 2.1.4 Assets Clusters of Original fund portfolio.
        st.write("**Assets Clusters**")
        with st.expander("The codependence or similarity matrix: pearson; Linkage method of hierarchical clustering: ward"):
            plot_AssetsClusters(stocks_df)

    elif subpage == 'Max Sharpe Weights':
        # 2.2.1 calculate the optimized max sharpe ratio's portfolio.
        rms_dict = {
            'MV': "Standard Deviation",
            'MAD': "Mean Absolute Deviation",
            'MSV': "Semi Standard Deviation",
            'FLPM': "First Lower Partial Moment (Omega Ratio)",
            'SLPM': "Second Lower Partial Moment (Sortino Ratio)",
            'CVaR': "Conditional Value at Risk",
            'EVaR': "Entropic Value at Risk",
            'WR':   "Worst Realization (Minimax)",
            'MDD': "Maximum Drawdown of uncompounded cumulative returns (Calmar Ratio)",
            'ADD': "Average Drawdown of uncompounded cumulative returns",
            'CDaR': "Conditional Drawdown at Risk of uncompounded cumulative returns",
            'EDaR': "Entropic Drawdown at Risk of uncompounded cumulative returns",
            'UCI': "Ulcer Index of uncompounded cumulative returns",
        }
        col1, col2 = st.columns(2)
        with col1:
            st.subheader(subpage)
        with col2:
            rm = st.selectbox('Select Risk Measures', rms_dict.keys(),
                              format_func=lambda x: x+' (' + rms_dict[x] + ')')
        stocks_df = get_stocks(symbolsDate_dict, 'close')
        
        stocks_df = stocks_df.dropna()
        
        pf = get_pfOpMS(stocks_df, rm)
        plot_pf(pf, select=False,
                name=f"{fund_name}-Max Sharpe Weights", show_recents=False)
        # 2.2.2 calculate the factors effect of optimized max sharpe ratio's portfolio.
        show_FactorExposure(symbolsDate_dict, pf, stocks_df)
    elif subpage == 'Fund Allocation':
        # update the fund's portfolio
        st.write('----')
        edit_weights_df = df.copy()
        
        total_capital = st.number_input('Total Capital (VND)', value=1000000)
        
        st.write("Capital Allocation")
        edit_weights_df = st.data_editor(
            edit_weights_df,
            key='edit_weights_df',
            disabled=['Stock', 'Ticker'],
            use_container_width=True)
        
        # plot pie chart of the fund's portfolio
        fig = px.pie(edit_weights_df, values='Portfolio (%)',
                     names='Ticker', title='Top 10 holdings')
        st.plotly_chart(fig)
        
        if st.button("Update Portfolio"):
            symbolsDate_dict['weights'] = edit_weights_df['Portfolio (%)'].tolist()
            fund_engine.update_capital(total_capital)
            fund_engine.update_fund_df(edit_weights_df)
            fund_engine.setSymbolsDate(symbolsDate_dict)
            fund_engine.save()
            st.info("Portfolio Updated")
        if st.button('Reset Portfolio'):
            fund_engine.reset()
            st.info("Portfolio Reset")
    else:
        st.text("Calculate the optimal return solution based on the values of rs_ratio and rs_momentum in the relative rotation graph")
        col1, col2 = st.columns(2)
        with col1:
            RARM_obj = st.selectbox('Risk Adjusted Return Method',
                                    ['sharpe_ratio', 'annualized_return', 'deflated_sharpe_ratio', 'calmar_ratio', 'sortino_ratio',
                                     'omega_ratio', 'information_ratio', 'tail_ratio'])
        with col2:
            showRRG_bool = st.checkbox("Show Relative Rotation Graphs")
        symbolsDate_dict['symbols'] += [symbol_benchmark]
        stocks_df = get_stocks(symbolsDate_dict, 'close')
        pf = RRG_Strategy(symbol_benchmark, stocks_df, RARM_obj, showRRG_bool)

        st.write("*Porfolio's Performance**")
        plot_pf(pf, name=fund_name+'-RRG', bm_symbol=symbol_benchmark,
                bm_price=stocks_df[symbol_benchmark], select=False, show_recents=False)


if check_password():
    run()
