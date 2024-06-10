import streamlit as st

from utils.component import input_SymbolsDate, check_password, form_SavePortfolio, params_selector
from utils.db import get_SymbolName
from utils.processing import get_stocks
from utils.vbt import display_pfbrief
from plotly import graph_objects as go
import pandas as pd

if check_password():
    symbolsDate_dict = input_SymbolsDate()
    simmulation_mode = st.sidebar.radio(
        "simmulation mode", ["Multi Strategies", "Single Strategy"])

    if len(symbolsDate_dict['symbols']) < 1:
        st.info("Please select symbols.")
        st.stop()

    if len(symbolsDate_dict['symbols']) > 0:
        st.header(
            f"{get_SymbolName(symbolsDate_dict['symbols'][0])} Strategies' comparision board")
        strategy_list = getattr(__import__(f"vbt_strategy"), 'strategy_list')

        defalut_strategies = []
        if simmulation_mode == "Single Strategy":
            selected_strategy = st.selectbox(
                "Please select a strategy", strategy_list)
            selected_strategies = [selected_strategy]
        else:
            selected_strategies = st.multiselect(
                "Please select strategies", strategy_list, defalut_strategies)

        stocks_df = get_stocks(symbolsDate_dict, 'close')

        fig = go.Figure()
        for symbol in symbolsDate_dict['symbols']:
            if symbol in stocks_df.columns:
                fig.add_trace(go.Scatter(x=stocks_df.index,
                              y=stocks_df[symbol], mode='lines', name=symbol))
        st.plotly_chart(fig, use_container_width=True)

        params = params_selector({})
        results = []

        for strategyname in selected_strategies:
            strategy_cls = getattr(__import__(
                f"vbt_strategy"), strategyname + 'Strategy')

            strategy = strategy_cls(symbolsDate_dict)
            if simmulation_mode == "Multi Strategies":
                if strategy.maxRARM(params, output_bool=False):
                    st.info(
                        f"Strategy '{strategyname}' Max {strategy.param_dict['RARM']} Result")
                    lastday_return, sharpe_ratio, maxdrawdown, annual_return =  display_pfbrief(strategy.pf, strategy.param_dict)
                    # form_SavePortfolio(
                    #     symbolsDate_dict, strategyname, strategy.param_dict, strategy.pf)
                    results.append({
                        'symbol': 'All',
                        'strategy': strategyname,
                        'lastday_return': lastday_return,
                        'sharpe_ratio': sharpe_ratio,
                        'maxdrawdown': maxdrawdown,
                        'annual_return': annual_return
                    })
                else:
                    st.error(f"Strategy '{strategyname}' failed.")
            else:
                symbols = symbolsDate_dict['symbols']

                for symbol in symbols:
                    st.write(f"Simulating {strategyname} for {symbol}")
                    symbolsDate_dict_copy = symbolsDate_dict.copy()
                    symbolsDate_dict_copy['symbols'] = [symbol]
                    key = f'Sim_{strategyname}_{symbol}'

                    strategy = strategy_cls(symbolsDate_dict_copy)
                    
                    if not strategy.validate():
                        st.error(f"Strategy '{strategyname}' - {symbol} failed.")
                        continue

                    if strategy.maxRARM(params, output_bool=False):
                        st.info(
                            f"Strategy '{strategyname}' Max {strategy.param_dict['RARM']} Result")
                        lastday_return, sharpe_ratio, maxdrawdown, annual_return = display_pfbrief(
                            strategy.pf, strategy.param_dict, key=key)
                        # form_SavePortfolio(
                        #     symbolsDate_dict_copy, strategyname, strategy.param_dict, strategy.pf, key=key)
                        
                        results.append({
                            'symbol': symbol,
                            'strategy': strategyname,
                            'lastday_return': lastday_return,
                            'sharpe_ratio': sharpe_ratio,
                            'maxdrawdown': maxdrawdown,
                            'annual_return': annual_return
                        
                        })
                    else:
                        st.error(f"Strategy '{strategyname}' failed.")

        # compare the results
        results_df = pd.DataFrame(results)
        st.dataframe(results_df, use_container_width=True)