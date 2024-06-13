import numpy as np
import pandas as pd
from datetime import timezone

import streamlit as st
import vectorbt as vbt
import datetime

from indicators.EventArb import generate_arbitrage_signal, get_EventArbInd
from indicators.SEventArb import get_SEventArbInd
from studies.stock_news_momentum_study import calculate_price_changes, get_events_signals_by_column
from utils.processing import get_stocks_document, get_stocks_events
from utils.tcbs_agent import load_calender_data_tp_df
from utils.tcbs_api import TCBSAPI
import utils.config as cfg
from .base import BaseStrategy
from utils.vbt import plot_CSCV

class FinReportArbStrategy(BaseStrategy):
    '''Finanacial Report Arbitrage strategy'''
    _name = "FinReport"
    desc = "Financial Report Arbitrage strategy aiming to capture the price difference between the financial report release date and the ex-dividend date. the study shown that for a bunch of stocks, the price of the stock will be raised before the financial report release date and drop after the ex-dividend date, the strategy aims to capture the price difference between the two dates"
    param_def = [
        {
            "name": 'days_before',
            "type": "int",
            "min":  0,
            "max":  1,
            "step": 0
        }, {
            "name": 'days_after',
            "type": "int",
            "min":  3,
            "max":  10,
            "step": 0
        }, {
            "name": 'signal_threshold',
            "type": "float",
            "min": -0.1,
            "max": 0.1,
            "step": 0.01,
        }
    ]
    pf_kwargs = dict(fees=0.0005, slippage=0.001, freq='1D')

    stacked_bool = True

    @vbt.cached_method
    def run(self, calledby='add'):
        stocks_df = self.stocks_df
        self.bm_symbol = 'VN30'
        symbols = self.symbolsDate_dict['symbols']
        
        days_before = self.param_dict['days_before']
        days_after = self.param_dict['days_after']
        signal_threshold = self.param_dict['signal_threshold']
        
        # st.write(signal_threshold.values)
        
        if isinstance(signal_threshold, np.ndarray):
            # to list
            signal_threshold = signal_threshold.tolist()
        
        bm_df= self.datas.get_stock(self.bm_symbol, self.start_date, self.end_date)
        self.bm_price = bm_df['close']
        
        # align the benchmark price with the stocks price
        self.bm_price = self.bm_price.reindex(stocks_df.index, method='nearest')
        
        # 1. initialize the variables

        close_price = stocks_df
        
        events_df = get_stocks_document(self.symbolsDate_dict, 'Title', doc_type='1', group_by_date=True)

        # price_changes_flat_df = calculate_price_changes(stocks_df, events_df, lower_bound=-20, upper_bound=20)

        # price_changes_df = get_events_signals_by_column(events_df, price_changes_flat_df, column='change_1')
        
        days_to_event_sig = generate_arbitrage_signal(stocks_df, events_df)
        price_changes_sig = stocks_df.pct_change().astype(float)
        
        # st.write(days_to_event_sig)
        # st.stop()
                        
        # 2. calculate the indicators
        indicator =  get_SEventArbInd().run(
            close_price,
            days_to_event_sig,
            price_changes_sig,
            days_before_threshold=days_before,
            days_after_threshold=days_after,
            signal_threshold=signal_threshold,
            param_product=True
        )    
                
        # # 3. remove all the name in param_def from param_dict
        for param in self.param_def:
            del self.param_dict[param['name']]

        # # 4. generate the vbt signal
        sizes = indicator.sizes.shift(periods=1)

        # fill na with 0
        sizes = sizes.fillna(0)

        st.write(events_df)
        st.write(sizes)
        # st.stop()
        
        num_symbol = len(stocks_df.columns)
        num_group = int(len(sizes.columns) / num_symbol)
        
        group_list = []
        for n in range(num_group):
            group_list.extend([n]*num_symbol)
            
        group_by = pd.Index(group_list, name='group')


        # 5. Build portfolios
        if calledby == 'add'  and self.param_dict['WFO'] != 'None':
            entries, exits = self.maxRARM_WFO(
                close_price, entries, exits, calledby)
            pf = vbt.Portfolio.from_orders(
                close_price,
                sizes,
                # init_cash=100,
                size_type='percent',
                group_by=group_by,
                cash_sharing=True,
                **self.pf_kwargs
            )
        else:
            pf = vbt.Portfolio.from_orders(
                close_price,
                sizes,
                # init_cash=100,
                init_cash=100,
                size_type='targetpercent',
                group_by=True,
                cash_sharing=True,
                **self.pf_kwargs
            )
            if calledby == 'add':
                RARMs = eval(f"pf.{self.param_dict['RARM']}()")
                if isinstance(RARMs, pd.Series):
                    idxmax = RARMs[RARMs != np.inf].idxmax()
                    if self.output_bool:
                        plot_CSCV(pf, idxmax, self.param_dict['RARM'])
                        
                    pf = pf[idxmax]
                
                    params_value = entries.columns[idxmax*num_symbol]
                    self.param_dict = dict(zip(['days_before', 'days_after', 'signal_threshold'], [int(params_value[0]), int(params_value[1]), signal_threshold[0]]))
                else:
                    self.param_dict = dict(zip(['days_before', 'days_after', 'signal_threshold'], [int(days_before[0]), int(days_after[0]), signal_threshold[0]]))
        self.pf = pf
        return True
