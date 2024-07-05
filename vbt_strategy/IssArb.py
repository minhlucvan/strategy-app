import numpy as np
import pandas as pd
from datetime import timezone

import streamlit as st
import vectorbt as vbt
import utils.config as cfg
import datetime
from utils.tcbs_api import TCBSAPI


from indicators.EventArb import generate_arbitrage_signal, get_EventArbInd
from utils.processing import get_stocks_events
from utils.tcbs_agent import load_calender_data_tp_df

from .base import BaseStrategy
from utils.vbt import plot_CSCV

class IssArbStrategy(BaseStrategy):
    '''issuance Arbitrage strategy'''
    _name = "IssArb"
    desc = "Issuance Arbitrage strategy aiming to capture the price difference between the issuance date and the dividend payment date. the study shown that for a bunch of stocks, the price of the stock will be raised before the issuance date and drop after the dividend payment date, the strategy aims to capture the price difference between the two dates"
    param_def = [
        {
            "name": 'days_before',
            "type": "int",
            "min":  1,
            "max":  9,
            "step": 1
        }, {
            "name": 'days_after',
            "type": "int",
            "min":  -3,
            "max":  9,
            "step": 1
        }
    ]
    stacked_bool = True

    @ vbt.cached_method
    def run(self, calledby='add'):
        stocks_df = self.stocks_df
        self.bm_symbol = 'VN30'
        
        bm_df= self.datas.get_stock(self.bm_symbol, self.start_date, self.end_date)
        self.bm_price = bm_df['close']
        
        # align the benchmark price with the stocks price
        self.bm_price = self.bm_price.reindex(stocks_df.index, method='nearest')
        
        # 1. initialize the variables

        close_price = stocks_df
        
        events_df = get_stocks_events(self.symbolsDate_dict, 'stockIssuePrice')
        
        symbols = self.symbolsDate_dict['symbols']
        
        
        if self.is_live:
            tcbs_id = cfg.get_config('tcbs.info.TCBSId')
            auth_token = cfg.get_config('tcbs.info.authToken')
                        
            tcbs_api = TCBSAPI(auth_token=auth_token)
            
            from_date = datetime.datetime.now().strftime('%Y-%m-%d')
            to_date = (datetime.datetime.now() + datetime.timedelta(days=30)).strftime('%Y-%m-%d')
        
            new_events = tcbs_api.get_market_calendar(tcbs_id=tcbs_id, from_date=from_date, to_date=to_date)
            
            new_events_df = load_calender_data_tp_df(new_events)
                        
            # filter defType = 'stock.div010' 
            new_events_df = new_events_df[new_events_df['defType'] == 'stock.rights']
            
            
            # filter the events that are not in the symbols
            new_events_df = new_events_df[new_events_df['mkCode'].isin(symbols)]
            
            if not new_events_df.empty:
                st.write('New events found')
                st.write(new_events_df)
                        
            for i, row in new_events_df.iterrows():
                date = row['date']
                date = datetime.datetime.strptime(date, '%Y-%m-%d')
                index_value = pd.Timestamp(date).tz_localize(None)   
                
                if row['mkCode'] in events_df.columns:
                    events_df = pd.concat([events_df, pd.DataFrame(index=[index_value], columns=events_df.columns)], axis=0)
                    events_df[row['mkCode']][index_value] = row['ratio'] * 100
                            
        
                
        # if the label is not I, A, then set it to nan
        for stock in events_df.columns:
            for i in range(len(events_df)):
                if events_df[stock][i] > 0:
                    events_df[stock][i] = events_df[stock][i]
                else:
                    events_df[stock][i] = np.nan
        
        days_to_event = generate_arbitrage_signal(stocks_df, events_df)
           
        days_before = self.param_dict['days_before']
        days_after = self.param_dict['days_after']
        
        # 2. calculate the indicators
        indicator = get_EventArbInd().run(
            close_price,
            days_to_event,
            days_before_threshold=days_before,
            days_after_threshold=days_after,
            param_product=True
        )    
                
        # # 3. remove all the name in param_def from param_dict
        for param in self.param_def:
            del self.param_dict[param['name']]

        # # 4. generate the vbt signal
        entries = indicator.entries.vbt.signals.fshift()
        exits = indicator.exits.vbt.signals.fshift()
                
        num_symbol = len(stocks_df.columns)
        num_group = int(len(entries.columns) / num_symbol)
        group_list = []
        for n in range(num_group):
            group_list.extend([n]*num_symbol)
        group_by = pd.Index(group_list, name='group')

        # 5. Build portfolios
        if 'WFO' in self.param_dict and self.param_dict['WFO'] != 'None':
            entries, exits = self.maxRARM_WFO(
                close_price, entries, exits, calledby)
            pf = vbt.Portfolio.from_signals(
                close=close_price,
                entries=entries,
                exits=exits,
                group_by=group_by,
                cash_sharing=True,  # share capital between columns
                **self.pf_kwargs
            )
        else:
            pf = vbt.Portfolio.from_signals(
                close=close_price,
                entries=entries,
                exits=exits,
                init_cash=100,
                size=0.25,
                size_type='percent',
                group_by=group_by,
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
                    self.param_dict = dict(zip(['days_before', 'days_after'], [int(params_value[0]), int(params_value[1])]))
                else:
                    self.param_dict = dict(zip(['days_before', 'days_after'], [int(days_before[0]), int(days_after[0])]))     
        self.pf = pf
        return True
