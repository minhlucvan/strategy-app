import pandas as pd
import numpy as np

import streamlit as st
import vectorbt as vbt

from utils.processing import AKData, get_stocks
from utils.vbt import plot_pf

def cal_TVWLength(n_days, O, R):
    '''
    calculate Train\Validate\Window days
    # t + t/(1-o/100)*(o/100)*r = n
    '''
    train_length = round(n_days*(100-O)/(100-O+O*R))
    validate_length = round(n_days*O/(100-O+O*R))
    window = round(n_days*100/(100-O+O*R))
    return train_length, validate_length, window

class BaseStrategy(object):
    '''base strategy'''
    _name = "base"
    desc = "......"
    symbolsDate_dict = {}
    param_dict = {}
    param_def = {}
    stock_dfs = []
    symbols = []
    market = ''
    pf_kwargs = dict(fees=0.0005, slippage=0.001, freq='1D')
    pf = None
    output_bool = False
    stacked_bool = False
    stocks_df = None
    bm_symbol = None
    bm_price = None
    use_rsc = False
    inlude_bm = False
    
    def __init__(self, symbolsDate_dict:dict):
        self.symbolsDate_dict = symbolsDate_dict
        self.market = symbolsDate_dict['market']
        self.symbols = symbolsDate_dict['symbols']
        self.start_date = symbolsDate_dict['start_date']
        self.end_date = symbolsDate_dict['end_date']
        self.datas = AKData(self.market)
        self.stock_dfs = []
        self.param_dict = {}

        if self.stacked_bool:
            self.get_stocks_stacked(symbolsDate_dict)
        else:
            self.init_stocks(symbolsDate_dict)
        
        if self.use_rsc:
            self.init_rsc()
        
        # initialize param_dict using default param_def
        for param in self.param_def:
            if param["step"] == 0:
                self.param_dict[param["name"]] = [int((param["min"] + param['max'])/ 2)]
            else:
                self.param_dict[param["name"]] = np.arange(param["min"], param["max"], param["step"])
        #initialize vbt setting
        vbt.settings.array_wrapper['freq'] = 'days'
        vbt.settings.returns['year_freq'] = '252 days'
        vbt.settings.portfolio.stats['incl_unrealized'] = True
    
    def get_stocks_stacked(self, symbolsDate_dict:dict):
        symbolsDate_dict_cp = symbolsDate_dict.copy()
        
        if self.inlude_bm and self.bm_symbol is not None:
            symbolsDate_dict_cp['symbols'].append(self.bm_symbol)
        
        self.stocks_df = get_stocks(symbolsDate_dict_cp, column='close')
        
        if self.inlude_bm and self.bm_symbol is not None:
            self.bm_price = self.stocks_df[self.bm_symbol]
            self.stocks_df = self.stocks_df.drop(self.bm_symbol, axis=1)
        
    def init_stocks(self, symbolsDate_dict:dict):
        for symbol in symbolsDate_dict['symbols']:
            if symbol!='':
                stock_df = self.datas.get_stock(symbol, self.start_date, self.end_date)
                if stock_df.empty:
                    print(f"Warning: stock '{symbol}' is invalid or missing. Ignore it")
                else:
                    self.stock_dfs.append((symbol, stock_df))

    def init_rsc(self):
        if self.bm_price is None:
            raise ValueError("Benchmark price is not initialized please config bm_symbol and include_bm to True")
                
        if self.stacked_bool:
            for col in self.stocks_df.columns.get_level_values(0).unique():
                self.stocks_df[col] = self.stocks_df[col] / self.bm_price
        else:
            for i in range(len(self.stock_dfs)):
                # calculate the relative strength
                self.stock_dfs[i][1]['close'] = self.stock_dfs[i][1]['close'] / self.bm_price['close']

    def log(self, txt, dt=None, doprint=False):
        pass

    def validate(self):
        if self.stacked_bool:
            return len(self.stocks_df) > 0
        
        return len(self.stock_dfs) > 0

    def get_assets_identifier(self):
        if self.symbolsDate_dict['group_name'] and len(self.symbolsDate_dict['group_name']) > 0:
            return self.symbolsDate_dict['group_name']
        
        if self.stacked_bool:
            cols = self.stocks_df.columns.get_level_values(0).unique()
            if len(cols) > 3:
                return '_'.join(cols[:3].values.tolist()) + f'__{len(cols)-3}'
            return '_'.join(cols.values.tolist())
        
        return self.stock_dfs[0][0]
    
    def maxRARM(self, param, output_bool=False):
        '''
        Maximize Risk-Adjusted Return Measurement
        '''
        self.param_dict.update(param)
        self.output_bool = output_bool
        
        if self.param_dict['WFO'] == 'None':
            self.param_dict['WFO'] = False
        
        # try:
        if True:
            if self.run(calledby='add'):
                if self.output_bool:
                    plot_pf(self.pf, name=self._name+ '_' + self.get_assets_identifier(), bm_symbol=self.bm_symbol, bm_price=self.bm_price)
                return True
            else:
                return False
        # except Exception as e:
        #     print(f"{self._name}-maxSR throws exception: {e}")
        #     return False

    def update(self, param_dict:dict):
        """
            update the strategy with the param dictiorary saved in portfolio
        """
        if len(self.stock_dfs) == 0:
            return None
        else:
            for k, v in param_dict.items():
                if k in ['RARM', 'WFO']:
                    self.param_dict[k] = v
                else:
                    self.param_dict[k] = [v]
            self.run(calledby='update')
            return self.pf

    def maxRARM_WFO(self, price, entries, exits, calledby='add'):
        '''
        Walk Foreward Optimization:
        Risk-Adjusted Return and Measurement Methods
        '''
        if calledby == 'add':
            Runs = [10, 15, 20, 25, 30]
            OOS = [10, 20, 30, 40, ]
        else:
            Runs = self.param_dict['WFO_Run']
            OOS =  self.param_dict['WFO_OOS']
        n_days = len(price)
        wfms_df = pd.DataFrame(columns=OOS, index=Runs)

        update_bar = st.progress(0)
        max_return = float('-inf')
        max_entries = np.full_like(price, False)
        max_exits = np.full_like(price, False)
        max_R = 0
        max_O = 0
        for i, R in enumerate(Runs):
            for O in OOS:
                train_length, validate_length, window = cal_TVWLength(n_days, O, R)
                tmp_entries = np.full_like(price, False)
                tmp_exits = np.full_like(price, False)
                for m in range(0, n_days-train_length, validate_length):
                    if self.param_dict['WFO'] == 'Non-anchored':
                        train_start = m
                    else:
                        train_start = 0
                    train_end = m - validate_length + train_length
                    pf = vbt.Portfolio.from_signals(price[train_start: train_end], 
                                                    entries[train_start: train_end], 
                                                    exits[train_start: train_end], 
                                                    **self.pf_kwargs)
                    RARMs = eval(f"pf.{self.param_dict['RARM']}()")
                    if len(RARMs[RARMs != np.inf]) > 0:
                        idxmax = RARMs[RARMs != np.inf].idxmax()
                        if(not pd.isna(idxmax)):
                            tmp_entries[train_end: ] = entries[train_end: ][idxmax]
                            tmp_exits[train_end: ] = exits[train_end: ][idxmax]

                pf = vbt.Portfolio.from_signals(price, tmp_entries, tmp_exits, **self.pf_kwargs)
                wfms_df.loc[R, O] = pf.annualized_return()
                if max_return < pf.annualized_return():
                    max_return = pf.annualized_return()
                    max_entries = tmp_entries
                    max_exits = tmp_exits
                    max_R = R
                    max_O = O
            update_bar.progress((i+1) / len(Runs))
        if self.output_bool:
            with st.expander("RUNS/OOS annualized returns' table"):
                st.table(wfms_df.style.format(formatter="{:.2%}").background_gradient(cmap='YlGn'),)

            # display max return's metric
            train_length, validate_length, window = cal_TVWLength(n_days, max_O, max_R)
            col1,col2,col3,col4,col5 = st.columns(5)
            with col1:
                st.metric("Max_Return's Run", max_R)
            with col2:
                st.metric("Max_Return's OOS", max_O)
            with col3:
                st.metric("Window", f'{window}')
            with col4:
                st.metric('Train days', train_length)
            with col5:
                st.metric('validate days', validate_length)
            
            # plot in-out of sample graph
            if max_R > 0:
                i = 0
                heatmap_df = pd.DataFrame(np.nan, columns=np.arange(max_R), index=price.index)
                for m in range(0, n_days-train_length, validate_length):
                    if self.param_dict['WFO'] == 'Non-anchored':
                        train_start = m
                    else:
                        train_start = 0
                    train_end = m + train_length
                    t_series = np.full_like(price, np.nan)
                    t_series[train_start: train_end] = 0
                    t_series[train_end: train_end+validate_length] = 1
                    heatmap_df[i] = t_series
                    i+=1
                fig = heatmap_df.vbt.ts_heatmap()
                st.plotly_chart(fig, use_container_width=True)

        self.param_dict['WFO_Run'] = max_R
        self.param_dict['WFO_OOS'] = max_O

        return max_entries, max_exits
