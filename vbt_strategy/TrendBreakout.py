import numpy as np
import pandas as pd
import vectorbt as vbt
from datetime import timezone
from .base import BaseStrategy
from utils.vbt import plot_CSCV
import streamlit as st
import talib

class TrendBreakoutStrategy(BaseStrategy):
    '''Trend Breakout strategy to detect end of downtrend and start of uptrend'''
    _name = "TREND_BREAKOUT"
    desc = "This strategy buys when price is below MA10, volume exceeds 20-day average, and shows high volatility"
    timeframe = 'D'
    stacked_bool = True
    column = None
    
    param_def = [
        {
            "name": "ma_window",
            "type": "int",
            "min": 5,
            "max": 50,
            "step": 1,
            "control": "single",
            "value": 10
        },
        {
            "name": "vol_window",
            "type": "int",
            "min": 5,
            "max": 50,
            "step": 1,
            "control": "single",
            "value": 20
        },
        {
            "name": "hold_period",
            "type": "int",
            "min": 1,
            "max": 20,
            "step": 1,
            "control": "single",
            "value": 5
        },
        {
            "name": "llr_threshold",
            "type": "float",
            "min": 0.0,
            "max": 30.0,
            "step": 0.1,
            "control": "single",
            "value": 2.0
        },
        {
            "name": "volatility_threshold",
            "type": "float",
            "min": 0.0,
            "max": 30.0,
            "step": 0.1,
            "control": "single",
            "value": 2.0
        }
    ]


    def generate_signals(self, open_df, high_df, low_df, close_df, volume_df, 
                        ma_window, vol_window, hold_period, llr_threshold, volatility_threshold):
        """Generate buy signals based on trend breakout conditions"""
        # Calculate 10-day moving average
        ma10 = close_df.rolling(window=ma_window).mean()
        
        # Calculate 20-day average volume
        avg_vol = volume_df.rolling(window=vol_window).mean()
        
        # find large volatility days
        volatility = high_df - low_df
        avg_volatility = volatility.rolling(window=vol_window).mean()
        
        # Define entry conditions
        price_condition = close_df < ma10
        volume_condition = volume_df > avg_vol * llr_threshold
        volatility_condition = volatility > avg_volatility * volatility_threshold
        
        # Combine all conditions
        entries = price_condition & volume_condition & volatility_condition
        
        # Generate exit signals after hold_period days
        exits = entries.shift(hold_period).fillna(False)
        
        return entries, exits

    @vbt.cached_method
    def run(self, calledby='add'):
        """Execute the trading strategy"""
        self.stocks_df = self.stocks_df.fillna(method='ffill')
        
        # Extract required data
        close_df = self.stocks_df['close']
        open_df = self.stocks_df['open']
        high_df = self.stocks_df['high']
        low_df = self.stocks_df['low']
        volume_df = self.stocks_df['volume']
        
        # Get parameters
        ma_window = self.param_dict['ma_window'][0]
        vol_window = self.param_dict['vol_window'][0]
        hold_period = self.param_dict['hold_period'][0]
        llr_threshold = self.param_dict['llr_threshold'][0]
        volatility_threshold = self.param_dict['volatility_threshold'][0]
        
        if calledby == 'update':
            ma_window = self.param_dict['ma_window'][0][0]
            vol_window = self.param_dict['vol_window'][0][0]
            hold_period = self.param_dict['hold_period'][0][0]
            
        # Generate signals
        entries, exits = self.generate_signals(
            open_df, high_df, low_df, close_df, volume_df,
            ma_window, vol_window, hold_period,
            llr_threshold, volatility_threshold
        )
        
        # Initialize vectorbt settings
        self.pf_kwargs = dict(fees=0.001, slippage=0.001, freq='1D')
        
        # Check if WFO (Walk-Forward Optimization) is requested
        if self.param_dict.get('WFO', 'None') != 'None':
            raise NotImplementedError('WFO not implemented')
        
        # Run portfolio simulation
        pf = vbt.Portfolio.from_signals(
            close=close_df,
            entries=entries,
            exits=exits,
            size=0.2,
            init_cash=50_000_000,
            size_type='percent',
            direction='longonly',
            group_by=None,
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
        
        self.pf = pf
        return True
