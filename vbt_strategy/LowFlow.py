import numpy as np
import pandas as pd
import vectorbt as vbt
from datetime import timezone
from .base import BaseStrategy
from utils.vbt import plot_CSCV
import streamlit as st

class LowFlowStrategy(BaseStrategy):
    '''Low Flow strategy using z-scores of volume and price with EMA filter'''
    _name = "LOWFLOW"
    desc = "This strategy buys when volume z-score < -1 & price z-score < -1 and price > 100-day EMA, holds for 3 days"
    timeframe = 'D'
    stacked_bool = True
    column = None
    
    param_def = [
        {
            "name": "window",
            "type": "int",
            "min": 5,
            "max": 500,
            "step": 5,
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
            "value": 3
        },
        {
            "name": "threshold",
            "type": "float",
            "min": -3.0,
            "max": 0.0,
            "step": 0.1,
            "control": "single",
            "value": -1.0
        },
        {
            "name": "ema_window",
            "type": "int",
            "min": 50,
            "max": 200,
            "step": 10,
            "control": "single",
            "value": 100
        }
    ]

    def calculate_zscores(self, close_df, volume_df, window):
        """Calculate z-scores for price and volume (matching analysis approach)"""
        # Calculate z-scores using raw price and volume (as in analysis)
        price_zscore = (close_df - close_df.rolling(window=window).mean()) / close_df.rolling(window=window).std()
        volume_zscore = (volume_df - volume_df.rolling(window=window).mean()) / volume_df.rolling(window=window).std()
        
        return price_zscore, volume_zscore

    def generate_signals(self, close_df, volume_df, window, hold_period, threshold, ema_window):
        """Generate buy signals based on z-score conditions and EMA filter"""
        # Calculate z-scores
        price_zscore, volume_zscore = self.calculate_zscores(close_df, volume_df, window)
        
        # Calculate EMA
        ema = close_df.ewm(span=ema_window, adjust=False).mean()
        
        # Entry conditions (matching analysis)
        price_condition = price_zscore < threshold
        volume_condition = volume_zscore < threshold
        trend_condition = close_df > ema
        
        # Combine conditions
        entries = price_condition & volume_condition & trend_condition
        
        # Generate exit signals after hold_period days
        exits = entries.shift(hold_period).fillna(False)
        
        return entries, exits
    
    @vbt.cached_method
    def run(self, calledby='add'):
        """Execute the trading strategy"""
        self.stocks_df = self.stocks_df.fillna(method='ffill')
        close_df = self.stocks_df['close']
        volume_df = self.stocks_df['volume']
        
        # Get parameters
        window = self.param_dict['window'][0]
        hold_period = self.param_dict['hold_period'][0]
        threshold = self.param_dict['threshold'][0]
        ema_window = self.param_dict['ema_window'][0]
        
        if calledby == 'update':
            window = self.param_dict['window'][0][0]
            hold_period = self.param_dict['hold_period'][0][0]
            threshold = self.param_dict['threshold'][0][0]
            ema_window = self.param_dict['ema_window'][0][0]
            
        # Generate signals
        entries, exits = self.generate_signals(
            close_df, 
            volume_df, 
            window, 
            hold_period, 
            threshold, 
            ema_window
        )
        
        # Initialize vectorbt settings
        self.pf_kwargs = dict(fees=0.001, slippage=0.001, freq='1D')
        
        # Check if WFO is requested
        if self.param_dict.get('WFO', 'None') != 'None':
            raise NotImplementedError('WFO not implemented')
        
        # Run portfolio simulation
        pf = vbt.Portfolio.from_signals(
            close=close_df,
            entries=entries,
            exits=exits,
            size=1.0,
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