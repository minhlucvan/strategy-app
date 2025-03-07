import numpy as np
import pandas as pd
import vectorbt as vbt
from datetime import timezone
from .base import BaseStrategy
from utils.vbt import plot_CSCV
import streamlit as st 


class LiqLockStrategy(BaseStrategy):
    '''Liquidity Lock strategy that buys on liquidity breakout and sells after 3 days'''
    _name = "LIQUIDITYLOCK"
    desc = "This strategy buys stocks when their liquidity breaks above the 14-day average and holds for 3 days"
    timeframe = 'D'
    stacked_bool = True
    column = None

    def calculate_liquidity(self, close_df, volume_df):
        """Calculate liquidity as price change * volume"""
        price_change = close_df.pct_change()
        volume_df = volume_df.fillna(0)
        return price_change * volume_df
        
    def generate_signals(self, close_df, volume_df, window, hold_period, threshold):
        """Generate buy signals based on liquidity breakout"""
        # Calculate liquidity
        liquidity_df = self.calculate_liquidity(close_df, volume_df)
        
        # Calculate rolling mean over window period
        liquidity_avg = liquidity_df.rolling(window=window).mean()
        
        liquidity_z_score_df = (liquidity_df - liquidity_avg) / liquidity_df.std()
        
        # Generate signals when liquidity z-score is above threshold
        entries = liquidity_z_score_df > threshold
        
        # Generate exit signals after hold_period days
        exits = entries.shift(hold_period).fillna(False)
        
        return entries, exits
    
    @vbt.cached_method
    def run(self, calledby='add'):  # Define calledby as a parameter with default value
        """Execute the trading strategy"""
        
        # filna with ffill
        self.stocks_df = self.stocks_df.fillna(method='ffill')
        
        close_df = self.stocks_df['close']
        volume_df = self.stocks_df['volume']
        
        window = 14
        hold_period = 3
        threshold = 2.0
        
        # Generate signals
        entries, exits = self.generate_signals(close_df, volume_df, window, hold_period, threshold)
        
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
            size=1.0,  # Equal weighting
            size_type='percent',
            direction='longonly',
            **self.pf_kwargs
        )
        
        # Handle the calledby logic
        if calledby == 'add':
            RARMs = eval(f"pf.{self.param_dict['RARM']}()")
            if isinstance(RARMs, pd.Series):
                idxmax = RARMs[RARMs != np.inf].idxmax()
                if self.output_bool:
                    plot_CSCV(pf, idxmax, self.param_dict['RARM'])
                pf = pf[idxmax]
        
        self.pf = pf
        return True