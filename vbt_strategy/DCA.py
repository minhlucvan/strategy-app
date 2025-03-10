import numpy as np
import pandas as pd
import vectorbt as vbt
import streamlit as st

from .base import BaseStrategy
from utils.vbt import init_vbtsetting, plot_CSCV

def compute_dca_orders(price_df, investment_amount):
    """
    Generate buy orders for a Dollar-Cost Averaging (DCA) strategy.

    - Buys on the first trading day of each month.
    - Accumulates assets over time (no selling).
    
    Parameters:
    - price_df (pd.DataFrame): Asset price data (daily timeframe).
    - investment_amount (float): Total capital allocated per month.

    Returns:
    - orders_df (pd.DataFrame): Order sizes (in shares).
    """
    orders_df = pd.DataFrame(index=price_df.index, columns=price_df.columns, dtype=float)
    
    # Identify the first trading day of each month
    first_trading_days = price_df.resample('M').first().dropna().index
    num_assets = len(price_df.columns)
    monthly_allocation = investment_amount / num_assets

    # Buy each month on the first trading day
    for date in first_trading_days:
        if date in price_df.index:
            for asset in price_df.columns:
                price = price_df.loc[date, asset]
                if pd.notna(price) and price > 0:  # Check for valid price
                    # Simply buy the monthly allocation worth of shares each month
                    orders_df.loc[date, asset] = monthly_allocation / price

    return orders_df.fillna(0)  # Fill NaN with 0 instead of ffill

class DCAStrategy(BaseStrategy):
    '''Dollar-Cost Averaging (DCA) Strategy'''
    _name = "DCA"
    desc = "This strategy invests a fixed amount periodically across multiple assets, reducing market timing risks."
    stacked_bool = True

    @vbt.cached_method
    def run(self, calledby='add'):
        stocks_df = self.stocks_df
        
        investment_amount = self.param_dict.get('investment_amount', 10_000_000)
        sizes = compute_dca_orders(stocks_df, investment_amount)
        
        # Reindex sizes to match stocks_df index
        sizes = sizes.reindex(stocks_df.index)
                
        init_vbtsetting()
        self.pf_kwargs = dict(fees=0.001, slippage=0.001, freq='1D')
        
        # Simulate portfolio using Vectorbt
        pf = vbt.Portfolio.from_orders(
            close=stocks_df,
            size=sizes,
            size_type='targetamount',
            direction='longonly',
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