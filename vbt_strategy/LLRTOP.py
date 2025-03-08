import numpy as np
import pandas as pd
import vectorbt as vbt
from .base import BaseStrategy
from utils.vbt import init_vbtsetting, plot_CSCV

class LLRTOPStrategy(BaseStrategy):
    '''Liquidity-Adjusted Log Return Top 3 Strategy'''
    _name = "LLRTOP3"
    desc = "This strategy selects the top 3 stocks based on Liquidity-Adjusted Log Return (LLR) calculated as (Volume × Sign(Price Change)) / Avg Volume over a 52-week window."
    stacked_bool = True
    column = None
    
    param_def = [
        {
            "name": "window",
            "type": "int",
            "min":  2,
            "max":  500,
            "step": 2,
            "control": "single",
            "value": 400
        },
        {
            "name": "top_n",
            "type": "int",
            "min":  1,
            "max":  10,
            "step": 1,
            "control": "single",
            "value": 3
        },
    ]
    
    def compute_llr(self, price_df, volume_df, window):
        """
        Compute Liquidity-Adjusted Log Return (LLR)
        LLR = (Volume × Sign(Price Change)) / Avg Volume over Window
        """
        # Calculate price change
        price_change = price_df.diff()
        
        # Calculate sign of price change (-1, 0, 1)
        price_sign = np.sign(price_change)
        
        # Calculate average volume over window
        avg_volume = volume_df.rolling(window=window, min_periods=1).mean()
        
        # Calculate LLR
        llr = (volume_df * price_sign) / avg_volume
        
        return llr

    def compute_sizes(self, price_df, volume_df, window, top_n):
        """
        Compute position sizes by selecting top 3 stocks with highest LLR
        """
        sizes_df = pd.DataFrame(index=price_df.index, columns=price_df.columns, data=0.0)
        
        # Calculate LLR
        llr = self.compute_llr(price_df, volume_df, window)
        
        for i in range(1, len(price_df)):
            prev_llr = llr.iloc[i - 1]
            if not prev_llr.empty and not prev_llr.isna().all():
                top_assets = prev_llr.nlargest(top_n).index
                size = 1 / top_n
                sizes_df.iloc[i][top_assets] = size
        
        return sizes_df

    @vbt.cached_method
    def run(self, calledby='add'):
        # Forward fill and drop NaN
        prices = self.stocks_df['close'].fillna(method='ffill').dropna()
        volumes = self.stocks_df['volume'].fillna(method='ffill').dropna()
        
        window = self.param_dict['window'][0]
        top_n = self.param_dict['top_n'][0]
        
        # Compute position sizes
        sizes = self.compute_sizes(prices, volumes, window, top_n)
        
        # Initialize vectorbt settings
        init_vbtsetting()
        self.pf_kwargs = dict(
            fees=0.001,      # 0.1% transaction fees
            slippage=0.001,  # 0.1% slippage
            freq='1W'        # Weekly rebalancing
        )

        # Build portfolio
        if self.param_dict['WFO'] != 'None':
            raise NotImplementedError('WFO not implemented')
        else:
            pf = vbt.Portfolio.from_orders(
                prices,
                sizes,
                size_type='targetpercent',
                init_cash=40_000_000,
                group_by=True,
                cash_sharing=True,
                **self.pf_kwargs,
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
