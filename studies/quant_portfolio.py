import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, Dict, List
from utils.processing import get_stocks

# Configuration remains unchanged
CONFIG = {
    'risk_free_rate': 0.045,
    'transaction_cost': 0.002,
    'freq': 52
}

@dataclass
class Trade:
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    ticker: str
    entry_price: float
    exit_price: float
    
    def net_return(self, cost: float) -> float:
        gross = self.exit_price / self.entry_price
        return gross - cost - 1

class FactorStrategy(ABC):
    def __init__(self, prices: pd.DataFrame, config: Dict = CONFIG):
        self.prices = prices.fillna(method='ffill')
        self.config = config
        self.returns = self._calculate_base_returns()
        
    def _calculate_base_returns(self) -> pd.DataFrame:
        return self.prices.pct_change() + 1
    
    @abstractmethod
    def select_assets(self, date: pd.Timestamp) -> List[str]:
        pass
    
    def execute(self) -> Tuple[pd.Series, pd.DataFrame]:
        portfolio_returns = []
        trades = []
        
        for i, date in enumerate(self.returns.index[1:-1]):
            try:
                selected_tickers = self.select_assets(date)
                if not selected_tickers:
                    continue
                    
                next_date = self.returns.index[i + 2]
                period_return = 0
                
                for ticker in selected_tickers:
                    entry_price = float(self.prices.loc[date, ticker])
                    exit_price = float(self.prices.loc[next_date, ticker])
                    trade = Trade(date, next_date, ticker, entry_price, exit_price)
                    net_ret = trade.net_return(self.config['transaction_cost'])
                    trades.append({
                        'entry_date': date,
                        'exit_date': next_date,
                        'ticker': ticker,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'net_return': net_ret  # Add net_return to trade log
                    })
                    period_return += net_ret + 1
                
                portfolio_returns.append(period_return / len(selected_tickers))
                
            except (KeyError, ValueError) as e:
                st.warning(f"Data issue at {date}: {str(e)}")
                continue
                
        portfolio_returns = pd.Series(portfolio_returns, 
                                    index=self.returns.index[1:len(portfolio_returns)+1])
        return portfolio_returns, pd.DataFrame(trades)


class PerformanceMetrics:
    # Unchanged
    @staticmethod
    def sharpe_ratio(returns: pd.Series, config: Dict) -> float:
        excess = returns - config['risk_free_rate'] / config['freq']
        return np.sqrt(config['freq']) * excess.mean() / excess.std()
    
    @staticmethod
    def win_rate(returns: pd.Series) -> float:
        return (returns > 1).mean()
    
    @staticmethod
    def cumulative_returns(returns: pd.Series) -> pd.Series:
        return returns.cumprod() - 1

def create_visualizations(data: pd.DataFrame, title: str, x: str, y: str, color: str = None):
    fig = px.bar(data, x=x, y=y, color=color, title=title) if color else px.histogram(data, x=x, nbins=20)
    fig.update_layout(yaxis_tickformat='.0%', hovermode='x unified')
    return fig