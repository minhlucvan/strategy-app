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
    'freq': 52,
    'holding_period': 2,
    'stop_loss': 0.0,
    'take_profit': 0.0
}

@dataclass
class Trade:
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    ticker: str
    entry_price: float
    exit_price: float
    stop_loss: float = 0.0
    take_profit: float = 0.0
    
    def net_return(self, cost: float) -> float:
        
        if np.isnan(self.exit_price):
            return 0
        
        gross = self.exit_price / self.entry_price
        
        if self.stop_loss > 0 and gross < 1 - self.stop_loss:
            gross = 1 - self.stop_loss
            
        if self.take_profit > 0 and gross > 1 + self.take_profit:
            gross = 1 + self.take_profit
        
        net = gross - cost - 1
        
        return net

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
                selected_tickers, signals = self.select_assets(date)
                if not selected_tickers:
                    continue
                
                hold_period = self.config.get('holding_period', 2)
                next_date = self.returns.index[i + hold_period] if i + hold_period < len(self.returns) else None
                period_return = 0
                
                
                for ticker in selected_tickers:
                    if next_date is None:
                        continue
                    signal = signals[ticker]
                    entry_price = float(self.prices.loc[date, ticker])
                    exit_price = float(self.prices.loc[next_date, ticker])
                    trade = Trade(date, next_date, ticker, entry_price, exit_price)
                    net_ret = trade.net_return(self.config['transaction_cost'])
                    trades.append({
                        'entry_date': date,
                        'exit_date': next_date,
                        'ticker': ticker,
                        'signal': signal,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'net_return': net_ret  # Add net_return to trade log
                    })
                    period_return += net_ret + 1
                
                portfolio_returns.append(period_return / len(selected_tickers))
                
            except (KeyError, ValueError) as e:
                raise ValueError(f"Error at {date}: {e}")
                
        portfolio_returns = pd.Series(portfolio_returns, 
                                    index=self.returns.index[1:len(portfolio_returns)+1])
        return portfolio_returns, pd.DataFrame(trades)


class PerformanceMetrics:
    # Unchanged
    @staticmethod
    def sharpe_ratio(returns: pd.Series, config: Dict) -> float:
        excess_returns = returns - config['risk_free_rate'] / config['freq']
        mean_excess = excess_returns.mean()
        std_excess = excess_returns.std()
        return mean_excess / std_excess
    
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

def create_signal_visualizations(data: pd.DataFrame, title: str, x: str, y: str, color: str = None):
    # scatter returns vs signals
    fig = px.scatter(data, x=x, y=y, color=color, title=title)
    return fig