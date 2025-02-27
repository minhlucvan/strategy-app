import pandas as pd
import plotly.graph_objects as go
import numpy as np
import streamlit as st
from utils.processing import get_stocks

class DCABacktest:
    def __init__(self, df, initial_capital=10_000_000, buy_amount=1_000_000):
        self.df = df.copy()
        self.initial_capital = initial_capital
        self.buy_amount = buy_amount
        
        # Zone strategy variables
        self.zone_cash = initial_capital
        self.zone_shares = 0
        self.zone_trades = []
        
        # Monthly DCA strategy variables (will be initialized if DCA is enabled)
        self.dca_cash = initial_capital
        self.dca_shares = 0
        self.dca_trades = []

    def calculate_zones(self, period=21, min_signal_gap=5, run_dca=False):
        """Calculate Fibonacci zones and trading signals"""
        df = self.df
        df['hl'] = df['high'].rolling(window=period).max()
        df['ll'] = df['low'].rolling(window=period).min()
        df['dist'] = df['hl'] - df['ll']
        
        df['hf'] = df['hl'] - df['dist'] * 0.236    # Greed zone
        df['lf'] = df['hl'] - df['dist'] * 0.764    # Fear zone
        
        df['prev_close'] = df['close'].shift(1)
        df['buy_signal'] = (df['prev_close'] > df['lf']) & (df['close'] <= df['lf'])
        df['sell_signal'] = (df['prev_close'] < df['hf']) & (df['close'] >= df['hf'])
        
        # Filter signals with minimum gap
        df['buy_exec'] = False
        df['sell_exec'] = False
        
        last_buy = -min_signal_gap - 1
        last_sell = -min_signal_gap - 1
        
        for i in range(len(df)):
            if df['buy_signal'].iloc[i] and (i - last_buy > min_signal_gap) and self.zone_cash >= self.buy_amount:
                df.loc[df.index[i], 'buy_exec'] = True
                last_buy = i
            if df['sell_signal'].iloc[i] and (i - last_sell > min_signal_gap) and self.zone_shares > 0:
                df.loc[df.index[i], 'sell_exec'] = True
                last_sell = i
        
        # Add monthly DCA signal only if enabled
        if run_dca:
            df['month'] = df.index.to_series().dt.to_period('M')
            df['is_first_day_of_month'] = df.groupby('month').cumcount() == 0
            df['dca_buy'] = df['is_first_day_of_month'] & (self.dca_cash >= self.buy_amount)
        
        self.df = df

    def run_backtest(self, run_dca=False):
        """Execute the Zone strategy and optionally the DCA strategy"""
        df = self.df
        
        for i in range(len(df)):
            # Zone Strategy
            if df['buy_exec'].iloc[i]:
                shares_to_buy = self.buy_amount / df['close'].iloc[i]
                self.zone_shares += shares_to_buy
                self.zone_cash -= self.buy_amount
                self.zone_trades.append({
                    'date': df.index[i],
                    'type': 'BUY',
                    'price': df['close'].iloc[i],
                    'shares': shares_to_buy,
                    'cash': self.zone_cash,
                    'total_shares': self.zone_shares
                })
            
            if df['sell_exec'].iloc[i]:
                sell_value = self.zone_shares * df['close'].iloc[i]
                self.zone_cash += sell_value
                self.zone_trades.append({
                    'date': df.index[i],
                    'type': 'SELL',
                    'price': df['close'].iloc[i],
                    'shares': self.zone_shares,
                    'cash': self.zone_cash,
                    'total_shares': 0
                })
                self.zone_shares = 0
            
            # Monthly DCA Strategy (if enabled)
            if run_dca and df['dca_buy'].iloc[i]:
                dca_shares_to_buy = self.buy_amount / df['close'].iloc[i]
                self.dca_shares += dca_shares_to_buy
                self.dca_cash -= self.buy_amount
                self.dca_trades.append({
                    'date': df.index[i],
                    'type': 'BUY',
                    'price': df['close'].iloc[i],
                    'shares': dca_shares_to_buy,
                    'cash': self.dca_cash,
                    'total_shares': self.dca_shares
                })
        
        # Calculate final metrics
        final_price = df['close'].iloc[-1]
        zone_final_value = self.zone_cash + (self.zone_shares * final_price)
        self.metrics = {
            'zone': {
                'initial_capital': self.initial_capital,
                'final_value': zone_final_value,
                'profit': zone_final_value - self.initial_capital,
                'profit_pct': (zone_final_value - self.initial_capital) / self.initial_capital * 100,
                'num_trades': len(self.zone_trades),
                'buy_trades': len([t for t in self.zone_trades if t['type'] == 'BUY']),
                'sell_trades': len([t for t in self.zone_trades if t['type'] == 'SELL'])
            }
        }
        
        if run_dca:
            dca_final_value = self.dca_cash + (self.dca_shares * final_price)
            self.metrics['dca'] = {
                'initial_capital': self.initial_capital,
                'final_value': dca_final_value,
                'profit': dca_final_value - self.initial_capital,
                'profit_pct': (dca_final_value - self.initial_capital) / self.initial_capital * 100,
                'num_trades': len(self.dca_trades),
                'buy_trades': len([t for t in self.dca_trades if t['type'] == 'BUY']),
                'sell_trades': 0  # DCA doesn't sell
            }
    def plot_results(self):
        """Visualize the Zone strategy results"""
        df = self.df
        fig = go.Figure()
        
        # Price line
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['close'],
            mode='lines',
            name='Price'
        ))
        
        # Greed zone
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['hl'],
            line=dict(color='rgba(0,255,255,0.2)'),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['hf'],
            fill='tonexty',
            fillcolor='rgba(0,255,255,0.2)',
            line=dict(color='rgba(0,255,255,0.2)'),
            name='Greed Zone'
        ))
        
        # Fear zone
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['lf'],
            line=dict(color='rgba(255,165,0,0.2)'),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['ll'],
            fill='tonexty',
            fillcolor='rgba(255,165,0,0.2)',
            line=dict(color='rgba(255,165,0,0.2)'),
            name='Fear Zone'
        ))
        
        # Zone Buy/Sell signals
        buys = df[df['buy_exec']]
        if not buys.empty:
            fig.add_trace(go.Scatter(
                x=buys.index,
                y=buys['close'],
                mode='markers',
                marker=dict(symbol='triangle-up', size=12, color='green'),
                name='Buy'
            ))
        
        sells = df[df['sell_exec']]
        if not sells.empty:
            fig.add_trace(go.Scatter(
                x=sells.index,
                y=sells['close'],
                mode='markers',
                marker=dict(symbol='triangle-down', size=12, color='red'),
                name='Sell'
            ))
        
        fig.update_layout(
            title='Zone Strategy: Buy in Fear Zone, Sell in Greed Zone',
            yaxis_title='Price',
            showlegend=True,
            template='plotly_dark',
            hovermode='x unified'
        )
        
        return fig

def run(symbol_benchmark, symbolsDate_dict):
    if len(symbolsDate_dict['symbols']) == 0:
        st.info("Please select symbols.")
        st.stop()
    
    # Get data
    stock_df = get_stocks(symbolsDate_dict, single=True)
    sample_data = stock_df[['open', 'high', 'low', 'close']]
    # Initialize and run backtest
    backtest = DCABacktest(sample_data)
    backtest.calculate_zones(period=21, min_signal_gap=5, run_dca=False)
    backtest.run_backtest(run_dca=False)
    
    # Plot (only Zone strategy)
    fig = backtest.plot_results()
    fig.update_yaxes(fixedrange=False)
    fig.update_layout(height=600, margin=dict(l=0, r=0, t=40, b=0))
    fig.update_xaxes(rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Display results
    st.subheader("Backtest Results")
    
    st.write("### Zone Strategy (Buy Fear, Sell Greed)")
    zone_metrics = backtest.metrics['zone']
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Initial Capital", f"{zone_metrics['initial_capital']:,.2f}")
        st.metric("Final Value", f"{zone_metrics['final_value']:,.2f}")
        st.metric("Profit", f"{zone_metrics['profit']:,.2f}")
    with col2:
        st.metric("Profit %", f"{zone_metrics['profit_pct']:,.2f}%")
        st.metric("Total Trades", zone_metrics['num_trades'])
        st.metric("Buy/Sell Trades", f"{zone_metrics['buy_trades']}/{zone_metrics['sell_trades']}")
    
    # plot total_returns
    st.write("### asset_value")
    zone_trade_df = pd.DataFrame(backtest.zone_trades)
    zone_trade_df['asset_value'] = zone_trade_df['price'] * zone_trade_df['total_shares']
    st.line_chart(zone_trade_df['asset_value'] )

    # Show trade logs
    if st.checkbox("Show Zone Strategy Trade Log"):
        zone_trade_df = pd.DataFrame(backtest.zone_trades)
        if not zone_trade_df.empty:
            zone_trade_df['date'] = zone_trade_df['date'].dt.strftime('%Y-%m-%d %H:%M')
            st.dataframe(zone_trade_df.style.format({
                'price': '{:.2f}',
                'shares': '{:.4f}',
                'cash': '{:.2f}',
                'total_shares': '{:.4f}'
            }))
   