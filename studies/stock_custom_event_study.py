import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objs as go

from utils.component import check_password, input_dates, input_SymbolsDate
from utils.processing import get_stocks, get_stocks_events, get_stocks_valuation
from studies.rrg import plot_RRG, rs_ratio, RRG_Strategy
from utils.vbt import plot_pf
from vbt_strategy.MOM_D import get_MomDInd
from studies.market_wide import MarketWide_Strategy


def get_event_affection(stock_df, event_df, days_before, days_after):
    event_affection_df = pd.DataFrame(index=stock_df.index, columns=['event_price_change'])
    if event_df.empty:
        return event_affection_df

    event_df['event_date'] = pd.to_datetime(event_df.index)
    event_df['event_before_date'] = event_df['event_date'] - pd.DateOffset(days=days_before)
    event_df['event_after_date'] = event_df['event_date'] + pd.DateOffset(days=days_after)
    
    for index, row in event_df.iterrows():
        event_before_price = stock_df[stock_df.index >= row['event_before_date']].iloc[0]
        event_price = stock_df[stock_df.index >= row['event_date']].iloc[0]
        
        event_after_price_df = stock_df[stock_df.index >= row['event_after_date']]
        event_after_price = event_after_price_df.iloc[0] if not event_after_price_df.empty else np.nan

        if pd.isna(event_before_price) or pd.isna(event_price) or pd.isna(event_after_price):
            continue

        event_price_change = (event_after_price - event_before_price) / event_before_price
        event_affection_df.loc[row['event_date'], 'event_price_change'] = event_price_change

    return event_affection_df['event_price_change']

def plot_event_distributions(events_df, events_affection_df, is_numeric=True):
    event_unstack_df = events_df.unstack().reset_index()
    event_unstack_df.columns = ['Stock', 'Date', 'Signal']
    event_unstack_df = event_unstack_df.dropna()
    event_unstack_df.index = event_unstack_df['Date']

    if is_numeric:
        st.write("Signal Scatter")
        fig = px.scatter(event_unstack_df, x="Date", y="Signal", color="Stock")
        st.plotly_chart(fig)

        st.write("Signal Distribution")
        fig = px.histogram(event_unstack_df, x="Signal", color="Stock", marginal="box", nbins=50)
        st.plotly_chart(fig)

    events_affection_unstack_df = events_affection_df.unstack().reset_index()
    events_affection_unstack_df.columns = ['Stock', 'Date', 'Price Change']
    events_affection_unstack_df = events_affection_unstack_df.dropna()
    events_affection_unstack_df.index = events_affection_unstack_df['Date']

    st.write("Price Change Distribution")
    fig = px.histogram(events_affection_unstack_df, x="Price Change", color="Stock", marginal="box", nbins=50)
    st.plotly_chart(fig)

    st.write("Price Change Scatter")
    fig = px.scatter(events_affection_unstack_df, x="Date", y="Price Change", color="Stock")
    st.plotly_chart(fig)

    return events_affection_unstack_df

def plot_event_summary(events_affection_unstack_df, benchmark_df, symbol_benchmark):
    events_affection_unstack_daily_df = events_affection_unstack_df.groupby(events_affection_unstack_df.index).agg({'Price Change': 'mean', 'Date': 'count'})
    events_affection_unstack_daily_df.columns = ['Price Change', 'total']

    st.write("Total Trade by Date")
    fig = px.bar(events_affection_unstack_daily_df, x=events_affection_unstack_daily_df.index, y="total")
    st.plotly_chart(fig)

    st.write("Price Change Mean by Date")
    fig = px.bar(events_affection_unstack_daily_df, x=events_affection_unstack_daily_df.index, y="Price Change",
                 color=events_affection_unstack_daily_df['Price Change'].apply(lambda x: 'green' if x >= 0 else 'red'))
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig)

    events_affection_unstack_daily_cumsum_df = events_affection_unstack_daily_df.cumsum()

    benchmark_return_cumsum = benchmark_df.pct_change().cumsum()
    benchmark_return_cumsum = benchmark_return_cumsum[benchmark_return_cumsum.index >= events_affection_unstack_daily_cumsum_df.index[0]]

    st.write("Price Change Cumsum by Date")
    fig = px.line(events_affection_unstack_daily_cumsum_df, x=events_affection_unstack_daily_cumsum_df.index, y="Price Change")
    fig.add_trace(go.Scatter(x=benchmark_return_cumsum.index, y=benchmark_return_cumsum[symbol_benchmark], mode='lines', name='Benchmark'))
    st.plotly_chart(fig)
    
    # st.write("Faily Return Cumsum (Devisify)")
    # st.write("By devesify, we can see the return of the strategy when allocating the same amount of money to each stock.")
    # fig = px.line(events_affection_unstack_daily_df, x=events_affection_unstack_daily_df.index, y="Price Change")
    # st.plotly_chart(fig)

    max_drawdown = events_affection_unstack_daily_cumsum_df['Price Change'].min()
    st.write(f"Max Drawdown: {max_drawdown}")

    annualized_return = events_affection_unstack_daily_cumsum_df['Price Change'].iloc[-1] / len(events_affection_unstack_daily_cumsum_df) * 252
    st.write(f"Annualized Return: {annualized_return}")

    benchmark_annualized_return = benchmark_return_cumsum[symbol_benchmark].iloc[-1] / len(benchmark_return_cumsum) * 252
    st.write(f"Benchmark Annualized Return: {benchmark_annualized_return}")

    daily_return = events_affection_unstack_daily_df['Price Change'].dropna()
    sharpe_ratio = daily_return.mean() / daily_return.std() * np.sqrt(252)
    st.write(f"Sharpe Ratio: {sharpe_ratio}")

def plot_stock_summary(stocks_affection_df):
    st.write("Stocks Price Change Summary")
    fig = px.bar(stocks_affection_df, x=stocks_affection_df.index, y=stocks_affection_df.values)
    st.plotly_chart(fig)

def run(symbol_benchmark, symbolsDate_dict, benchmark_df=None, stocks_df=None, events_df=None, def_days_before=6, def_days_after=0):
    if len(symbolsDate_dict['symbols']) < 1:
        st.info("Please select symbols.")
        st.stop()

    benchmark_dict = symbolsDate_dict.copy()
    benchmark_dict['symbols'] = [symbol_benchmark]

    benchmark_df = get_stocks(benchmark_dict, 'close') if benchmark_df is None else benchmark_df
    stocks_df = get_stocks(symbolsDate_dict, 'close') if stocks_df is None else stocks_df
    events_df = get_stocks_events(symbolsDate_dict, 'cashDividend') if events_df is None else events_df


    days_before = st.number_input('Days before event', min_value=-10, max_value=10, value=def_days_before)
    days_after = st.number_input('Days after event', min_value=0, max_value=30, value=def_days_after)

    event_affections = {}
    for stock in stocks_df.columns:
        if stock not in events_df.columns:
            continue
        
        stock_df = stocks_df[stock]
        event_df = pd.DataFrame(events_df[stock])
        
        event_df = event_df.dropna()
        
        event_affection = get_event_affection(stock_df, event_df, days_before, days_after)
        if not event_affection.empty:
            event_affections[stock] = event_affection

    events_affection_df = pd.DataFrame()
    
    for key in event_affections.keys():
        events_affection_df[key] = event_affections[key]
    
    is_numeric = type(events_df.iloc[0, 0]) in [int, float]
    
    events_affection_unstack_df = plot_event_distributions(events_df, events_affection_df, is_numeric=is_numeric)
    plot_event_summary(events_affection_unstack_df, benchmark_df, symbol_benchmark)
    
    # plot stock summary, plot stock that has the most event affect
    stocks_affection_df = events_affection_df.mean()
    
    plot_stock_summary(stocks_affection_df)
    
