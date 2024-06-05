import pandas as pd
import numpy as np
import json

import streamlit as st
import vectorbt as vbt

import plotly.express as px
import plotly.graph_objs as go
import plotly.subplots as sp


from utils.component import  check_password, input_dates, input_SymbolsDate
import matplotlib.pyplot as plt

from utils.processing import get_stocks, get_stocks_financial
from studies.rrg import plot_RRG, rs_ratio, RRG_Strategy
from utils.vbt import plot_pf
from vbt_strategy.MOM_D import get_MomDInd

from studies.market_wide import MarketWide_Strategy

def plot_multi_line(df, title, x_title, y_title, legend_title):
    fig = go.Figure()
    for col in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df[col], mode='lines', name=col))
    fig.update_layout(title=title, xaxis_title=x_title, yaxis_title=y_title, legend_title=legend_title)
    st.plotly_chart(fig, use_container_width=True)

def plot_snapshot(df, title, x_title, y_title, legend_title):
    # plot bar chart fe each stock
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df.columns, y=df.iloc[-1], name='PEB Ratio'))
    fig.update_layout(title=title, xaxis_title=x_title, yaxis_title=y_title, legend_title=legend_title)
    st.plotly_chart(fig, use_container_width=True)

def plot_snapshot_comparison(snapshot_df, title, x_title, y_title, legend_title):
    # Create a subplot with a row for each metric
    metrics = snapshot_df.index
    fig = sp.make_subplots(rows=len(metrics), cols=1, shared_xaxes=False, vertical_spacing=0.01)

    # Add each metric's stacked bar chart to the subplot
    for i, metric in enumerate(metrics, start=1):
        for stock in snapshot_df.columns:
            stock_index = snapshot_df.columns.get_loc(stock)
            fig.add_trace(
                go.Bar(
                    y=[metric],
                    x=[snapshot_df.loc[metric, stock]],
                    name=stock,
                    orientation='h',
                    # color by stock name
                    marker=dict(
                        color=px.colors.qualitative.Plotly[stock_index % len(px.colors.qualitative.Plotly)]
                    )
                ),
                row=i, col=1
            )

    # Update layout
    fig.update_layout(
        title='Snapshot of Raw Values by Metric and Stock',
        xaxis_title='',
        yaxis_title='Metric',
        barmode='stack',
        legend_title='Stock',
        height=140 * len(metrics)  # Adjust the height based on the number of metrics
    )

    # Adjust x-axis titles for each subplot
    for i in range(1, len(metrics) + 1):
        # fig.update_xaxes(title_text='Value', row=i, col=1)
        fig.update_xaxes(showticklabels=False, row=i, col=1)

    names = set()
    fig.for_each_trace(
        lambda trace:
            trace.update(showlegend=False)
            if (trace.name in names) else names.add(trace.name))

    # Display the plot
    st.plotly_chart(fig, use_container_width=True)

def plot_contrbution(df, title, x_title, y_title, legend_title):
    df_sorted = df.sort_values(ascending=False)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df_sorted.index, y=df_sorted, name='Contribution'))
    fig.update_layout(title=title, xaxis_title=x_title, yaxis_title=y_title, legend_title=legend_title)
    st.plotly_chart(fig, use_container_width=True)

@st.cache_data
def calculate_market_metrics(union_df):
    market_data_list = []
    
    # loop through each index
    for index in union_df.index:
        market_data = {}
        market_data['date'] = index
        for metric in union_df.columns.get_level_values(0).unique():
            market_data[metric] = union_df.loc[index, metric].mean()
        market_data_list.append(market_data)

    market_data_df = pd.DataFrame(market_data_list)
    
    # set date as index
    market_data_df = market_data_df.set_index('date')
    
    return market_data_df

# calculate the ratio metrics = stock / market
def calculate_raitio_metrics(union_df, market_df):
    ratios_df = union_df.copy()
    
    # set full nan
    ratios_df.loc[:, :] = np.nan
    
    for metric in union_df.columns.get_level_values(0).unique():
        for stock in union_df.columns.get_level_values(1).unique():
            for index in union_df.index:
                if market_df.loc[index, metric] != 0 and not pd.isna(union_df.loc[index, (metric, stock)]):
                    ratios_df.loc[index, (metric, stock)] = union_df.loc[index, (metric, stock)] / market_df.loc[index, metric]
    
    return ratios_df
    
def run(symbol_benchmark, symbolsDate_dict):
    
    with st.expander("Magic Formula Study"):
        st.markdown("""Magic Formula is a value investing strategy that selects stocks based on a combination of two factors:
                       
- Earnings Yield (E/P)
- Return on Capital (ROIC)

The strategy ranks stocks based on these two factors and selects the top stocks for investment.
""")
        
    symbolsDate_dict['symbols'] =  symbolsDate_dict['symbols']
    
    is_multi = len(symbolsDate_dict['symbols']) > 1
    
    if len(symbolsDate_dict['symbols']) < 1:
        st.info("Please select symbols.")
        st.stop()
    
    # copy the symbolsDate_dict
    benchmark_dict = symbolsDate_dict.copy()
    benchmark_dict['symbols'] = [symbol_benchmark]
    
    # Assuming get_stocks and get_stocks_funamental are defined elsewhere
    stocks_df = get_stocks(symbolsDate_dict, stack=True)
    financial_df = get_stocks_financial(symbolsDate_dict, stack=True)
            
    # Ensure both indices are timezone-naive
    stocks_df.index = pd.to_datetime(stocks_df.index).tz_localize(None)
    financial_df.index = pd.to_datetime(financial_df.index).tz_localize(None)
    
    # filter the stocks_df and financial_df to the same date range
    start_date = pd.to_datetime(symbolsDate_dict['start_date']).tz_localize(None)
    
    # Create a union of the indices
    union_index = stocks_df.index.union(financial_df.index)
    
    # Reindex both DataFrames to the union index
    reindexed_stocks_df = stocks_df.reindex(union_index)
    reindexed_fundamental_df = financial_df.reindex(union_index)
    
    union_df = pd.concat([reindexed_stocks_df, reindexed_fundamental_df], axis=1)
    # close,priceToEarning,priceToBook,valueBeforeEbitda,dividend,roe,roa,daysReceivable,daysInventory,daysPayable,ebitOnInterest,earningPerShare,bookValuePerShare,interestMargin,nonInterestOnToi,badDebtPercentage,provisionOnBadDebt,costOfFinancing,equityOnTotalAsset,equityOnLoan,costToIncome,equityOnLiability,currentPayment,quickPayment,epsChange,ebitdaOnStock,grossProfitMargin,operatingProfitMargin,postTaxMargin,debtOnEquity,debtOnAsset,debtOnEbitda,shortOnLongDebt,assetOnEquity,capitalBalance,cashOnEquity,cashOnCapitalize,cashCirculation,revenueOnWorkCapital,capexOnFixedAsset,revenueOnAsset,postTaxOnPreTax,ebitOnRevenue,preTaxOnEbit,preProvisionOnToi,postTaxOnToi,loanOnEarnAsset,loanOnAsset,loanOnDeposit,depositOnEarnAsset,badDebtOnAsset,liquidityOnLiability,payableOnEquity,cancelDebt,ebitdaOnStockChange,bookValuePerShareChange,creditGrowth
    
    # filter date > start_date
    union_df = union_df.loc[start_date:]

    market_df = calculate_market_metrics(union_df)
    
    ratio_df = calculate_raitio_metrics(union_df, market_df)
    
    # fill missing values with the last available value
    union_df = union_df.fillna(method='ffill')
    market_df = market_df.fillna(method='ffill')
    ratio_df = ratio_df.fillna(method='ffill')
    
    # index (code, factor)
    metrics = union_df.columns.get_level_values(0).unique()
    
    price_df = union_df['close']
    
    st.write("## Stock Prices")
    plot_multi_line(price_df, 'Stock Prices', 'Date', 'Price', 'Stocks')
    
    # drop open, high, low
    metrics = metrics.drop(['close', 'open', 'high', 'low'])
    default_metrics = ['priceToEarning', 'priceToBook']

    selected_metrics = st.multiselect('Select Metrics to Plot', metrics, default_metrics)
    
    for metric in selected_metrics:
        plot_multi_line(union_df[metric], f'{metric} comparsion', 'Date', metric, 'Stocks')
        plot_multi_line(ratio_df[metric], f'{metric} Ratio comparsion', 'Date', metric, 'Stocks')
        
    # plot snapshot
    snapshot_df = union_df[selected_metrics].iloc[-1]
    # reset multi index to single index
    # (metric, stock) -> stock | metric1 | metric2 | metric3
    snapshot_df = snapshot_df.reset_index()
    snapshot_df.columns = ['metric', 'stock', 'value']

    # convert metric to column
    snapshot_df = snapshot_df.pivot(index='metric', columns='stock', values='value')
    # snapshot_df = metric | stock1 | stock2 | stock3

    # set index to stock
    snapshot_df = snapshot_df.reset_index()
    snapshot_df = snapshot_df.set_index('metric')
    
    # calculate the snapshot contribution = snapshot / sum by row
    snapshot_contrib_df = snapshot_df.copy()
    
    # calculate the sum by row
    snapshot_sum = snapshot_contrib_df.sum(axis=1)
    
    # calculate the contribution
    snapshot_contrib_df = snapshot_contrib_df.div(snapshot_sum, axis=0)
    
    # add new metric 'contribution' = sum by column
    snapshot_contrib_df.loc['contribution'] = snapshot_contrib_df.sum()
    
    plot_snapshot_comparison(snapshot_df, 'Snapshot of Raw Values by Metric and Stock', 'Value', 'Metric', 'Stock')
    
    plot_contrbution(snapshot_contrib_df.loc['contribution'], 'Snapshot of Contribution by Stock', 'Contribution', 'Stock', 'Contribution')