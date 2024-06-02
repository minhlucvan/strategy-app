import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from time import sleep
import plotly.express as px
import plotly.graph_objects as go
from pytz import UTC  # Import the UTC time zone
from utils.vndirect import get_fund_ratios, load_fund_ratios_to_df

import requests

@st.cache_data
def fetch_data():
    data = get_fund_ratios()
    
    data_df = load_fund_ratios_to_df(data)
    
    return data_df

def run(symbol_benchmark, symbolsDate_dict):

    etf_data = fetch_data()

    # group by code and reportDate
    etf_data = etf_data.groupby(['code', 'reportDate']).sum().reset_index()

    # last report
    last_report = etf_data['reportDate'].max()
    etf_data_last_report = etf_data[etf_data['reportDate'] == last_report]

    # first report
    first_report = etf_data['reportDate'].min()
    etf_data_first_report = etf_data[etf_data['reportDate'] == first_report]

    # last etf structure
    etf_data_last_report = etf_data_last_report.sort_values(by=['value'], ascending=False)

    # last etf tickers
    etf_tickers_last_report = etf_data_last_report['code'].unique().tolist()

    # etf tickers
    all_etf_tickers = etf_data['code'].unique().tolist()

    # etf reports
    etf_report_dates = etf_data['reportDate'].sort_values().unique().tolist()

    # etf reports
    etf_reports = etf_data.groupby(['reportDate'])

    # plot etf structure
    fig = px.bar(etf_data_last_report, x='code', y='value', color='code')
    fig.update_layout(
        title="ETF Structure",
        xaxis_title="ETF",
        yaxis_title="Value",
        xaxis_rangeslider_visible=False
    )
    st.plotly_chart(fig, use_container_width=True)


    # loop through reports, calculate etf structure changes by ticker
    temp_report = {}

    for ticker in all_etf_tickers:
        temp_report[ticker] = {
            'value': 0,
            'value_change': 0,
            'value_change_pct': 0,
        }

    for report in etf_report_dates:
        etf_data_report = etf_reports.get_group(report)

        report_tickers = etf_data_report['code'].unique().tolist()

        for ticker in report_tickers:
            current_value = etf_data_report[etf_data_report['code'] == ticker]['value'].values[0]
            value_change = current_value - temp_report[ticker]['value']
            value_change_pct = value_change / temp_report[ticker]['value'] * 100 if temp_report[ticker]['value'] > 0 else 0

            if temp_report[ticker]['value'] == 0:
                value_change_pct = 100

            etf_data.loc[(etf_data['code'] == ticker) & (etf_data['reportDate'] == report), 'value_change'] = value_change
            etf_data.loc[(etf_data['code'] == ticker) & (etf_data['reportDate'] == report), 'value_change_pct'] = value_change_pct
            
            temp_report[ticker].update({
                'value': current_value,
                'value_change': value_change,
                'value_change_pct': value_change_pct,
            })

                
        

    # plot changes by ticker
    fig = px.bar(etf_data, x='reportDate', y='value_change_pct', color='code')
    fig.update_layout(
        title="ETF Structure Changes",
        xaxis_title="Date",
        yaxis_title="Value Change %",
        xaxis_rangeslider_visible=False
    )
    st.plotly_chart(fig, use_container_width=True)

    # plot etf positvie changes by ticker
    etf_data_positive_change = etf_data[etf_data['value_change_pct'] > 0]
    fig = px.bar(etf_data_positive_change, x='reportDate', y='value_change_pct', color='code')
    fig.update_layout(
        title="ETF Structure Positive Changes",
        xaxis_title="Date",
        yaxis_title="Value Change %",
        xaxis_rangeslider_visible=False
    )
    st.plotly_chart(fig, use_container_width=True)

    # plot newly added etf by ticker
    etf_data_newly_added = etf_data[etf_data['value_change_pct'] == 100]

    fig = px.bar(etf_data_newly_added, x='reportDate', y='value_change_pct', color='code')
    fig.update_layout(
        title="ETF Structure Newly Added",
        xaxis_title="Date",
        yaxis_title="Value Change %",
        xaxis_rangeslider_visible=False
    )
    st.plotly_chart(fig, use_container_width=True)

    # plot etf structure changes by ticker
    etf_ticker = st.selectbox('Select ETF', etf_tickers_last_report)

    if etf_ticker is None:
        st.stop()

    etf_data_ticker = etf_data[etf_data['code'] == etf_ticker]

    fig = px.bar(etf_data_ticker, x='reportDate', y='value', color='code')
    fig.update_layout(
        title=f"ETF Structure Changes - {etf_ticker}",
        xaxis_title="Date",
        yaxis_title="Value Change %",
        xaxis_rangeslider_visible=False
    )
    st.plotly_chart(fig, use_container_width=True)