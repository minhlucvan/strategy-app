import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from time import sleep
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pytz import UTC  # Import the UTC time zone
from utils.processing import get_stocks

import requests

def get_fund_ratios():
    url = 'https://finfo-api.vndirect.com.vn/v4/fund_ratios?q=code:FUEKIV30,FUEMAV30,FUESSVFL,FUEVFVND,VNMETF,00885,2804,FUEIP100,E1VFVN30,FTSE~ratioCode:FUND_NAV_CR&sort=reportDate:desc&size=1000'
    
    headers = {
        'Accept': '*/*',
        'Accept-Language': 'en-US,en;q=0.9,vi;q=0.8,vi-VN;q=0.7,fr-FR;q=0.6,fr;q=0.5,de;q=0.4',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
        'Content-Type': 'application/json',
        'DNT': '1',
        'Origin': 'https://dstock.vndirect.com.vn',
        'Pragma': 'no-cache',
        'Referer': 'https://dstock.vndirect.com.vn/',
        'Sec-Fetch-Dest': 'empty',
        'Sec-Fetch-Mode': 'cors',
        'Sec-Fetch-Site': 'same-site',
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
        'sec-ch-ua': '"Google Chrome";v="119", "Chromium";v="119", "Not?A_Brand";v="24"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"macOS"'
    }
    
    response = requests.get(url, headers=headers, timeout=5)
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}")
        return None

def get_fund_data():
    url = 'https://finfo-api.vndirect.com.vn/v4/fund_ratios'
    headers = {
        'Accept': '*/*',
        'Accept-Language': 'en-US,en;q=0.9,vi;q=0.8,vi-VN;q=0.7,fr-FR;q=0.6,fr;q=0.5,de;q=0.4',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
        'Content-Type': 'application/json',
        'DNT': '1',
        'Origin': 'https://dstock.vndirect.com.vn',
        'Pragma': 'no-cache',
        'Referer': 'https://dstock.vndirect.com.vn/',
        'Sec-Fetch-Dest': 'empty',
        'Sec-Fetch-Mode': 'cors',
        'Sec-Fetch-Site': 'same-site',
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
        'sec-ch-ua': '"Google Chrome";v="119", "Chromium";v="119", "Not?A_Brand";v="24"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"macOS"',
    }

    params = {
        'q': 'ratioCode:WEEKLY_FUND_FLOW_SUM_CR,WEEKLY_ACCUM_FUND_FLOW_SUM_CR,WEEKLY_ACCUM_FUND_FLOW_SUM_CR_YD,WEEKLY_ACCUM_FUND_FLOW_SUM_CR_1Y,WEEKLY_ACCUM_FUND_FLOW_SUM_CR_2Y,MONTHLY_FUND_FLOW_SUM_CR,MONTHLY_ACCUM_FUND_FLOW_SUM_CR_YD,MONTHLY_ACCUM_FUND_FLOW_SUM_CR_1Y,MONTHLY_ACCUM_FUND_FLOW_SUM_CR_2Y~reportDate:gte:2020-01-01',
        'size': '10000',
    }

    response = requests.get(url, headers=headers, params=params)

    if response.status_code == 200:
        data = response.json()
        return data
    else:
        print(f"Error: {response.status_code}")
        return None


@st.cache_data
def fetch_data():
    data = get_fund_ratios()
    # """
    # {
    # "data": [
    #     {
    #         "code": "ILB",
    #         "type": "IFC",
    #         "period": "1M",
    #         "ratioCode": "IFC_HOLDING_COUNT_CR",
    #         "reportDate": "2022-09-30",
    #         "value": 1.0
    #     },
    # ]
    # }
    # """
    data = data['data']
    df = pd.DataFrame(data)
    df['reportDate'] = pd.to_datetime(df['reportDate'])
    df = df.sort_values(by=['reportDate'])

    data = get_fund_data()
    data = data['data']
    df2 = pd.DataFrame(data)
    df2['reportDate'] = pd.to_datetime(df2['reportDate'])
    df2 = df2.sort_values(by=['reportDate'])


    return df, df2

def run(symbol_benchmark, symbolsDate_dict):
    df, df2 = fetch_data()

    st.write("### ETFs NAV")
    st.dataframe(df.sort_values(by=['value'], ascending=False), use_container_width=True)

    fig = px.bar(df, x="code", y="value", color="code", barmode="group")
    st.plotly_chart(fig, use_container_width=True)

    # Etf flow
    st.write("### ETFs Flow")
    # st.dataframe(df2)

    weekly_fund_flow_sum_cr = df2[df2['ratioCode'] == 'WEEKLY_FUND_FLOW_SUM_CR']
    weekly_accum_fund_flow_sum_cr = df2[df2['ratioCode'] == 'WEEKLY_ACCUM_FUND_FLOW_SUM_CR_2Y']

    symbolsDate_dict['symbols'] = ['VN30']
    benchmark_df = get_stocks(symbolsDate_dict, 'close')
    
    # plot ratioCode WEEKLY_FUND_FLOW_SUM_CR, and WEEKLY_ACCUM_FUND_FLOW_SUM_CR 
    # ratioCode WEEKLY_FUND_FLOW_SUM_CR as bar chart by reportDate
    # ratioCode WEEKLY_ACCUM_FUND_FLOW_SUM_CR as line chart by reportDate
    fig = make_subplots(rows=2, cols=1, row_heights=[0.7, 0.3])
    fig.add_trace(go.Bar(x=weekly_fund_flow_sum_cr['reportDate'], y=weekly_fund_flow_sum_cr['value'], name='SUM'), row=1, col=1)
    fig.add_trace(go.Scatter(x=weekly_accum_fund_flow_sum_cr['reportDate'], y=weekly_accum_fund_flow_sum_cr['value'], mode='lines', name='ACCUM'), row=1, col=1)
    # add benchmark
    fig.add_trace(go.Scatter(x=benchmark_df.index, y=benchmark_df['VN30'], mode='lines', name='VN30'), row=2, col=1)

    # Customize the chart layout
    fig.update_layout(
        title="ETFs Flow",
        xaxis_title="Date",
        yaxis_title="Value",
        xaxis_rangeslider_visible=False
    )

    # Display the chart
    st.plotly_chart(fig, use_container_width=True)
