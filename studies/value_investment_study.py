import pandas as pd
import numpy as np
import json

import streamlit as st
import vectorbt as vbt

import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots # creating subplots


from utils.component import  check_password, input_dates, input_SymbolsDate
import matplotlib.pyplot as plt

from utils.processing import get_stocks, get_stocks_valuation
from studies.rrg import plot_RRG, rs_ratio, RRG_Strategy
from utils.vbt import plot_pf
from vbt_strategy.MOM_D import get_MomDInd

from studies.market_wide import MarketWide_Strategy

def plot_evaluation_pe(price_df, evaluation_df, industry_df):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02)
    
    for symbol in price_df.columns:
        fig.add_trace(go.Scatter(x=price_df.index, y=price_df[symbol], mode='lines', name=f'{symbol} Price'), row=1, col=1)
    
    for symbol in evaluation_df.columns:
        fig.add_trace(go.Scatter(x=evaluation_df.index, y=evaluation_df[symbol], mode='lines', name=f'{symbol} PE'), row=2, col=1)
    
    for symbol in industry_df.columns:
        fig.add_trace(go.Scatter(x=industry_df.index, y=industry_df[symbol], mode='lines', name=f'{symbol} Industry PE'), row=2, col=1)
    
    st.plotly_chart(fig)
    
def plot_evaluation_pb(price_df, evaluation_df, industry_df):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02)
    
    for symbol in price_df.columns:
        fig.add_trace(go.Scatter(x=price_df.index, y=price_df[symbol], mode='lines', name=f'{symbol} Price'), row=1, col=1)
        
    for symbol in evaluation_df.columns:
        fig.add_trace(go.Scatter(x=evaluation_df.index, y=evaluation_df[symbol], mode='lines', name=f'{symbol} PB'), row=2, col=1)
        
    for symbol in industry_df.columns:
        fig.add_trace(go.Scatter(x=industry_df.index, y=industry_df[symbol], mode='lines', name=f'{symbol} Industry PB'), row=2, col=1)
    
    st.plotly_chart(fig)

def plot_evaluation_pe_ratio(price_df, evaluation_df):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02)
    
    for symbol in price_df.columns:
        fig.add_trace(go.Scatter(x=price_df.index, y=price_df[symbol], mode='lines', name=symbol), row=1, col=1)
    
    for symbol in evaluation_df.columns:
        fig.add_trace(go.Scatter(x=evaluation_df.index, y=evaluation_df[symbol], mode='lines', name=symbol), row=2, col=1)
    
    st.plotly_chart(fig)
    
def plot_evaluation_pb_ratio(price_df, evaluation_df):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02)
    
    for symbol in price_df.columns:
        fig.add_trace(go.Scatter(x=price_df.index, y=price_df[symbol], mode='lines', name=symbol), row=1, col=1)
    
    for symbol in evaluation_df.columns:
        fig.add_trace(go.Scatter(x=evaluation_df.index, y=evaluation_df[symbol], mode='lines', name=symbol), row=2, col=1)
    
    st.plotly_chart(fig)

def plot_pe_comparison(pe_df, industry_pe_df):
    # plot bar chart fe each stock
    fig = make_subplots(rows=2, cols=1)
    fig.add_trace(go.Bar(x=pe_df.columns, y=pe_df.iloc[-1], name='PE'), row=1, col=1)
    fig.add_trace(go.Bar(x=industry_pe_df.columns, y=industry_pe_df.iloc[-1], name='Industry PE'), row=2, col=1)
    st.plotly_chart(fig)
    
def plot_pb_comparison(pb_df, industry_pb_df):
    # plot bar chart fe each stock
    fig = make_subplots(rows=2, cols=1)
    fig.add_trace(go.Bar(x=pb_df.columns, y=pb_df.iloc[-1], name='PB'), row=1, col=1)
    fig.add_trace(go.Bar(x=industry_pb_df.columns, y=industry_pb_df.iloc[-1], name='Industry PB'), row=2, col=1)
    st.plotly_chart(fig)
    
def plot_pe_ratio_comparison(pe_ratio_df):
    # plot bar chart fe each stock
    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Bar(x=pe_ratio_df.columns, y=pe_ratio_df.iloc[-1], name='PE Ratio'))
    st.plotly_chart(fig)
    
def plot_pb_ratio_comparison(pb_ratio_df):
    # plot bar chart fe each stock
    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Bar(x=pb_ratio_df.columns, y=pb_ratio_df.iloc[-1], name='PB Ratio'))
    st.plotly_chart(fig)
    
def plot_peb_ratio_comparison(peb_ratio_df):
    # plot bar chart fe each stock
    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Bar(x=peb_ratio_df.columns, y=peb_ratio_df.iloc[-1], name='PEB Ratio'))
    st.plotly_chart(fig)
    
def run(symbol_benchmark, symbolsDate_dict):
        
    symbolsDate_dict['symbols'] =  symbolsDate_dict['symbols']
    # copy the symbolsDate_dict
    benchmark_dict = symbolsDate_dict.copy()
    benchmark_dict['symbols'] = [symbol_benchmark]
    
    benchmark_df = get_stocks(benchmark_dict,'close')
    stocks_df = get_stocks(symbolsDate_dict,'close')
   
    pe_df = get_stocks_valuation(symbolsDate_dict, indicator='pe')
    pb_df = get_stocks_valuation(symbolsDate_dict, indicator='pb')
    industry_pe_df = get_stocks_valuation(symbolsDate_dict, indicator='industryPe')
    industry_pb_df = get_stocks_valuation(symbolsDate_dict, indicator='industryPb')
    pe_ratio_df = pe_df.div(industry_pe_df)
    pb_ratio_df = pb_df.div(industry_pb_df)
    
    # mean pe and pb
    peb_ratio_df = (pe_df + pb_df) / 2
    
    st.write(f"### PE")
    plot_evaluation_pe(stocks_df, pe_df, industry_pe_df)
    
    st.write(f"### PE Ratio")
    # plot pb ratio
    plot_evaluation_pe_ratio(stocks_df, pb_ratio_df)
    
    st.write(f"### PB")
    plot_evaluation_pb(stocks_df, pb_df, industry_pb_df)
    
    st.write(f"### PB Ratio")
    # plot pe ratio
    plot_evaluation_pb_ratio(stocks_df, pe_ratio_df)
    
    
    # snapshot analysis
    st.write(f"### Snapshot Analysis")
        
    st.write(f"### PE Comparison")
    plot_pe_comparison(pe_df, industry_pe_df)
    
    st.write(f"### PB Comparison")
    plot_pb_comparison(pb_df, industry_pb_df)
    
    st.write(f"### PE Ratio Comparison")
    plot_pe_ratio_comparison(pe_ratio_df)
    
    st.write('### Top 10 PE Ratio')
    minimal_pe_ratio = pe_ratio_df.iloc[-1].replace([np.inf, -np.inf], np.nan).dropna().sort_values(ascending=True).head(10)
    # set column as row
    # minimal_pe_ratio = minimal_pe_ratio.reset_index()
    st.write(minimal_pe_ratio)
    
    st.write(f"### PB Ratio Comparison")
    plot_pb_ratio_comparison(pb_ratio_df)
    
    st.write('### Top 10 PB Ratio')
    minimal_pb_ratio = pb_ratio_df.iloc[-1].replace([np.inf, -np.inf], np.nan).dropna().sort_values(ascending=True).head(10)
    # set row as column
    # minimal_pb_ratio = minimal_pb_ratio.reset_index()
    st.write(minimal_pb_ratio)
    


    st.write(f"### PEB Ratio Comparison")
    plot_peb_ratio_comparison(peb_ratio_df)
    
    st.write('### Top 10 PEB Ratio')
    minimal_peb_ratio = peb_ratio_df.iloc[-1].replace([np.inf, -np.inf], np.nan).dropna().sort_values(ascending=True).head(10)
    # set row as column
    minimal_peb_ratio = minimal_peb_ratio.reset_index()
    st.write(minimal_peb_ratio)
    