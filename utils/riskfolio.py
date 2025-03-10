import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.cm
from datetime import datetime
import riskfolio as rp
import vectorbt as vbt

from utils.processing import AKData
from utils.vbt import get_pfByWeight,  plot_pf

import matplotlib.pyplot as plt
from datetime import datetime
import riskfolio as rp  # Assuming these functions are from the Riskfolio-Lib

def report(
    returns,
    w,
    rm="MV",
    rf=0,
    alpha=0.05,
    others=0.05,
    nrow=25,
    height=6,
    width=14,
    t_factor=252,
    ini_days=1,
    days_per_year=252,
    bins=50,
):
    cov = returns.cov()
    nav = returns.cumsum()

    # Create a figure with a specific layout
    fig, ax = plt.subplots(
        nrows=6,
        figsize=(width, height * 6),
        gridspec_kw={"height_ratios": [2, 1, 1.5, 1, 1, 1]},
    )

    # Generate table plot
    rp.plot_table(
        returns,
        w,
        MAR=rf,
        alpha=alpha,
        t_factor=t_factor,
        ini_days=ini_days,
        days_per_year=days_per_year,
        ax=ax[0],
    )

    # Generate drawdown plots
    st.write(returns)
    # rp.plot_drawdown(returns=returns, w=w, ax=[ax[1], ax[5]])

    # Generate pie chart for portfolio composition
    rp.plot_pie(
        w=w,
        title="Portfolio Composition",
        others=others,
        nrow=nrow,
        cmap="tab20",
        ax=ax[2],
    )

    # Generate risk contribution plot
    rp.plot_risk_con(
        w=w,
        cov=cov,
        returns=returns,
        rm=rm,
        rf=rf,
        alpha=alpha,
        t_factor=t_factor,
        ax=ax[3],
    )

    # Generate histogram plot
    rp.plot_hist(returns=returns, w=w, alpha=alpha, bins=bins, ax=ax[4])

    # Adding titles and subtitles
    year = str(datetime.now().year)
    title = "Riskfolio-Lib Report"
    subtitle = "Copyright (c) 2020-" + year + ", Dany Cajas. All rights reserved."

    fig.suptitle(title, fontsize="xx-large", y=1.011, fontweight="bold")
    ax[0].set_title(subtitle, fontsize="large", ha="center", pad=10)

    # Display the plot
    st.pyplot(fig)

# Example usage:
# report(returns, w)


def get_pfOpMS(stocks_df, rm="MV", show_report=False, return_w=False, plot=True):
    '''
    calculate portfolio Optimized max sharpe ratio
    '''
    pct_df = pd.DataFrame()
    
    for symbol in stocks_df.columns:
        pct_df[symbol] = stocks_df[symbol].pct_change().dropna()
        
    port = rp.Portfolio(returns=pct_df)
    method_mu='hist'
    method_cov='hist'
    port.assets_stats(method_mu=method_mu, method_cov=method_cov)
    # rm = 'MV' # Risk measure used, this time will be variance
    model="Classic"
    obj = 'Sharpe' # Objective function, could be MinRisk, MaxRet, Utility or Sharpe
    hist = True # Use historical scenarios for risk measures that depend on scenarios
    rf = 0 # Risk free rate
    l = 0 # Risk aversion factor, only useful when obj is 'Utility'
    weights_df = port.optimization(model=model, rm=rm, obj=obj, rf=rf, l=l, hist=hist)

    if show_report:
        report(stocks_df, weights_df, rm='MV', rf=0, alpha=0.05, height=6, width=14, others=0.05, nrow=25)   
    # Calculate returns of portfolio with optimized weights
    pfs_df=stocks_df.copy()
    pfs_df['Optimized Portfolio'] = 0
    i = 0
    
    if weights_df is None:
        st.error("No Optimzied max sharpe portfolio solution.")
        return
    for symbol, row in weights_df.iterrows():
        pfs_df['Optimized Portfolio'] += stocks_df[symbol].pct_change() * row["weights"] * 100
        i+=1
	
 # Display everything on Streamlit
    weights_df['Ticker'] = weights_df.index
    if plot:
        fig = px.pie(weights_df.iloc[0:10], values='weights', names='Ticker', title='Optimized Max Shape Portfolio Weights')
        st.plotly_chart(fig)
    
    # Plot Optimized Portfolio
    pf = get_pfByWeight(stocks_df, weights_df['weights'].values)
    if return_w:
        return pf, weights_df
    
    return pf

def FactorExposure(main_df:pd.DataFrame, factors_df:pd.DataFrame)-> pd.DataFrame:
    '''
        因子ETF代码，这是美国市场的因子ETF基金，使用基金的收益作为因子收益
        reference:
            https://www.sohu.com/a/521169688_505915
    '''
    #计算因子暴露：
    step = 'Forward'
    df = pd.concat([main_df, factors_df], axis=1).dropna()
    X = df[factors_df.columns].pct_change().dropna()
    Y = df[main_df.columns].pct_change().dropna()
    try:
        loadings = rp.loadings_matrix(X=X, Y=Y, stepwise=step)
    except Exception as e:
        print(f"FactorExposure Error:    {e}.")
        loadings = pd.DataFrame()
    return loadings

def plot_AssetsClusters(stocks_df):
    # Plotting Assets Clusters
    Y = stocks_df.pct_change().dropna()
    if Y.empty or len(Y) < 2 or Y.shape[1] < 2:
        st.error("Not enough data to plot clusters.")
    else:
        fig, ax = plt.subplots()
        ax = rp.plot_clusters(returns=Y,
                          codependence='pearson',
                          linkage='ward',
                          k=None,
                          max_k=10,
                          leaf_order=True,
                          dendrogram=True,
                          #linecolor='tab:purple',
                          ax=ax)
        st.pyplot(fig)
    