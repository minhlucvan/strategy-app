from utils.processing import get_stocks, get_stocks_financial
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
import utils.plot_utils as pu
import matplotlib.pyplot as plt

def run(symbol_benchmark, symbolsDate_dict, n_clusters=5):
    if len(symbolsDate_dict['symbols']) < 1:
        st.info("Please select symbols.")
        st.stop()

    # Fetch stock and financial data
    stocks_df = get_stocks(symbolsDate_dict, 'close')
    stocks_df.index = pd.to_datetime(stocks_df.index).tz_localize(None)
    financials_df = get_stocks_financial(symbolsDate_dict, stack=True)

    # Extract fundamental investment factors
    roe_df = financials_df['roe']
    roa_df = financials_df['roa']
    eps_df = financials_df['earningPerShare']
    pe_df = financials_df['priceToEarning']
    pb_df = financials_df['priceToBook']

    # Compute investment features
    feature_df = pd.DataFrame({
        'PE': pe_df.mean(),
        'PB': pb_df.mean(),
        'ROE': roe_df.mean(),
        'ROA': roa_df.mean(),
    })

    # Drop NaNs
    feature_df.dropna(inplace=True)

    # Normalize data
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(feature_df)

    # Apply Hierarchical Clustering
    linkage_matrix = linkage(scaled_features, method='ward')
    feature_df['Cluster'] = fcluster(linkage_matrix, n_clusters, criterion='maxclust')

    # Plot Dendrogram
    plt.figure(figsize=(10, 5))
    dendrogram(linkage_matrix, labels=feature_df.index, leaf_rotation=90)
    plt.title("Stock Clustering Dendrogram")
    st.pyplot(plt)

    # plot each cluster
    for cluster in range(1, n_clusters + 1):
        st.write(f"### Cluster {cluster}")
        st.write(f"Stocks in Cluster {cluster}")
        st.write(f"{feature_df[feature_df['Cluster'] == cluster].index.tolist()}")
        
        # price plot
        pu.plot_multi_line(stocks_df[feature_df[feature_df['Cluster'] == cluster].index], title=f"Cluster {cluster} Price")