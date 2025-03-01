from utils.processing import get_stocks
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from sklearn.preprocessing import StandardScaler
import utils.plot_utils as pu

def run(symbol_benchmark, symbolsDate_dict):
    if len(symbolsDate_dict['symbols']) < 1:
        st.info("Please select symbols.")
        st.stop()

    n_clusters = st.slider("Number of Clusters", 2, 20, 5)

    # Fetch historical stock prices
    stocks_df = get_stocks(symbolsDate_dict, 'close')
    stocks_df.index = pd.to_datetime(stocks_df.index).tz_localize(None)

    # Calculate log returns to remove price scale effect
    log_returns = np.log(stocks_df / stocks_df.shift(1)).dropna()

    # Compute correlation matrix
    correlation_matrix = log_returns.corr()

    # Convert correlation to distance (1 - correlation)
    distance_matrix = 1 - correlation_matrix

    # Apply Hierarchical Clustering
    linkage_matrix = linkage(distance_matrix, method='ward')
    clusters = fcluster(linkage_matrix, n_clusters, criterion='maxclust')

    # Store cluster assignments
    cluster_df = pd.DataFrame({'Stock': correlation_matrix.index, 'Cluster': clusters})
    cluster_df.set_index('Stock', inplace=True)

    # Plot Correlation Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=False, cmap="coolwarm", linewidths=0.5)
    plt.title("Stock Price Correlation Heatmap")
    st.pyplot(plt)

    # Plot Dendrogram
    plt.figure(figsize=(10, 5))
    dendrogram(linkage_matrix, labels=correlation_matrix.index, leaf_rotation=90)
    plt.title("Stock Price Clustering Dendrogram")
    st.pyplot(plt)

    # Display Clustered Stocks
    st.write("### Clustered Stocks by Price Movement")

    # plot price movement for each cluster
    for cluster in range(1, n_clusters + 1):
        st.write(f"### Cluster {cluster}")
        st.write(f"Stocks in Cluster {cluster}")
        st.write(f"{cluster_df[cluster_df['Cluster'] == cluster].index.tolist()}")

        # price plot
        stocks_in_cluster = cluster_df[cluster_df['Cluster'] == cluster].index
        pu.plot_multi_line(stocks_df[stocks_in_cluster], title=f"Cluster {cluster} Price")