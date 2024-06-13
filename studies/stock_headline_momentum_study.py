import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

from studies.stock_news_momentum_study import calculate_price_changes, filter_events, plot_correlation, plot_scatter_matrix, run_simulation
from utils.component import input_SymbolsDate
from utils.plot_utils import plot_events
from utils.processing import get_stocks, get_stocks_news
from studies.stock_custom_event_study import run as run_custom_event_study


def plot_wordcloud(news_df):
    import streamlit as st
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    
    display_df = news_df

    # Create some sample text
    text = ""
    
    for col in display_df.columns:
        for index in display_df.index:
            headline = display_df.loc[index, col]
            if pd.isna(headline):
                continue
            headline = headline.lower()
            
            # remove blacklisted words
            headline = headline.replace("thông báo", "").replace("cổ phiếu", "")
            
            text += headline + ". "

    # Create and generate a word cloud image:
    wordcloud = WordCloud().generate(text)

    # Display the generated image:
    fig = plt.figure(figsize=(10, 10))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    st.pyplot(fig)
    

def run(symbol_benchmark, symbolsDate_dict):

    if len(symbolsDate_dict['symbols']) < 1:
        st.info("Please select symbols.")
        st.stop()
        
    benchmark_dict = symbolsDate_dict.copy()
    benchmark_dict['symbols'] = [symbol_benchmark]
    
    benchmark_df = get_stocks(benchmark_dict, 'close')
    stocks_df = get_stocks(symbolsDate_dict, 'close')
    
    if benchmark_df.empty or stocks_df.empty:
        st.warning("No data available.")
        st.stop()
        
    new_types = ["-1", "1", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15"]
    news_type = st.selectbox("Select news type", new_types)
    
    news_df = get_stocks_news(symbolsDate_dict, 'title', channel_id=news_type)
    
    # Localize index to None
    for df in [news_df, stocks_df, benchmark_df]:
        df.index = df.index.tz_localize(None)
        
    price_changes_flat_df = calculate_price_changes(stocks_df, news_df)
    
    st.write("Select the column and threshold to filter the news. negative column means you are looking into the future")
    column = st.selectbox("Select column", price_changes_flat_df.columns, index=1)
    threshold = st.number_input('Threshold', min_value=0.0, max_value=5.0, value=0.0)
    word_filter = st.text_input("Word filter", "")
    
    # look into future, filter out the news that profitable
    news_df, original_news_df  = filter_events(news_df, price_changes_flat_df, threshold=threshold, column=column, text_filter=word_filter)
    
    show_data = st.checkbox("Show data")
    if show_data:
        display_df = original_news_df[original_news_df.notnull().any(axis=1)]
        st.dataframe(display_df, use_container_width=True)
        
    plot_wordcloud(original_news_df)
    
    if len(news_df.columns) == 1:
        plot_events(stocks_df.iloc[:, 0], original_news_df.iloc[:, 0], label="")
        
    plot_correlation(price_changes_flat_df)
    plot_scatter_matrix(price_changes_flat_df)
    
    enable_simulate = st.checkbox("Enable simulate")
    if enable_simulate:
        run_simulation(symbol_benchmark, symbolsDate_dict, benchmark_df, stocks_df, news_df)

