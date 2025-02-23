import streamlit as st
import pandas as pd
from utils.processing import get_stocks, get_stocks_financial, get_stocks_income_statement
from utils.plot_utils import plot_multi_line, plot_multi_bar
import utils.plot_utils as pu
import plotly.express as px

def normalize_score(series):
    """Normalize a series to a 0-100 scale using percentile ranking."""
    return (series.rank(pct=True) * 100).clip(0, 100)

# --- Data Fetching and Processing ---
def fetch_data(symbol_benchmark, symbolsDate_dict):
    """Fetch price, financial, and income statement data for given symbols."""
    if len(symbolsDate_dict['symbols']) < 1:
        st.info("Please select symbols (e.g., VNM, HPG, VCB for Vietnamese stocks).")
        st.stop()

    # Benchmark data (e.g., VNINDEX)
    benchmark_dict = symbolsDate_dict.copy()
    benchmark_dict['symbols'] = [symbol_benchmark]

    # Price data
    stocks_df = get_stocks(symbolsDate_dict, stack=True)
    stocks_df.index = pd.to_datetime(stocks_df.index).tz_localize(None)

    # Financial data
    financials_df = get_stocks_financial(symbolsDate_dict, stack=True)

    # Income statement data
    income_df = get_stocks_income_statement(symbolsDate_dict, stack=True)

    return stocks_df, financials_df, income_df

def compute_growth_factors(stocks_df, financials_df, income_df):
    """Compute individual growth factors from price, financial, and income data."""
    factors = {}
    
    # Income Statement Factors
    factors['revenue'] = income_df['revenue']
    
    
    liqudity_df = stocks_df['close'] * stocks_df['volume']
    factors['liquidity'] = liqudity_df
    
    market_liquidity = liqudity_df.sum(axis=1)
    factors['market_liquidity'] = pd.DataFrame(index=liqudity_df.index, data=market_liquidity.values, columns=['market_liquidity'])
    
    # smooth window 5
    factors['market_liquidity'] = factors['market_liquidity'].rolling(window=5).mean()
    
    
    liquidity_ratio = liqudity_df.div(market_liquidity, axis=0)
    factors['liquidity_ratio'] = liquidity_ratio
    factors['liquidity_ratio'] = factors['liquidity_ratio'].rolling(window=5).mean()
    
    MA_20 = liquidity_ratio.rolling(window=20).mean()
    SD_20 = liquidity_ratio.rolling(window=20).std()
    lower_channel = MA_20 - 0.8 * SD_20
    factors['lower_channel'] = lower_channel
        
    liquidity_growth_ratio = liquidity_ratio.pct_change()
    factors['liquidity_growth_ratio'] = liquidity_growth_ratio.fillna(0)
    
    liquidity_growth_ratio_acc = liquidity_growth_ratio.cumsum()
    factors['liquidity_growth_ratio_acc'] = liquidity_growth_ratio_acc.fillna(0)
    
    liqdity_growth = liqudity_df.pct_change()
    factors['liquidity_growth'] = liqdity_growth
        
    liquidity_growth_acc = liqdity_growth.cumsum()
    factors['liquidity_growth_acc'] = liquidity_growth_acc
    
    return factors

# --- Main Execution ---
def run(symbol_benchmark, symbolsDate_dict):
    """Run the full analysis."""
    # Fetch data
    stocks_df, financials_df, income_df = fetch_data(symbol_benchmark, symbolsDate_dict)

    factors = compute_growth_factors(stocks_df, financials_df, income_df)
    
    # liquidity_rank scatter
    selected_symbols = st.multiselect('Select symbols to highlight', symbolsDate_dict['symbols'])
        
    pu.plot_multi_line(stocks_df['close'][selected_symbols], title='Stocks Close')
    
    pu.plot_multi_line(factors['liquidity'][selected_symbols], title='Revenue')
        
    pu.plot_multi_line(factors['liquidity_ratio'][selected_symbols], title='Liquidity Ratio')
    
    if len(selected_symbols) == 1:
        # plot the lower channel and liquidity ratio
        symbol = selected_symbols[0]
        lower_channel = factors['lower_channel'][symbol]
        liquidity_ratio = factors['liquidity_ratio'][symbol]
        
        fig = px.line()
        fig.add_scatter(x=lower_channel.index, y=lower_channel.values, name='Lower Channel')
        fig.add_scatter(x=liquidity_ratio.index, y=liquidity_ratio.values, name='Liquidity Ratio')
        st.plotly_chart(fig)
        
        # where liquidity ratio is below lower channel
        liquidity_ratio_below_lower_channel = liquidity_ratio[liquidity_ratio < lower_channel]
        pu.plot_single_bar(liquidity_ratio_below_lower_channel, title='Liquidity Ratio below Lower Channel', x_title='Date', y_title='Liquidity Ratio', legend_title='Liquidity Ratio', price_df=stocks_df['close'][symbol])
        
    