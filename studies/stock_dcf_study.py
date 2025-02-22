import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objs as go
import utils.plot_utils as pu
from plotly.subplots import make_subplots
from utils.stock_utils import get_stock_evaluation_snapshot, load_stock_evaluation_snapshot_to_dataframe, get_intraday_snapshots, load_intraday_snapshots_to_dataframe

def calculate_free_cash_flow(cash_flow):
    return cash_flow['freeCashFlow']

def calculate_dcf_valuation(cash_flow_df, discount_rate=0.1, growth_rate=0.02, projection_years=5):
    """Perform DCF valuation based on historical cash flow data.
    
    Args:
        cash_flow_df: DataFrame containing projected free cash flows for the next 5 years columns: ['Year', 'freeCashFlow']
        discount_rate: Discount rate for future cash flows (default: 10%)
        growth_rate: Perpetual growth rate for terminal value (default: 2%)
        projection_years: Number of years to project future cash flows (default: 5)
    
    Returns:
        float: The DCF valuation
    """
    if cash_flow_df.empty or cash_flow_df is None:
        print("Error retrieving cash flow data")
        return None
   
    # Get the most recent FCF as base
    last_fcf = cash_flow_df['freeCashFlow'].iloc[-1]
    last_date = pd.to_datetime(cash_flow_df.index[-1])
    
    # Project future cash flows
    future_dates = [last_date + pd.DateOffset(years=i) for i in range(1, projection_years + 1)]
    future_fcfs = [last_fcf * (1 + growth_rate)**i for i in range(1, projection_years + 1)]
    
    # Calculate present value of projected cash flows
    present_values = [fcf / (1 + discount_rate)**i for i, fcf in enumerate(future_fcfs, 1)]
    
    # Calculate terminal value
    terminal_fcf = future_fcfs[-1] * (1 + growth_rate)
    terminal_value = terminal_fcf / (discount_rate - growth_rate)
    pv_terminal = terminal_value / (1 + discount_rate)**projection_years
    
    # Total valuation
    total_pv = sum(present_values) + pv_terminal
    
    return total_pv
   

def plot_stock_evaluation_snapshot(ticker_data_dict):
    all_main_dfs = []
    all_top5_dfs = []
    all_cashflow_df = []
    ratios_df = pd.DataFrame()
    
    for ticker, data in ticker_data_dict.items():
        main_df, top5_df, cashflow_df = load_stock_evaluation_snapshot_to_dataframe(data)
                
        # Get intraday price snapshots for top 5 + current ticker
        top_6_tickers = top5_df['ticker'].tolist() + [ticker]
        price_snapshot = get_intraday_snapshots(tickers=top_6_tickers)
        price_snapshot_df = load_intraday_snapshots_to_dataframe(price_snapshot)
        prices_df = price_snapshot_df['price']
        
        # forwards fill missing prices or 0 prices
        prices_df = prices_df.ffill().bfill()
        last_prices = prices_df.iloc[-1]
        
        last_prices_df = pd.DataFrame(last_prices)
        last_prices_df.columns = ['price']
        last_prices_df.index.name = 'ticker'        
                
        main_last_price = last_prices_df.loc[ticker]['price']
        main_df['price'] = main_last_price
        main_df.index = [ticker]  # Set ticker as index

        # estimated price dcfs
        main_df['estimatedPriceDcf'] = calculate_dcf_valuation(cashflow_df)

        # Select relevant columns for ratios
        main_df['priceDcfPct'] = (main_df['price'] - main_df['estimatedPriceDcf']) / main_df['price']
        
        valuation_df = main_df[[
            'price', 'estimatedPriceDcf', 'priceDcfPct'
        ]]
        all_main_dfs.append(main_df)
        all_top5_dfs.append(top5_df)
        all_cashflow_df.append(cashflow_df)
        
        ratios_df = pd.concat([ratios_df, valuation_df])
    
    # Display results in Streamlit
    st.write("### Stock Valuation Analysis")
    pu.plot_single_bar(ratios_df['priceDcfPct'], title="Estimated Price (DCF)")
    

def run_evaluation_snapshot(tickers=['MWG']):
    """Main function to run the evaluation snapshot for multiple tickers"""
    ticker_data_dict = {}
    
    for ticker in tickers:
        data = get_stock_evaluation_snapshot(ticker)
        if data and data['index'] is not None:
            ticker_data_dict[ticker] = data
    
    if ticker_data_dict:
        plot_stock_evaluation_snapshot(ticker_data_dict)
    else:
        st.write("No data available for the specified tickers")

# Example usage
def run(symbol_benchmark, symbolsDate_dict):
    if len(symbolsDate_dict['symbols']) < 1:
        st.info("No stock selected")
        return
    
    symbols = symbolsDate_dict['symbols']
    run_evaluation_snapshot(symbols)

# For testing locally
if __name__ == "__main__":
    symbols_dict = {'symbols': ['MWG']}
    run("benchmark", symbols_dict)