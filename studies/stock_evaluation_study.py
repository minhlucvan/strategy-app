import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objs as go
import utils.plot_utils as pu
from plotly.subplots import make_subplots
from utils.stock_utils import get_stock_evaluation_snapshot, load_stock_evaluation_snapshot_to_dataframe, get_intraday_snapshots, load_intraday_snapshots_to_dataframe
def calculate_dcf_valuation(data, discount_rate=0.12, growth_rate=0.05):
    """Perform DCF valuation based on the provided JSON data.
    
    Args:
        data: Dictionary containing cash flow data and financial parameters
        discount_rate: Weighted Average Cost of Capital (WACC), default 12%
        growth_rate: Perpetual growth rate, default 5%
    
    Returns:
        float: The DCF valuation per share
    """
    # Extract relevant data from the input JSON
    cash_flow_list = data.get("cashFlow", [])
    shares_outstanding = data.get("shareOutstanding", 1)  # Avoid division by zero

    if not cash_flow_list:
        print("Error: No cash flow data provided")
        return None

    # Use the provided 5 years of projected cash flows directly
    projection_years = len(cash_flow_list)
    future_fcfs = [cf["freeCashFlow"] for cf in cash_flow_list]

    # Calculate present value of projected cash flows
    present_values = [fcf / (1 + discount_rate)**i for i, fcf in enumerate(future_fcfs, 1)]

    # Calculate terminal value using the last projected FCF
    terminal_fcf = future_fcfs[-1] * (1 + growth_rate)
    terminal_value = terminal_fcf / (discount_rate - growth_rate)
    pv_terminal = terminal_value / (1 + discount_rate)**projection_years

    # Total enterprise value (sum of PV of cash flows + PV of terminal value)
    total_pv = sum(present_values) + pv_terminal

    # Adjust for net debt and convert to equity value
    net_debt = data.get("netDebt", 0)
    equity_value = total_pv - net_debt

    # Calculate per-share value
    dcf_per_share = equity_value / shares_outstanding

    return round(dcf_per_share, 0)

def plot_stock_evaluation_snapshot(ticker_data_dict):
    all_main_dfs = []
    all_top5_dfs = []
    all_cashflow_df = []
    ratios_df = pd.DataFrame()
    
    for ticker, data in ticker_data_dict.items():
        main_df, top5_df, cashflow_df = load_stock_evaluation_snapshot_to_dataframe(data)
                
        # Include industry averages in the top6_df for std calculation
        top6_df = pd.concat([top5_df, pd.DataFrame({
            'ticker': ['industry'],
            'pe': main_df['industry_pe'].iloc[0],
            'pb': main_df['industry_pb'].iloc[0],
            'evebitda': main_df['industry_evebitda'].iloc[0]
        })])
                
        # Calculate standard deviations across top 5 + industry
        main_df['pe_std'] = top6_df['pe'].std()
        main_df['pb_std'] = top6_df['pb'].std()
        main_df['evebitda_std'] = top6_df['evebitda'].std()
                
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
        # Estimated prices based on industry multiples
        main_df['estimatedPricePe'] = main_df['eps'] * main_df['industry_pe']
        main_df['estimatedPricePb'] = main_df['bvps'] * main_df['industry_pb']
        main_df['estimatedPriceEvEbitda'] = main_df['ebitda'] * main_df['industry_evebitda']
        
        # estimated price dcfs
        main_df['estimatedPriceDcf'] = calculate_dcf_valuation(cashflow_df)
        main_df['priceDcfPercentDiff'] = (main_df['price'] - main_df['estimatedPriceDcf']) / main_df['price']
            
        # Percent differences between actual and estimated prices
        main_df['pricePePercentDiff'] = (main_df['price'] - main_df['estimatedPricePe']) / main_df['price']
        main_df['pricePbPercentDiff'] = (main_df['price'] - main_df['estimatedPricePb']) / main_df['price']
        main_df['priceEvEbitdaPercentDiff'] = (main_df['price'] - main_df['estimatedPriceEvEbitda']) / main_df['price'] 
        
        if pd.isna(main_df['priceEvEbitdaPercentDiff'].iloc[0]):
            main_df['evebitda_weight'] = 0
            main_df['priceEvEbitdaPercentDiff'] = 0

        # fill missing pe_std, pb_std, evebitda_std
        main_df['pe_std'] = main_df['pe_std'].fillna(0)
        main_df['pb_std'] = main_df['pb_std'].fillna(0)
        main_df['evebitda_std'] = main_df['evebitda_std'].fillna(0)
        
        
        # Calculate weights based on inverse of standard deviations
        std_values = [main_df['pe_std'].iloc[0], main_df['pb_std'].iloc[0], main_df['evebitda_std'].iloc[0]]
        # Handle potential zero or NaN std values by replacing with a small number if necessary
        std_values = [max(val, 0.001) if not pd.isna(val) else 0.001 for val in std_values]
        inverse_std = [1 / std for std in std_values]
        total_inverse_std = sum(inverse_std)
        weights = [inv / total_inverse_std for inv in inverse_std]  # Normalize weights to sum to 1
        
        # Assign weights to each method, excluding methods with zero weights
        pe_weight, pb_weight, evebitda_weight = weights
        if main_df['pe_std'].iloc[0] == 0:
            pe_weight = 0
        if main_df['pb_std'].iloc[0] == 0:
            pb_weight = 0
        if main_df['evebitda_std'].iloc[0] == 0:
            evebitda_weight = 0
        
        # Normalize weights again to ensure they sum to 1
        total_weight = pe_weight + pb_weight + evebitda_weight
        if total_weight > 0:
            pe_weight /= total_weight
            pb_weight /= total_weight
            evebitda_weight /= total_weight
        
        main_df['pe_weight'] = pe_weight
        main_df['pb_weight'] = pb_weight
        main_df['evebitda_weight'] = evebitda_weight
                  
        
        # Calculate weighted valuation instead of simple mean
        main_df['weightedValuation'] = (
            main_df['pricePePercentDiff'] * pe_weight +
            main_df['pricePbPercentDiff'] * pb_weight +
            main_df['priceEvEbitdaPercentDiff'] * evebitda_weight
        )
        
        # Combine all ratios and weights for display
        valuation_df = main_df[[
            'pricePePercentDiff', 'pricePbPercentDiff', 'priceEvEbitdaPercentDiff', 'priceDcfPercentDiff',
            'pe_std', 'pb_std', 'evebitda_std',
            'pe_weight', 'pb_weight', 'evebitda_weight',
            'weightedValuation'
        ]]
        
        all_main_dfs.append(main_df)
        all_top5_dfs.append(top5_df)
        all_cashflow_df.append(cashflow_df)
        
        ratios_df = pd.concat([ratios_df, valuation_df])
    
    # Display results in Streamlit
    st.write("### Stock Valuation Analysis")
    st.dataframe(ratios_df.style.format({
        'pricePePercentDiff': '{:.2%}', 'pricePbPercentDiff': '{:.2%}', 'priceEvEbitdaPercentDiff': '{:.2%}', 'priceDcfPercentDiff': '{:.2%}',
        'pe_std': '{:.2f}', 'pb_std': '{:.2f}', 'evebitda_std': '{:.2f}',
        'pe_weight': '{:.2%}', 'pb_weight': '{:.2%}', 'evebitda_weight': '{:.2%}',
        'weightedValuation': '{:.2%}'
    }))
        
    # top weighted valuation
    top_weighted_valuation = ratios_df['weightedValuation'].sort_values(ascending=True)
    
    pu.plot_single_bar(top_weighted_valuation, title='Top Weighted Valuation', x_title='Ticker', y_title='Weighted Valuation')
    
    # top 10 valuation by DCF
    top10_dcf_valuation = ratios_df['priceDcfPercentDiff'].sort_values(ascending=True).dropna()
    pu.plot_single_bar(top10_dcf_valuation, title='Top 10 DCF Valuation', x_title='Ticker', y_title='DCF Valuation')
    

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