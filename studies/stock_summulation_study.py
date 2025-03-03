import numpy as np
import pandas as pd
from arch import arch_model
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils.processing import get_stocks

def monte_carlo_simulation(df, num_simulations, num_days, lookback_days=252, break_even_price=0):
    # Ensure lookback_days doesn't exceed available data
    lookback_days = min(lookback_days, len(df))
    last_year_df = df.iloc[-lookback_days:]
    returns = last_year_df.pct_change().dropna() * 100  # Scale to percentage for GARCH
    
    if returns.empty:
        raise ValueError("No valid returns data available")

    # Compute mean returns (assumed constant for simplicity)
    returns_mean = returns.mean().values

    # Fit GARCH(1,1) model for each stock to estimate conditional volatility
    garch_vol = []
    for stock in returns.columns:
        model = arch_model(returns[stock], vol='Garch', p=1, q=1, mean='Constant', dist='Normal')
        res = model.fit(disp='off')
        forecast = res.forecast(horizon=1, method='analytic')
        cond_vol = np.sqrt(forecast.variance.iloc[-1].values[0]) / 100  # Convert from %^2 to daily std
        garch_vol.append(cond_vol)
    
    returns_std = np.array(garch_vol)  # Dynamic volatility from GARCH

    # Get the last close prices for each stock
    last_close = df.iloc[-1].values

    # Simulate price paths
    simulation_results = np.zeros((num_simulations, num_days, df.shape[1]))
    simulation_results[:, 0, :] = last_close

    # Generate random returns with GARCH volatility
    for t in range(1, num_days):
        random_returns = np.random.normal(returns_mean / 100, returns_std, (num_simulations, df.shape[1]))
        random_returns = np.clip(random_returns, -0.07, 0.07)  # Vietnamese market daily limit
        simulation_results[:, t, :] = simulation_results[:, t-1, :] * (1 + random_returns)

    # Compute stats for all stocks
    stats = []
    for stock_idx, stock_name in enumerate(df.columns):
        sim_results = simulation_results[:, -1, stock_idx]
        stats.append({
            'stock': stock_name,
            'min': np.min(sim_results),
            'max': np.max(sim_results),
            'mean': np.mean(sim_results),
            'median': np.median(sim_results),
            'std_high': np.mean(sim_results) + np.std(sim_results),
            'std_low': np.mean(sim_results) - np.std(sim_results),
            'win_rate': np.sum(sim_results > break_even_price) / num_simulations
        })

    result_df = pd.DataFrame(stats)
    return result_df, simulation_results  # Return simulation paths for visualization

def visualize_simulations(df, simulation_results, num_simulations, num_days):
    num_stocks = df.shape[1]
    # Create one subplot per stock
    fig = make_subplots(rows=num_stocks, cols=2, 
                        subplot_titles=[f"{stock} Paths" for stock in df.columns] + 
                                      [f"{stock} Final Dist" for stock in df.columns],
                        column_widths=[0.6, 0.4],
                        vertical_spacing=0.1)

    days = np.arange(num_days)
    for stock_idx, stock_name in enumerate(df.columns):
        # Plot a subset of simulation paths (e.g., 50 paths for clarity)
        sample_size = min(50, num_simulations)
        sample_indices = np.random.choice(num_simulations, sample_size, replace=False)
        
        # Price paths
        for sim in sample_indices:
            fig.add_trace(
                go.Scatter(x=days, y=simulation_results[sim, :, stock_idx], 
                          mode='lines', line=dict(width=0.5), opacity=0.3, showlegend=False),
                row=stock_idx + 1, col=1
            )
        
        # Add historical data
        fig.add_trace(
            go.Scatter(x=days - len(df), y=df[stock_name], 
                      mode='lines', line=dict(color='black', width=2), name='Historical'),
            row=stock_idx + 1, col=1
        )

        # Histogram of final prices
        final_prices = simulation_results[:, -1, stock_idx]
        fig.add_trace(
            go.Histogram(x=final_prices, nbinsx=30, opacity=0.7, showlegend=False),
            row=stock_idx + 1, col=2
        )

        # Add mean and break-even lines to histogram
        mean_price = np.mean(final_prices)
        fig.add_vline(x=mean_price, line=dict(color='red', dash='dash'), 
                     row=stock_idx + 1, col=2)
        fig.add_vline(x=0, line=dict(color='green', dash='dash'), 
                     row=stock_idx + 1, col=2)

    # Update layout
    fig.update_layout(
        height=300 * num_stocks, width=1000,
        title_text="Monte Carlo Simulation Results",
        showlegend=True
    )
    fig.update_xaxes(title_text="Days", col=1)
    fig.update_xaxes(title_text="Final Price", col=2)
    fig.update_yaxes(title_text="Price", col=1)
    fig.update_yaxes(title_text="Frequency", col=2)

    return fig

def run(symbol_benchmark, symbolsDate_dict):
    if not symbolsDate_dict['symbols']:
        st.info("Please select symbols, mate!")
        st.stop()

    stock_df = get_stocks(symbolsDate_dict, 'close')
    
    st.write("Stock Data", stock_df)

    # Run simulation
    num_simulations = 1000
    num_days = 30
    break_even_price = 110
    result_df, simulation_results = monte_carlo_simulation(
        stock_df, num_simulations, num_days, break_even_price=break_even_price
    )
    
    st.write("Simulation Statistics", result_df)

    # Visualize simulations
    fig = visualize_simulations(stock_df, simulation_results, num_simulations, num_days)
    st.plotly_chart(fig)

# Example usage: Run this in Streamlit with appropriate symbolsDate_dict