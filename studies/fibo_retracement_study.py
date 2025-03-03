import os
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
import pickle
from fibonacci_ml.variables import RECOVERY_CRITERIA, TARGET_DENSITY
from utils.processing import get_stocks
import streamlit as st
from fibonacci_ml.subjective_drawdown_finder import SubjectiveDrawdown
from fibonacci_ml.core import FibonacciTechnicalAnalysis
import plotly.graph_objects as go

CORE_PATH = os.path.abspath(os.path.dirname(__file__))

def train_and_save_models(stock_df, target_density=TARGET_DENSITY, 
                         recovery_criteria=RECOVERY_CRITERIA,
                         model_pred_path=None, model_refine_path=None):
    """
    Train and save the predictor and refiner models for SubjectiveDrawdown
    """
    if model_pred_path is None:
        model_pred_path = os.path.join(CORE_PATH, "subjective_drawdown_models/subjective_drawdown_model1.pkl")
    if model_refine_path is None:
        model_refine_path = os.path.join(CORE_PATH, "subjective_drawdown_models/subjective_drawdown_model2.pkl")

    sd = SubjectiveDrawdown(target_density=target_density, recovery_criteria=recovery_criteria, auto_load=False, unit_test=False)
    
    X_pred, y_pred, X_refine, y_refine = prepare_training_data(stock_df, sd, target_density)
    
    model_pred = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    model_pred.fit(X_pred, y_pred)
    
    model_refine = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    model_refine.fit(X_refine, y_refine)
    
    os.makedirs(os.path.dirname(model_pred_path), exist_ok=True)
    
    with open(model_pred_path, 'wb') as f:
        pickle.dump(model_pred, f)
    with open(model_refine_path, 'wb') as f:
        pickle.dump(model_refine, f)
    
    return model_pred, model_refine

def prepare_training_data(stock_df, sd, target_density):
    """
    Prepare training data for both models by simulating various drawdown criteria
    """
    X_pred = []
    y_pred = []
    X_refine = []
    y_refine = []
    
    trend, std = sd.prefeature_trend(stock_df)
    vol, vold = sd.prefreature_realizedvol(stock_df)
    
    drawdown_crits = np.linspace(0.05, 0.7, 50)
    delta_time = (stock_df.index[-1] - stock_df.index[0]).days / 365.0
    
    for drawdown_crit in drawdown_crits:
        X_pred.append([drawdown_crit, trend, std, vol, vold])
        
        fibs = sd._get_fibs(stock_df, drawdown_crit)
        density = sd._density_of_drawdowns_given_fibs(fibs, delta_time=delta_time)
        y_pred.append(density)
        
        residual = target_density - density
        X_refine.append([trend, std, vol, vold, target_density, drawdown_crit, residual])
        
        results = sd._densities_by_kulling(fibs, delta_time, orig_drawdown=drawdown_crit)
        if len(results) > 1:
            sorted_results = results.sort_values('density')
            for i in range(len(sorted_results)-1):
                if sorted_results['density'].iloc[i] <= target_density <= sorted_results['density'].iloc[i+1]:
                    d1, d2 = sorted_results['density'].iloc[i], sorted_results['density'].iloc[i+1]
                    c1, c2 = sorted_results['drawdown_crit'].iloc[i], sorted_results['drawdown_crit'].iloc[i+1]
                    true_drawdown = c1 + (target_density - d1) * (c2 - c1) / (d2 - d1)
                    y_refine.append(true_drawdown)
                    break
            else:
                y_refine.append(sorted_results['drawdown_crit'].iloc[-1] if density < target_density else sorted_results['drawdown_crit'].iloc[0])
        else:
            y_refine.append(drawdown_crit)
    
    return (np.array(X_pred), np.array(y_pred), 
            np.array(X_refine), np.array(y_refine))
# Integrate with your run function

def run(symbol_benchmark, symbolsDate_dict):
    if not symbolsDate_dict['symbols']:
        st.info("Please select symbols, mate!")
        st.stop()
    
    stock_df = get_stocks(symbolsDate_dict, single=True)
    # Rename columns
    stock_df.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    }, inplace=True)
    
    st.line_chart(stock_df['Close'])
    
        
    if st.button('Train and Save Models'):
        train_and_save_models(stock_df)
        st.success('Models trained and saved successfully!')
    
    target_density = st.slider('Select Target Density', 0.01, 0.5, 4.0)
    
    # run the models
    if st.button('Run Models'):
        pred_model_path = os.path.join(CORE_PATH, "subjective_drawdown_models/subjective_drawdown_model1.pkl")
        refine_model_path = os.path.join(CORE_PATH, "subjective_drawdown_models/subjective_drawdown_model2.pkl")
        
        # Initialize SubjectiveDrawdown to find optimal drawdown
        sd = SubjectiveDrawdown(verbose=True, target_density=target_density, path_to_model_pred=pred_model_path, path_to_model_refine=refine_model_path)
        optimal_drawdown, results = sd.fit(stock_df)
        st.write(f"Optimal drawdown criteria: {optimal_drawdown:.3f}")

        # Initialize FibonacciTechnicalAnalysis with optimal drawdown
        fib_maker = FibonacciTechnicalAnalysis(stock_df, drawdown_criteria=optimal_drawdown, do_plot=True, plot_path='./images/')
        
        # Make the features
        features = fib_maker.make_fib_features()
        
        # show the images from the plot_path
        st.image('./images/fibonacci_timeseries.png')
        st.image('./images/price_and_fibs-1.png')
        st.image('./images/price_and_fibs-2.png')
        st.image('./images/price-snake-through-fib-1.png')
        st.image('./images/price-snake-through-fib-2.png')
 
 
    