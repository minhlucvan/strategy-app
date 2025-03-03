import streamlit as st
import pandas as pd
import json
from importlib import import_module
from typing import Dict, Optional, Callable

from utils.component import input_SymbolsDate, check_password, params_selector, form_SavePortfolio
from utils.db import get_SymbolsName
from studies import STUDY_CONFIG

def check_params(params: Dict) -> bool:
    """Validate parameters dictionary"""
    return bool(params)  # Simplified validation, adjust as needed

def load_study_module(study_name: str) -> Optional[Callable]:
    """Dynamically load study module only when selected"""
    if study_name == "Select Study" or not study_name:
        return None
    
    try:
        module_name = STUDY_CONFIG.get(study_name)
        if not module_name:
            return None
            
        module = import_module(f".{module_name}", package="studies")
        return getattr(module, "run")
    except (ImportError, AttributeError) as e:
        st.error(f"Failed to load study {study_name}: {str(e)}")
        return None

def main():
    """Main application logic"""
    if not check_password():
        return

    # Sidebar study selection
    study_names = ["Select Study"] + list(STUDY_CONFIG.keys())
    selected_study = st.sidebar.selectbox("Please select Study", study_names)

    # Input symbols and date
    symbols_date_dict = input_SymbolsDate(group=True)
    symbol_benchmark = symbols_date_dict.get('benchmark')

    if selected_study and selected_study != "Select Study":
        st.write(f"### Study: {selected_study}")
        
        # Load study module only after selection
        study_module = load_study_module(selected_study)
        
        if study_module:
            try:
                study_module(symbol_benchmark, symbols_date_dict)
            except Exception as e:
                st.error(f"Error running study: {str(e)}")
                raise
        else:
            st.warning(f"Could not load study: {selected_study}")
    else:
        st.info("Please select a study to begin.")

if __name__ == "__main__":
    main()