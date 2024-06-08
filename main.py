import streamlit as st
from st_pages import show_pages_from_config

# Import the page configuration
show_pages_from_config("pages_config.toml")

def main():
    st.title("BeQuant Trading Platform 🐝")

if __name__ == "__main__":
    main()
