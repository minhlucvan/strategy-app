from json.tool import main
from utils.component import check_password
import streamlit as st
import utils.config as config
import json

def main():
    st.title("Configuration")
    config_dict = config.load_config()
    
    # password
    passowrd = st.text_input("Password", type="password")
    
    # telegram token
    telegram_token = st.text_input("Telegram Token", config_dict.get('telegram', {}).get('token', ''))
    
    # tcbs secrets
    tcbs_info_raw = config_dict.get('tcbs', {}).get('info', '')
    tcbs_info_str = json.dumps(tcbs_info_raw, indent=4) if tcbs_info_raw else ''
    tcbs_info = st.text_area("TCBS Info", tcbs_info_str)
    
    # update button
    if st.button("Update"):
        tcbs_info_parsed = json.loads(tcbs_info)
        
        config.update_config({
            "password": passowrd,
            "tcbs": {
                "info": tcbs_info_parsed
            },
            "telegram": {
                "token": telegram_token
            }
        })
        st.write("Updated successfully.")
    
    # clear cache
    if st.button("Clear Cache"):
        config.clear_cache()
        st.write("Cache cleared.")
    

if __name__ == "__main__":
    if check_password():
        main()
