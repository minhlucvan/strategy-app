import json
import utils.vietstock as vietstock
import utils.data_bin as data_bin
import streamlit as st

fund_file = "funds.json"

def get_extra_group(group: dict):
    extra_group = []
    if "provider" in group:
        if group["provider"] == "vietstock":
            extra_group = vietstock.get_group_symbols(url=group["url"])
        elif group["provider"] == "binance":
            data = data_bin.get_symbol_groups(group['code'])
            extra_group = data['symbol'].values.tolist()
        else:
            print(f"Provider {group['provider']} not implemented")
        
    return extra_group

def get_maket_groups(market: str):
    groups_data = {}
    with open(fund_file, 'r', encoding='UTF-8') as f:
        groups_data = json.load(f)
    
    market_groups = groups_data[market]
    
    if "extraGroup" in market_groups:
        extra_groups = market_groups["extraGroup"]
        for group_name in extra_groups.keys():
            group = extra_groups[group_name]
            extra_group = get_extra_group(group)
            market_groups['group'][group_name] = extra_group
            
    return market_groups