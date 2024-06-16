
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import utils.algotrade as at

from studies.stock_news_momentum_study import calculate_price_changes
from studies.stock_overnight_study import plot_scatter_2_sources
from utils.plot_utils import plot_double_side_bars, plot_multi_bar, plot_multi_line, plot_single_bar, plot_single_line, plot_multi_scatter, plot_events
from utils.processing import get_stocks, get_stocks_foregin_flow

import plotly.graph_objects as go
import streamlit as st

def run(symbol_benchmark, symbolsDate_dict):
    
    if len(symbolsDate_dict['symbols']) < 1:
        st.info("Please select symbols.")
        st.stop()
        
    foreign_buy_volume_dt = at.get_foreign_buy_volume()
    foreign_buy_volume_df = at.load_foreign_buy_volume_to_dataframe(foreign_buy_volume_dt)
    # st.write(foreign_buy_volume_df)
    
    foreign_sell_volume_dt = at.get_foreign_sell_volume()
    foreign_sell_volume_df = at.load_foreign_sell_volume_to_dataframe(foreign_sell_volume_dt)
    
    # stock_foreign_buy_flow_df = get_stocks_foregin_flow(symbolsDate_dict, 'foreignBuy')
    # stock_foreign_sell_flow_df = get_stocks_foregin_flow(symbolsDate_dict, 'foreignSell')
    stock_foreign_net_flow_df = get_stocks_foregin_flow(symbolsDate_dict, 'netForeignVol')
    
    # st.write(foreign_sell_volume_df)
    # union data
    union_index = foreign_sell_volume_df.index.union(foreign_buy_volume_df.index)
    foreign_flow_df = pd.DataFrame(index=union_index, columns=['foreign_sell_volume', 'foreign_buy_volume'])
    
    # remove duplicate labels
    foreign_sell_volume_df = foreign_sell_volume_df[~foreign_sell_volume_df.index.duplicated()]
    foreign_buy_volume_df = foreign_buy_volume_df[~foreign_buy_volume_df.index.duplicated()]
    
    foreign_flow_df['foreign_sell_volume_abs'] = foreign_sell_volume_df['value']
    foreign_flow_df['foreign_sell_volume'] = -foreign_sell_volume_df['value']
    foreign_flow_df['foreign_buy_volume'] = foreign_buy_volume_df['value']
    
    foreign_flow_df['net_foreign_flow'] = foreign_flow_df['foreign_buy_volume'] + foreign_flow_df['foreign_sell_volume']
    
    foreign_flow_df['accumulated_foreign_flow'] = foreign_flow_df['net_foreign_flow'].cumsum()
    foreign_flow_df['accumulated_sell_volume'] = foreign_flow_df['foreign_sell_volume'].cumsum()
    foreign_flow_df['accumulated_sell_volume_abs'] = foreign_flow_df['foreign_sell_volume_abs'].cumsum()

    # plot total foreign flow
    total_foreign_flow = stock_foreign_net_flow_df.sum(axis=1)

    # copy the symbolsDate_dict
    # benchmark_dict = symbolsDate_dict.copy()
    symbolsDate_dict['symbols'] = symbolsDate_dict['symbols'] + ['VN30F1M']
    
    stock_df = get_stocks(symbolsDate_dict, 'close')
    
    stock_df = stock_df[stock_df.index >= total_foreign_flow.index[0]]

    foreign_flow_df = foreign_flow_df[foreign_flow_df.index >= total_foreign_flow.index[0]]

    benchmark_df = stock_df['VN30F1M']
    
    stock_df.drop(columns=['VN30F1M'], inplace=True)
    
    plot_single_line(benchmark_df, title='Stock Prices', x_title='Date', y_title='Price', legend_title='Stocks')
    
    plot_double_side_bars(foreign_flow_df, 'foreign_buy_volume', 'foreign_sell_volume', 'accumulated_foreign_flow', 'Foreign Buy Volume', 'Foreign Sell Volume', 'Net Foreign Flow', 'Foreign Flow')
    
    plot_single_bar(foreign_flow_df['foreign_sell_volume_abs'], title='Accumulated Sell Volume', x_title='Date', y_title='Volume', legend_title='Volume')
    
    plot_multi_bar(stock_foreign_net_flow_df, title='Foreign Net Flow', x_title='Date', y_title='Volume', legend_title='Stocks')
    
    accumulated_stock_foreign_net_flow = stock_foreign_net_flow_df.cumsum()
    
    plot_multi_bar(accumulated_stock_foreign_net_flow, title='Accumulated Foreign Net Flow', x_title='Date', y_title='Volume', legend_title='Stocks')
    
    plot_multi_line(stock_df, title='Stock Prices', x_title='Date', y_title='Price', legend_title='Stocks')
    
    plot_single_bar(total_foreign_flow, title='Total Foreign Flow', x_title='Date', y_title='Volume', legend_title='Stocks', price_df=benchmark_df)
    
    accumulated_foreign_flow = total_foreign_flow.cumsum()
    plot_single_line(accumulated_foreign_flow, title='Accumulated Foreign Flow', x_title='Date', y_title='Volume', legend_title='Stocks')
    
    plot_single_line(foreign_flow_df['accumulated_foreign_flow'], title='Accumulated Foreign Flow', x_title='Date', y_title='Volume', legend_title='Stocks')