import config
import sqlite3
import akshare as ak
import pandas as pd

connection = sqlite3.connect('db/portfolio.db')
cursor = connection.cursor()

def gen_us_symbol():
    print('Fetching us stock data...')
    # assets = api.list_assets()
    assets_df = ak.stock_us_spot_em()

    for _, row in assets_df.iterrows():
        symbol = row['代码'].split('.')[1]
        exchange = row['代码'].split('.')[0]
        name = row['名称']

        cursor.execute("""
            INSERT INTO stock (name, symbol, exchange, is_etf, category)
            VALUES (?, ?, ?, false, 'STOCK')
        """, (name, symbol, exchange))
    print(f'Successfully inserted {len(assets_df)} records into stock table')

def gen_cn_symbol():
    print('Fetching cn stock data...')
    assets_df = ak.stock_zh_a_spot_em()

    for index, row in assets_df.iterrows():
        symbol = row['代码']
        exchange = 'A'

        cursor.execute("""
            INSERT INTO stock (name, symbol, exchange, is_etf, category)
            VALUES (?, ?, ?, false, 'STOCK')
        """, (row['名称'], symbol, exchange))
    print(f'Successfully inserted {len(assets_df)} records into stock table')

def gen_cn_symbol():
    print('Fetching cn stock data...')
    assets_df = ak.stock_zh_a_spot_em()

    for index, row in assets_df.iterrows():
        symbol = row['代码']
        exchange = 'A'

        cursor.execute("""
            INSERT INTO stock (name, symbol, exchange, is_etf, category)
            VALUES (?, ?, ?, false, 'STOCK')
        """, (row['名称'], symbol, exchange))
    
    print(f'Successfully inserted {len(assets_df)} records into stock table')

def gen_hk_symbol():
    print('Fetching hk stock data...')
    assets_df = ak.stock_hk_spot_em()

    for index, row in assets_df.iterrows():
        symbol = row['代码']
        exchange = 'HK'

        cursor.execute("""
            INSERT INTO stock (name, symbol, exchange, is_etf, category)
            VALUES (?, ?, ?, false, 'STOCK')
        """, (row['名称'], symbol, exchange))
    print(f'Successfully inserted {len(assets_df)} records into stock table')

def gen_cnindex_symbol():
    print('Fetching cn index data...')
    assets_df = ak.stock_zh_index_spot_em()

    for index, row in assets_df.iterrows():
        symbol = row['代码']
        exchange = 'CNINDEX'

        cursor.execute("""
            INSERT INTO stock (name, symbol, exchange, is_etf, category)
            VALUES (?, ?, ?, false, 'INDEX')
        """, (row['名称'], symbol, exchange))
    print(f'Successfully inserted {len(assets_df)} records into stock table')
    
def gen_cnfund_etf():
    print('Fetching fund etf data...')
    fund_etf_lof_df = ak.fund_etf_category_sina(symbol="LOF基金")
    fund_etf_etf_df = ak.fund_etf_category_sina(symbol="ETF基金")
    fund_etf_fb_df = ak.fund_etf_category_sina(symbol="封闭式基金")

    fund_eft_df=pd.concat([fund_etf_lof_df,fund_etf_etf_df,fund_etf_fb_df])
    for index, row in fund_eft_df.iterrows():
        cursor.execute(" INSERT INTO stock (id, name, symbol, exchange, is_etf, category) \
            VALUES (?, ?, ?, 'CN', false, 'FUND_ETF' )", (None, row['名称'], row['代码']))
    print(f'Successfully inserted {len(fund_eft_df)} records into stock table')

def gen_vnindex_symbol():
    print('Fetching vn index data...')
    assets_dfs = pd.read_html('https://vi.wikipedia.org/wiki/Danh_s%C3%A1ch_c%C3%B4ng_ty_tr%C3%AAn_s%C3%A0n_giao_d%E1%BB%8Bch_ch%E1%BB%A9ng_kho%C3%A1n_Vi%E1%BB%87t_Nam')
    
    assets_df = pd.DataFrame()
    
    for df in assets_dfs:
        if 'CK' in df.columns:
            assets_df = pd.concat([assets_df, df])
            print(f'Added {len(df)} records')
    
    # drop nan rows
    assets_df = assets_df.dropna(subset=['CK', 'SÀN', 'TÊN CÔNG TY'])
    
    # drop duplicates CK
    assets_df = assets_df.drop_duplicates(subset=['CK'])
    
    # select only SÀN in HSX, HNX
    assets_df = assets_df[assets_df['SÀN'].isin(['HSX', 'HNX'])]
    
    for index, row in assets_df.iterrows():
        symbol = row['CK']
        exchange = row['SÀN']
        name = row['TÊN CÔNG TY']
        
        cursor.execute("""
            INSERT INTO stock (name, symbol, exchange, is_etf, category)
            VALUES (?, ?, ?, false, 'STOCK')
        """, (name, symbol, exchange))
    
    print(f'Successfully inserted {len(assets_df)} records into stock table')

def clear_table():
    cursor.execute("DELETE FROM stock")

clear_table()    
gen_us_symbol()
gen_cn_symbol()
gen_hk_symbol()
gen_cnindex_symbol()
gen_cnfund_etf()
gen_vnindex_symbol()
connection.commit()
