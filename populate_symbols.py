import config
import sqlite3
import akshare as ak
import pandas as pd
import utils.data_bin as bin

connection = sqlite3.connect('db/portfolio.db')
cursor = connection.cursor()

def gen_vn_symbol():
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

def gen_vn_index_symbol():
    data = [
        ('VN30', 'VN30', 'HSX', False, 'INDEX'),
        ('E1VFVN30', 'E1VFVN30', 'HSX', True, 'ETF'),
        ('FUEVN100', 'FUEVN100', 'HSX', True, 'ETF'),
        ('HNX-INDEX', 'HNX-INDEX', 'HNX', False, 'INDEX'),
        ('HNX30-INDEX', 'HNX30-INDEX', 'HNX', False, 'INDEX'),
    ]
    
    for row in data:
        cursor.execute("""
            INSERT INTO stock (name, symbol, exchange, is_etf, category)
            VALUES (?, ?, ?, ?, ?)
        """, row)
        print(f'Successfully inserted {row[0]} into stock table')

def gen_usdt_symbols():
    print('Fetching USDT data...')
    tickers_df = bin.get_all_USDT_tickers()
    
    for ticker in tickers_df['symbol']:
        print(f'Inserting {ticker} into stock table')
        cursor.execute("""
            INSERT INTO stock (name, symbol, exchange, is_etf, category)
            VALUES (?, ?, ?, false, 'CRYPTO')
        """, (ticker, ticker, 'BINANCE'))

def clear_table():
    cursor.execute("DELETE FROM stock")

clear_table()    
gen_vn_symbol()
gen_vn_index_symbol()
gen_usdt_symbols()
connection.commit()
