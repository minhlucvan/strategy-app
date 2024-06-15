import config
import sqlite3
import akshare as ak
import pandas as pd
import utils.data_bin as bin

class PortfolioDB:
    def __init__(self, db_path):
        self.db_path = db_path

    def __enter__(self):
        self.connection = sqlite3.connect(self.db_path)
        self.cursor = self.connection.cursor()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.connection.commit()
        self.connection.close()

    def execute_query(self, query, params=None):
        if params:
            self.cursor.execute(query, params)
        else:
            self.cursor.execute(query)

    def drop_table(self, table_name):
        self.execute_query(f"DROP TABLE IF EXISTS {table_name}")

    def create_stock_table(self):
        self.execute_query("""
            CREATE TABLE stock (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                symbol TEXT NOT NULL,
                exchange TEXT NOT NULL,
                is_etf BOOLEAN NOT NULL,
                category TEXT NOT NULL,
                market TEXT NOT NULL
            )
        """)

    def insert_stock(self, name, symbol, exchange, is_etf, category, market):
        self.execute_query("""
            INSERT INTO stock (name, symbol, exchange, is_etf, category, market)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (name, symbol, exchange, is_etf, category, market))

def fetch_vn_symbol_data():
    print('Fetching VN index data...')
    assets_dfs = pd.read_html('https://vi.wikipedia.org/wiki/Danh_s%C3%A1ch_c%C3%B4ng_ty_tr%C3%AAn_s%C3%A0n_giao_d%E1%BB%8Bch_ch%E1%BB%A9ng_kho%C3%A1n_Vi%E1%BB%87t_Nam')
    assets_df = pd.DataFrame()
    
    for df in assets_dfs:
        if 'CK' in df.columns:
            assets_df = pd.concat([assets_df, df])
            print(f'Added {len(df)} records')
    
    # Clean data
    assets_df.dropna(subset=['CK', 'SÀN', 'TÊN CÔNG TY'], inplace=True)
    assets_df.drop_duplicates(subset=['CK'], inplace=True)
    assets_df = assets_df[assets_df['SÀN'].isin(['HSX', 'HNX', 'Upcom'])]
    
    # upeprcase columns
    assets_df['SÀN'] = assets_df['SÀN'].str.upper()
    
    # deduplicate ticker
    assets_df['CK'] = assets_df['CK'].str.replace(' ', '')
    assets_df['CK'] = assets_df['CK'].str.replace('-', '')
    assets_df['CK'] = assets_df['CK'].str.replace('(', '')
    assets_df['CK'] = assets_df['CK'].str.replace(')', '')
    assets_df['CK'] = assets_df['CK'].str.replace('.', '')
    
    # drop duplicates
    assets_df.drop_duplicates(subset=['CK'], inplace=True)
    
    return assets_df

def fetch_usdt_data():
    print('Fetching USDT data...')
    return bin.get_all_USDT_tickers()

def populate_vn_symbols(db):
    assets_df = fetch_vn_symbol_data()
    
    for index, row in assets_df.iterrows():
        db.insert_stock(row['TÊN CÔNG TY'], row['CK'], row['SÀN'], False, 'STOCK', 'VN')
    print(f'Successfully inserted {len(assets_df)} records into stock table')

def populate_vn_index_symbols(db):
    data = [
        ('VN30', 'VN30', 'HSX', False, 'INDEX', 'VN'),
        ('E1VFVN30', 'E1VFVN30', 'HSX', True, 'ETF', 'VN'),
        ('FUEVN100', 'FUEVN100', 'HSX', True, 'ETF', 'VN'),
        ('HNX-INDEX', 'HNX-INDEX', 'HNX', False, 'INDEX', 'VN'),
        ('HNX30-INDEX', 'HNX30-INDEX', 'HNX', False, 'INDEX', 'VN'),
    ]
    for row in data:
        db.insert_stock(*row)
        print(f'Successfully inserted {row[0]} into stock table')

def populate_vn_derivative(db):
    data = [
        ('VN30F1M', 'VN30F1M', 'HSX', False, 'DERIVATIVE', 'VN')
    ]
    for row in data:
        db.insert_stock(*row)
        print(f'Successfully inserted {row[0]} into stock table')

def populate_usdt_symbols(db):
    tickers_df = fetch_usdt_data()
    for ticker in tickers_df['symbol']:
        db.insert_stock(ticker, ticker, 'BINANCE', False, 'CRYPTO', 'USDT')
    print(f'Successfully inserted {len(tickers_df)} records into stock table')

def main():
    with PortfolioDB('db/portfolio.db') as db:
        db.drop_table('stock')
        db.create_stock_table()
        populate_vn_symbols(db)
        populate_vn_index_symbols(db)
        populate_usdt_symbols(db)
        populate_vn_derivative(db)

if __name__ == '__main__':
    main()
