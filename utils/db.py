from distutils.log import error
import pandas as pd
import warnings
import sqlite3

from functools import cache

warnings.filterwarnings('ignore')

DBNAME = "db/portfolio.db" 

@cache
def init_connection():
    try:
        connection = sqlite3.connect(DBNAME, check_same_thread=False)
        cursor = connection.cursor()
        return connection, cursor
    except Exception  as e:
        print("Connnecting Database Error:", e)
        return None, None

# Initialize connection.
connection, cursor = init_connection()

@cache
def load_symbols(market:str=None):
        # cursor.execute("SELECT strategy_id, symbol, exchange, name FROM strategy_stock \
        #                     JOIN stock ON stock.id = strategy_stock.stock_id")
        # symbols_df = cursor.fetchall()
        query = "SELECT * FROM stock"
        if market:
            query += f" WHERE market='{market}'"
        result_df = pd.read_sql(query, connection)
        return result_df

@cache
def load_symbol(symbol:str):
        # cursor.execute("SELECT strategy_id, symbol, exchange, name FROM strategy_stock \
        #                     JOIN stock ON stock.id = strategy_stock.stock_id")
        # symbols_df = cursor.fetchall()
        try:
            result_df = pd.read_sql(f"SELECT * FROM stock WHERE symbol='{symbol}'", connection)
            return result_df

        except Exception as e:
            print(f"Error while loading symbol {symbol}: {e}")
            return symbol

@cache
def get_SymbolName(symbol:str | list):
        # cursor.execute("SELECT strategy_id, symbol, exchange, name FROM strategy_stock \
        #                     JOIN stock ON stock.id = strategy_stock.stock_id")
        # symbols_df = cursor.fetchall()
        try:
            result_df = pd.read_sql(f"SELECT name FROM stock WHERE symbol='{symbol}'", connection)
            return result_df.loc[0, 'name']

        except Exception as e:
            print(f"Error while loading symbol name {symbol}: {e}")
            return symbol

def get_SymbolsNames(symbols:list):
        
    if len(symbols) > 10:
        return symbols[:5] + [f'and {len(symbols)-5} more']
    
    if len(symbols) >= 5:
        return symbols

    
    symbols_slug = [f"'{symbol}'" for symbol in symbols]
    result_df = pd.read_sql(f"SELECT name FROM stock WHERE symbol IN ({','.join(symbols_slug)})", connection)
    
    results = result_df['name'].tolist()
    
    if len(results) > 3:
        return results[:5] + [f'and {len(results)-5} more']
    
    return results

def get_SymbolsName(symbols:list):
    names = set()
    for symbol in symbols:
        names.add(get_SymbolName(symbol))
    return names