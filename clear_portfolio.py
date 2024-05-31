import config
import sqlite3
import akshare as ak
import pandas as pd

connection = sqlite3.connect('db/portfolio.db')
cursor = connection.cursor()


def clear_portfolio():
    cursor.execute("DELETE FROM portfolio")
    connection.commit()
    print('Successfully cleared stock and portfolio tables')
    

if __name__ == '__main__':
    try:
        clear_portfolio()
    finally:
        connection.close()