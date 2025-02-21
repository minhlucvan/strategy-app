import datetime
import vectorbt as vbt

from utils.portfolio import Portfolio

# from telegram.ext import CommandHandler

import toml
import logging

logging.basicConfig(level=logging.INFO)  # enable logging
def update_portfolio():
    portfolio = Portfolio()
    logging.info("--Update portfolio.")
    if portfolio.updateAll():
        logging.info("--Update portfolio sucessfully.")
        print("Update portfolios sucessfully.")

        check_df = portfolio.check_records(dt=datetime.date.today())
        # send the notification to telegram users
        if len(check_df) == 0:
            print("No signal found.")
        else:
            print(f"Found {len(check_df)} signal.")
            for i, row in check_df.iterrows():
                symbol_str = str(i+1) + '.' + row['name'] + ' : ' + row['records']
                print(symbol_str)
    else:
        logging.error("--Failed to update portfolio.")
        
if __name__ == '__main__':
    update_portfolio()