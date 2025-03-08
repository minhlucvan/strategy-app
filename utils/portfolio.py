import pandas as pd
import pandas
from datetime import datetime, date, timedelta
import pytz
import json
import numpy as np
import os

import config
import warnings
import vectorbt as vbt
import streamlit as st

from utils.db import init_connection, get_SymbolName
from utils.stock_utils import get_last_trading_date

warnings.filterwarnings('ignore')
# Initialize connection.
connection, cursor = init_connection()


def selectpf_bySymbols(df, symbols: list):
    ids = []
    for i, row in df.iterrows():
        ids.append(i)
    return df.loc[ids, :]


class Portfolio(object):
    """
    manage the database of portforlio, the pf file in directory
    """
    allocation_dict =  {}
    is_live = False
    
    def __init__(self, is_live=False):
        self.df = pd.read_sql("SELECT * FROM portfolio", connection)
        self.df.set_index('id', inplace=True, drop=False)
        self.is_live = is_live

    def allocate(self, allocation_dict: dict):
        self.allocation_dict = allocation_dict

    def add(self, symbolsDate_dict: dict, strategyname: str, strategy_param, pf, description="desc", name=None) -> bool:
        """
            add a portforlio to db/table
            input:
                symbolsDate_dict = dictonary of symbols and date
                strategy = the strategy name
                strategy_param =  the strategy parameters related to the portfolio
                pf = the vbt portfolio 
            return:
                False = fail to add too the db/table or save the pf file
                True  = add to the library and save the pf file successfully

        """

        market = symbolsDate_dict['market']
        symbols = symbolsDate_dict['symbols']
        start_date = symbolsDate_dict['start_date']
        end_date = pf.value().index[-1].strftime("%Y-%m-%d")

        name = name if name else strategyname + '_' + ','.join(symbols)

        filename = str(datetime.now().timestamp()) + '.pf'
        pf.save(config.PORTFOLIO_PATH + filename)
        with open(config.PORTFOLIO_PATH + filename, 'rb') as pf_file:
            pf_blob = pf_file.read()
            pf_file.close()
            os.remove(config.PORTFOLIO_PATH + filename)

            try:
                tickers = "','".join(symbols)
                tickers = "'" + tickers + "'"
                sql_stat = f"SELECT * FROM stock WHERE symbol in ({tickers})"
                cursor.execute(sql_stat)
                stocks = cursor.fetchall()

                if len(stocks) != len(symbols):
                    print(f"Waring: some symbols are not in the stock table")

                if len(stocks) > 0:
                    param_json = json.dumps(strategy_param)
                    tickers = ','.join(symbols)

                    total_return = round(pf.stats('total_return')[0]/100.0, 2)
                    sharpe_ratio = round(pf.stats('sharpe_ratio')[0], 2)
                    maxdrawdown = round(pf.stats('max_dd')[0]/100.0, 2)
                    annual_return = round(pf.annualized_return(), 2)
                    lastday_return = round(pf.returns()[-1], 4)

                    cursor.execute("INSERT INTO portfolio (id, name, description, create_date, start_date, end_date, total_return, annual_return, lastday_return, sharpe_ratio, maxdrawdown, param_dict, strategy, symbols, market, vbtpf) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                                   (None, name, description, datetime.today(), start_date, end_date, total_return, annual_return, lastday_return, sharpe_ratio, maxdrawdown, param_json, strategyname, tickers, market, pf_blob))
                    connection.commit()

            except Exception as e:
                print("Portforlio.add error occurs:", e)
                connection.rollback()
                return False
        self.__init__()
        return True

    def delete(self, id) -> bool:
        """
            delete one of portforlios
            input:
                id = the id of portforlio in db/table
            return:
                False = fail to delete from the db/table or the pf file
                True  = delete in the db/table and save the pf file successfully

        """

        try:
            sql_stat = f"DELETE FROM portfolio WHERE id= {id}"
            cursor.execute(sql_stat)
            connection.commit()

            self.__init__()
            return True
        except Exception as e:
            print("'Fail to Delete the Portfolio...", e)
            connection.rollback()
            return False

    def update(self, id, force: bool = True) -> bool:
        """
            update the result of portforlio to today
            input:
                id = the id of portforlio in db/table
            return:
                False = fail to update the db/table or save the pf file
                True  = update the library and save the pf file successfully

        """
        id = int(id)
        market = self.df.loc[self.df['id'] == id, 'market'].values[0]
        symbols = self.df.loc[self.df['id'] ==
                              id, 'symbols'].values[0].split(',')
        strategyname = self.df.loc[self.df['id'] == id, 'strategy'].values[0]
        start_date = self.df.loc[self.df['id'] == id, 'start_date'].values[0]
        param_dict = self.df.loc[self.df['id'] == id, 'param_dict'].values[0]
        oend_date = pd.to_datetime(
            self.df.loc[self.df['id'] == id, 'end_date'].values[0])

        if market == 'US':
            end_date = datetime.now(pytz.timezone(
                'US/Eastern')) - timedelta(hours=9, minutes=30)
        else:
            end_date = date.today()

        end_date = datetime(
            year=end_date.year, month=end_date.month, day=end_date.day, tzinfo=pytz.utc)

        if force is False and oend_date == end_date:
            print(
                f"Portfolio_update_{self.df.loc[self.df['id']==id, 'name'].values[0]}: Today has been updated already.")
            return True

        if type(param_dict) == str:
            param_dict = json.loads(param_dict)

        if isinstance(start_date, np.datetime64) or type(start_date) == str:
            start_date = pd.to_datetime(start_date)

        symbolsDate_dict = {
            "market":   market,
            "symbols":  symbols,
            "start_date": start_date,
            "end_date": end_date,
        }

        # get the strategy class according to strategy name
        strategy_cli = getattr(__import__(
            f"vbt_strategy"), f"{strategyname}Strategy")
        strategy = strategy_cli(symbolsDate_dict)
        
        if self.is_live:
            strategy.enable_live()
        
        pf = strategy.update(param_dict)
        if pf is None:
            return False
        
        end_date = pf.value().index[-1].strftime("%Y-%m-%d")
        total_return = round(pf.stats('total_return')[0]/100.0, 2)
        
        returns = pf.returns()
        
        last_return = returns.iloc[-1]
        
        lastday_return = round(last_return, 4)
        
        sharpe_ratio = round(pf.stats('sharpe_ratio')[0], 2)
        maxdrawdown = round(pf.stats('max_dd')[0]/100.0, 2)
        annual_return = pf.annualized_return()
    
        if pd.isna(maxdrawdown):
            maxdrawdown = 0.0
        
        if isinstance(annual_return, pd.Series) and len(annual_return) > 0:
            annual_return = round(annual_return.iloc[-1], 2)
        else:
            annual_return = round(annual_return, 2)

        try:
            filename = str(datetime.now().timestamp()) + '.pf'
            pf.save(config.PORTFOLIO_PATH + filename)
            with open(config.PORTFOLIO_PATH + filename, 'rb') as pf_file:
                pf_blob = pf_file.read()
                pf_file.close()
                cursor.execute("UPDATE portfolio SET end_date=?, total_return=?, lastday_return=?, annual_return=?, sharpe_ratio=?, maxdrawdown=?, vbtpf=? WHERE id=?",
                               (end_date, total_return, lastday_return, annual_return, sharpe_ratio, maxdrawdown, pf_blob, id))
                connection.commit()
                os.remove(config.PORTFOLIO_PATH + filename)
                self.__init__(is_live=self.is_live)

        except FileNotFoundError as e:
            print(e)

        except Exception as e:
            st.write("Update portfolio error:", e)
            print("Update portfolio error:", e)
            connection.rollback()
            return False

        return True

    def updateAll(self) -> bool:
        for i in range(len(self.df)):
            if not self.update(self.df.loc[i, 'id']):
                print(f"Fail to update portfolio('{self.df.loc[i, 'name']}')")
                continue
            else:
                print(
                    f"Update portfolio('{self.df.loc[i,'name']}') successfully.")

        return True

    def check_records(self, dt: date,  last_trading_date=None, first_trade_date_of_week=None, selected_pfs=None) -> pd.DataFrame:
        '''
        Check all the portfolios which there're transations on dt:date
        '''
        is_today = dt == date.today()
        results = []
        for i in self.df.index:
            
            if selected_pfs is not None and i not in selected_pfs:
                continue
            
            try:
                pf = vbt.Portfolio.loads(self.df.loc[i, 'vbtpf'])
                records_df = pf.orders.records_readable.sort_values(by=[
                                                                    'Timestamp'])
                records_df['date'] = records_df['Timestamp'].dt.date
                name = self.df.loc[i, 'name']
                # TODO: store the timeformat in the database
                if name.startswith('MOMTOP_'):
                    # MONTOP is weekly strategy so we need to check the first trading day of the week
                    records_df = records_df[records_df['date'] == pd.to_datetime(first_trade_date_of_week).date()]
                elif is_today:
                    records_df = records_df[records_df['date'] == pd.to_datetime(last_trading_date).date()]
                else:
                    records_df = records_df[records_df['date'] == dt]

                if len(records_df) > 0:
                    for index, row in records_df.iterrows():
                        symbol_str = self.df.loc[i, 'symbols']
                        if type(row['Column']) == str:
                            symbol_str = row['Column']  
                        elif type(row['Column']) == tuple and type(row['Column'][-1]) == str:
                            symbol_str = row['Column'][-1]
                        
                        record = row.to_dict()
                        record['Symbol'] = symbol_str
                        record['Name'] = name
                        
                        # drop column
                        record.pop('Column')
                        
                        results.append(record)

            except ValueError as ve:
                st.error(ve)
                print(
                    f"portfolio-check_records:{self.df.loc[i,'name']} error --{ve}")
                continue
        
        result_df = pd.DataFrame(results)
        if self.is_live:
            result_df = self.ajust_positions(result_df)
                
        return result_df

    def ajust_positions(self, result_df: pd.DataFrame) -> bool:
        if len(self.allocation_dict) == 0:
            return result_df
        
        if result_df.empty:
            return result_df
        
        capital = self.allocation_dict['capital']
        # modify position to size to match the asset allocation
        for asset in self.allocation_dict['assets']:
            asset_ratio = self.allocation_dict['assets'][asset]
            
            if asset_ratio > 1:
                asset_ratio = asset_ratio / 100.0

            asset_capital = capital * asset_ratio
            asset_trades_df = result_df[result_df['Name'] == asset]
            asset_buy_df = asset_trades_df[asset_trades_df['Side'] == 'Buy']
            num_asset_buy = len(asset_buy_df)
            buy_capital = asset_capital / num_asset_buy
                        
            for i, row in asset_buy_df.iterrows():
                price = row['Price']
                size = int(buy_capital / price)
                
                size = max(size, 100)
                
                size = size - size % 100
                
                result_df.loc[i, 'Size'] = size
                
        return result_df
    def get_byName(self, svalue: str = 'MOM_AAPL') -> pd.DataFrame:
        result_df = self.df[self.df['name'] == svalue]
        return result_df

    def get_bySymbol(self, symbols: list = ['AAPL']) -> pd.DataFrame:
        result_df = selectpf_bySymbols(self.df, symbols)
        return result_df
