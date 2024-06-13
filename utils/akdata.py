from utils.data_vn import *

class AKData(object):
    def __init__(self, market):
        self.market = market

    @vbt.cached_method
    def get_stock(self, symbol: str, start_date: datetime.datetime, end_date: datetime.datetime, timeframe='1D') -> pd.DataFrame:
        stock_df = pd.DataFrame()
        print(f"AKData-get_stock: {symbol}, {self.market}")
        symbol_df = load_symbol(symbol)
        
        # hot fix for vietnam stock
        if symbol == 'E1VFVN30':
            symbol_df = pd.DataFrame([{'category': 'index'}])
            
        if symbol == 'VN30':
            symbol_df = pd.DataFrame([{'category': 'index'}])

        if len(symbol_df) == 1:  # self.symbol_dict.keys():
            print(
                f"AKData-get_stock: {symbol}, {self.market}, {symbol_df.at[0, 'category']}")
            func = ('get_' + self.market + '_' +
                    symbol_df.at[0, 'category']).lower()
            symbol_full = symbol
            if self.market == 'US':
                symbol_full = symbol_df.at[0, 'exchange'] + '.' + symbol

            try:
                stock_df = eval(func)(symbol=symbol_full, start_date=start_date.strftime(
                    "%Y%m%d"), end_date=end_date.strftime("%Y%m%d"), timeframe=timeframe)
            except Exception as e:
                print(f"AKData-get_stock {func} error: {e}")

            if not stock_df.empty:
                if len(stock_df.columns) == 6:
                    stock_df.columns = ['date', 'open',
                                        'close', 'high', 'low', 'volume']
                elif len(stock_df.columns) == 7:
                    stock_df.columns = [
                        'date', 'open', 'close', 'high', 'low', 'volume', 'amount']
                else:
                    pass
                stock_df.index = pd.to_datetime(stock_df['date'])
                stock_df.index = stock_df.index.tz_localize(None)
        return stock_df

    @vbt.cached_method
    def get_pettm(self, symbol: str) -> pd.DataFrame:
        print(f"AKData-get_pettm: {symbol}, {self.market}")
        stock_df = pd.DataFrame()
        symbol_df = load_symbol(symbol)

        if len(symbol_df) == 1:  # self.symbol_dict.keys():
            func = ('get_' + self.market + '_valuation').lower()
            try:
                stock_df = eval(func)(symbol=symbol, indicator='市盈率(TTM)')
            except Exception as e:
                print("get_pettm()---", e)

            if not stock_df.empty:
                stock_df.index = pd.to_datetime(stock_df['date'], utc=True)
                stock_df = stock_df['value']
        return stock_df
    
    @vbt.cached_method
    def get_valuation(self, symbol: str, indicator='pe') -> pd.DataFrame:
        print(f"AKData-get_valuation: {symbol}, {self.market}")
        stock_df = pd.DataFrame()
        symbol_df = load_symbol(symbol)

        if len(symbol_df) == 1:  # self.symbol_dict.keys():
            func = ('get_' + self.market + '_valuation').lower()
            try:
                stock_df = eval(func)(symbol=symbol, indicator=indicator)
            except Exception as e:
                print("get_pettm()---", e)

            if not stock_df.empty:
                stock_df.index = pd.to_datetime(stock_df['date'], utc=True)
                
        return stock_df
        
    @vbt.cached_method
    def get_fundamental(self, symbol: str) -> pd.DataFrame:
        print(f"AKData-get_fundamental: {symbol}, {self.market}")
        stock_df = pd.DataFrame()
        symbol_df = load_symbol(symbol)

        if len(symbol_df) == 1:
            func = ('get_' + self.market + '_fundamental').lower()
            try:
                stock_df = eval(func)(symbol=symbol)
            except Exception as e:
                print(e)

        return stock_df    

    @vbt.cached_method
    def get_financial(self, symbol: str, start_date: datetime.datetime = None, end_date: datetime.datetime = None) -> pd.DataFrame:
        print(f"AKData-get_financial: {symbol}, {self.market}, {start_date} - {end_date}")
        stock_df = pd.DataFrame()
        symbol_df = load_symbol(symbol)

        if len(symbol_df) == 1:
            func = ('get_' + self.market + '_financial').lower()
            try:
                stock_df = eval(func)(symbol=symbol)

                stock_df.index = pd.to_datetime(stock_df.index, utc=True)
                if start_date is not None and end_date is not None:
                    start_date = pd.to_datetime(start_date)
                    end_date = pd.to_datetime(end_date)
                    stock_df = stock_df[(stock_df.index >= start_date) & (stock_df.index <= end_date)]
                    
            except Exception as e:
                print(e)

        return stock_df
    
    @vbt.cached_method
    def get_events(self, symbol: str, start_date: datetime.datetime, end_date: datetime.datetime) -> pd.DataFrame:
        print(f"AKData-get_events: {symbol}, {self.market}")
        stock_df = pd.DataFrame()
        symbol_df = load_symbol(symbol)

        if len(symbol_df) == 1:
            func = ('get_' + self.market + '_events').lower()
            try:
                stock_df = eval(func)(symbol=symbol, start_date=start_date, end_date=end_date)
            except Exception as e:
                print(e)

        return stock_df
    
    @vbt.cached_method
    def get_stock_foregin_flow(self, symbol: str, start_date: datetime.datetime, end_date: datetime.datetime) -> pd.DataFrame:
        print(f"AKData-get_stock_foregin_flow: {symbol}, {self.market}")
        stock_df = pd.DataFrame()
        symbol_df = load_symbol(symbol)

        if len(symbol_df) == 1:
            func = ('get_' + self.market + '_foregin_flow').lower()
            try:
                stock_df = eval(func)(symbol=symbol, start_date=start_date, end_date=end_date)
            except Exception as e:
                print(e)

        return stock_df
    
    @vbt.cached_method
    def get_pegttm(self, symbol: str) -> pd.DataFrame:
        stock_df = pd.DataFrame()
        symbol_df = load_symbol(symbol)
        mv_df = pd.DataFrame()
        pettm_df = pd.DataFrame()

        if len(symbol_df) == 1:  # self.symbol_dict.keys():
            func = ('get_' + self.market + '_valuation').lower()
            try:
                pettm_df = eval(func)(symbol=symbol, indicator='市盈率(TTM)')
                mv_df = eval(func)(symbol=symbol, indicator='总市值')
            except Exception as e:
                print("get_pettm()---", e)

            if not mv_df.empty and not pettm_df.empty:
                pettm_df.index = pd.to_datetime(pettm_df['date'], utc=True)
                mv_df.index = pd.to_datetime(mv_df['date'], utc=True)
                stock_df = pd.DataFrame()
                stock_df['pettm'] = pettm_df['value']
                stock_df['mv'] = mv_df['value']
                stock_df['earning'] = stock_df['mv']/stock_df['pettm']
                stock_df['cagr'] = stock_df['earning'].pct_change(periods=252)
                stock_df['pegttm'] = stock_df['pettm'] / stock_df['cagr']/100
                stock_df = stock_df['pegttm']
        return stock_df

    @vbt.cached_method
    def get_news(self, symbol: str, start_date: datetime.datetime, end_date: datetime.datetime, channel_id='-1') -> pd.DataFrame:
        print(f"AKData-get_news: {symbol}, {self.market}")
        stock_df = pd.DataFrame()
        symbol_df = load_symbol(symbol)

        if len(symbol_df) == 1:
            func = ('get_' + self.market + '_news').lower()
            try:
                stock_df = eval(func)(symbol=symbol, start_date=start_date, end_date=end_date, channel_id=channel_id)
            except Exception as e:
                print("get_news()---", e)
                print(e)

        return stock_df

    @vbt.cached_method
    def get_document(self, symbol: str, start_date: datetime.datetime, end_date: datetime.datetime, doc_type='1') -> pd.DataFrame:
        print(f"AKData-get_document: {symbol}, {self.market}")
        stock_df = pd.DataFrame()
        symbol_df = load_symbol(symbol)

        if len(symbol_df) == 1:
            func = ('get_' + self.market + '_document').lower()
            try:
                stock_df = eval(func)(symbol=symbol, start_date=start_date, end_date=end_date, doc_type=doc_type)
            except Exception as e:
                print("get_document()---", e)
                print(e)

        return stock_df
    