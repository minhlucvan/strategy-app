
from portfolios.portfolio_base import PortfolioBase
from utils.processing import get_stocks


class PortfolioStocks(PortfolioBase):
    name = None
    is_stock = False
    symbolDate_dict = {}
    stocks_df = None
    
    def __init__(self, symbolDate_dict) -> None:
        super().__init__()
        self.symbolDate_dict = symbolDate_dict
        
    
    def get_assets(self):
        self.stocks_df = get_stocks(self.symbolDate_dict, 'close')