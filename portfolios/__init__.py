from .portfolio_stocks import PortfolioStocks
from .portfolio_general import PortfolioGeneral

def get_portfolio(portfolio_name, symbolsDate_dict):
    if portfolio_name == 'stocks':
        return PortfolioStocks(symbolsDate_dict)
    elif portfolio_name == 'general':
        return PortfolioGeneral(symbolsDate_dict)
    else:
        raise ValueError("Portfolio not found")
    
def get_list_of_portfolios():
    return ['stocks', 'general']