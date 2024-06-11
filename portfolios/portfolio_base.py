
class PortfolioBase:
    name = None
    is_stock = False
    symbols = []
    
    def get_assets(self):
        return None
    
    def is_ready(self):
        return len(self.symbols) > 0