class TraderBase:
    def __init__(self, config: dict):
        self.config = config
        self.api = None

    def place_preorder(*args, **kwargs):
        raise NotImplementedError()
    
    def use_account(*args, **kwargs):
        pass
    
    def get_account_list(*args, **kwargs):
        return []
    
    def get_pending_orders(*args, **kwargs):
        return []