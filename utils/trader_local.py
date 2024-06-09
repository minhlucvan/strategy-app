from utils.trader_base import TraderBase


class TraderLocal(TraderBase):
    def __init__(self):
        self.api = None