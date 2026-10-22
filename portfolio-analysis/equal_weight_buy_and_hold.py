
from AlgorithmImports import *


class EqualWeightBuyAndHold(QCAlgorithm):

    def initialize(self):
        self.set_start_date(2023, 1, 1)
        self.set_end_date(2026, 1, 1)
        self.set_cash(100000)

        tickers = [
            "AAPL", "MSFT", "JPM", "LLY", "TGT",
            "AOS", "NOG", "POR", "FDX", "PYPL"
        ]

        self.symbols = [
            self.add_equity(ticker, Resolution.DAILY).symbol
            for ticker in tickers
        ]

        self.invested_once = False

    def on_data(self, data):
        if self.invested_once:
            return

        if not all(
            self.securities[symbol].price > 0
            for symbol in self.symbols
        ):
            return

        weight = 1.0 / len(self.symbols)

        targets = [
            PortfolioTarget(symbol, weight)
            for symbol in self.symbols
        ]

        self.set_holdings(targets)
        self.invested_once = True
