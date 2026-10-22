# region imports
from AlgorithmImports import *
# endregion

import numpy as np
import pandas as pd
from datetime import datetime
from scipy.optimize import minimize


class MinimumVarianceMonthly(QCAlgorithm):
    """
    Q2(ii): Long-only minimum-variance portfolio, rebalanced on the
    first trading day of every month.

    - Universe: the 10 assignment securities
    - Constraint: weights sum to 1, each weight >= 5% (long only)
    - Estimation window: previous 252 daily returns (1 trading year)
    - Only data from 1 Jan 2023 onward is used (assignment data window).
      Until 252 daily returns exist, the portfolio is equal-weighted
      (invested as soon as all 10 stocks have a price).
    """

    def initialize(self):

        # ------------------------------------------------------------
        # 1. Backtest period and capital
        # ------------------------------------------------------------
        self.set_start_date(2023, 1, 1)
        self.set_end_date(2026, 1, 1)
        self.set_cash(100000)

        # Do not let QuantConnect silently skip small rebalance trades
        self.settings.minimum_order_margin_portfolio_percentage = 0

        # ------------------------------------------------------------
        # 2. Parameters (easy to change in one place)
        # ------------------------------------------------------------
        self.window = 252                      # estimation window (days)
        self.min_weight = 0.05                 # minimum holding per stock
        self.data_start = datetime(2023, 1, 1) # no pre-2023 data is used

        # ------------------------------------------------------------
        # 3. Securities (daily resolution is enough: we trade monthly)
        # ------------------------------------------------------------
        self.tickers = [
            "AAPL", "MSFT", "JPM", "LLY", "TGT",
            "AOS", "NOG", "POR", "FDX", "PYPL"
        ]

        self.symbols = [
            self.add_equity(ticker, Resolution.DAILY).symbol
            for ticker in self.tickers
        ]

        # Becomes True once the initial equal-weight portfolio is bought
        self.initial_invested = False

        # ------------------------------------------------------------
        # 4. Monthly rebalance: first trading day, 5 minutes after open
        # ------------------------------------------------------------
        self.schedule.on(
            self.date_rules.month_start(self.symbols[0]),
            self.time_rules.after_market_open(self.symbols[0], 5),
            self.rebalance
        )

    # ==================================================================
    # INITIAL EQUAL-WEIGHT INVESTMENT (same start as Q2(i))
    # ==================================================================
    def on_data(self, data):

        if self.initial_invested:
            return

        # Wait until every stock has a valid price
        if not all(self.securities[s].price > 0 for s in self.symbols):
            return

        weight = 1.0 / len(self.symbols)
        self.set_holdings(
            [PortfolioTarget(symbol, weight) for symbol in self.symbols]
        )
        self.initial_invested = True
        self.debug(f"{self.time.date()} | initial equal-weight portfolio")

    # ==================================================================
    # MONTHLY REBALANCE
    # ==================================================================
    def rebalance(self):

        if not self.initial_invested:
            return

        current_date = self.time.date()

        # ------------------------------------------------------------
        # Daily history from 1 Jan 2023 up to now. At 9:35 the latest
        # bar is yesterday's, so there is no look-ahead.
        # ------------------------------------------------------------
        history = self.history(
            self.symbols,
            self.data_start,
            self.time,
            Resolution.DAILY
        )

        if history.empty:
            return

        # Rows = dates, columns = stocks
        close_prices = history["close"].unstack(level=0)
        close_prices.columns = [
            str(column).split(" ")[-1] for column in close_prices.columns
        ]

        if any(t not in close_prices.columns for t in self.tickers):
            return

        close_prices = close_prices[self.tickers]
        daily_returns = close_prices.pct_change().dropna()

        # Not enough data yet -> keep the equal-weight portfolio
        if len(daily_returns) < self.window:
            return

        # Most recent 252 daily returns
        estimation_window = daily_returns.iloc[-self.window:]

        # ------------------------------------------------------------
        # Minimum-variance optimisation
        # ------------------------------------------------------------
        weights = self.minimum_variance_weights(estimation_window)

        if weights is None:
            # Keep existing holdings if the optimiser fails
            return

        targets = [
            PortfolioTarget(symbol, weights[symbol])
            for symbol in self.symbols
        ]

        self.set_holdings(targets)

        weight_text = ", ".join(
            f"{ticker}: {weights[symbol]:.3f}"
            for ticker, symbol in zip(self.tickers, self.symbols)
        )
        self.debug(f"Rebalanced {current_date} | {weight_text}")

    # ==================================================================
    # MINIMUM-VARIANCE WEIGHTS (long only, min 5% each, sum = 1)
    # ==================================================================
    def minimum_variance_weights(self, returns):

        n = len(self.tickers)

        # Daily-return covariances are ~1e-4, so the objective is
        # scaled by 1e6 (does not change the minimiser).
        covariance_scaled = returns.cov().values * 1e6

        # Objective: portfolio variance w' C w
        def portfolio_variance(w):
            return w @ covariance_scaled @ w

        # Analytic gradient 2 C w. Supplying it makes SLSQP far more
        # reliable than numerical differentiation.
        def gradient(w):
            return 2.0 * covariance_scaled @ w

        bounds = [(self.min_weight, 1.0) for _ in range(n)]

        constraint = {
            "type": "eq",
            "fun": lambda w: np.sum(w) - 1,
            "jac": lambda w: np.ones(n)
        }

        initial_weights = np.ones(n) / n

        # Try progressively looser tolerances; extremely tight ones
        # (e.g. 1e-12) make SLSQP fail with "Positive directional
        # derivative for linesearch".
        last_message = ""
        for ftol in (1e-9, 1e-8, 1e-6):

            result = minimize(
                portfolio_variance,
                initial_weights,
                jac=gradient,
                method="SLSQP",
                bounds=bounds,
                constraints=constraint,
                options={"ftol": ftol, "maxiter": 1000}
            )

            last_message = result.message

            # Accept only if the solution truly satisfies the constraints
            if (
                result.success
                and abs(np.sum(result.x) - 1) < 1e-6
                and np.min(result.x) >= self.min_weight - 1e-8
            ):
                return {
                    symbol: float(weight)
                    for symbol, weight in zip(self.symbols, result.x)
                }

        self.debug(
            f"Optimisation failed on {self.time.date()}: {last_message}"
        )
        return None
