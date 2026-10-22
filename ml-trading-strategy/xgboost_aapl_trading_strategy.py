from AlgorithmImports import *

import numpy as np
import pandas as pd
import xgboost as xgb
from datetime import datetime


class Q1IV_XGBoostStrategy(QCAlgorithm):

    def initialize(self):

        # ============================================================
        # 1. BACKTEST PERIOD
        # ============================================================

        self.set_start_date(2023, 1, 1)
        self.set_end_date(2026, 1, 1)

        # Assignment: initial cash = $100,000
        self.set_cash(100000)

        # Margin account (lecture: cash account vs margin account).
        # Short selling needs a margin account; the account equity is the
        # collateral. Stated explicitly so the setup is clear.
        self.set_brokerage_model(
            BrokerageName.DEFAULT,
            AccountType.MARGIN
        )

        # ============================================================
        # 2. SECURITY
        # ============================================================

        # AAPL is selected because it was one of the securities
        # used in Q1(iii).

        self.aapl = self.add_equity(
            "AAPL",
            Resolution.MINUTE,
            extended_market_hours=True
        ).symbol

        # ============================================================
        # 3. ASSIGNMENT PARAMETERS
        # ============================================================

        # $50,000 is reserved as margin/capital reserve.
        # $50,000 is the maximum trading capital.

        self.margin_reserve = 50000
        self.trading_capital = 50000

        # Prediction threshold = +/- 0.5%
        self.threshold = 0.005

        # XGBoost feature window:
        # previous 20 daily closing prices
        self.lookback = 20

        # Maximum number of historical observations used
        # to train the model.
        self.training_days = 504

        # The assignment data window begins here.
        self.data_start = datetime(2023, 1, 1)

        # ============================================================
        # 4. MODEL / SIGNAL VARIABLES
        # ============================================================

        self.model = None

        self.yesterday_close = None
        self.predicted_close = None

        # Signal can be:
        # "LONG", "SHORT", or "CASH"
        self.signal = "CASH"

        # ============================================================
        # 5. ORDER TICKETS
        # ============================================================

        self.long_entry_ticket = None
        self.short_entry_ticket = None

        self.long_exit_ticket = None

        # Actual price at which long position was opened
        self.long_entry_price = None

        # ============================================================
        # 6. TRAIN MODEL BEFORE MARKET OPEN
        # ============================================================

        # Train 30 minutes before market open.
        self.schedule.on(
            self.date_rules.every_day(self.aapl),
            self.time_rules.before_market_open(
                self.aapl,
                30
            ),
            self.train_xgboost
        )

        # ============================================================
        # 7. SUBMIT MARKET-ON-OPEN ORDER
        # ============================================================

        # Submit MOO order 5 minutes before market open.
        # This is deliberately separate from model training.
        self.schedule.on(
            self.date_rules.every_day(self.aapl),
            self.time_rules.before_market_open(
                self.aapl,
                5
            ),
            self.place_opening_order
        )

        # ============================================================
        # 8. CLOSE SHORT POSITION BEFORE MARKET CLOSE
        # ============================================================

        self.schedule.on(
            self.date_rules.every_day(self.aapl),
            self.time_rules.before_market_close(
                self.aapl,
                5
            ),
            self.close_short_position
        )

    # =================================================================
    # 9. TRAIN XGBOOST AND GENERATE SIGNAL
    # =================================================================

    def train_xgboost(self):

        try:

            # --------------------------------------------------------
            # Get daily data ONLY from 1 Jan 2023 onward.
            # This prevents accidental use of pre-assignment data.
            # --------------------------------------------------------

            history = self.history(
                self.aapl,
                self.data_start,
                self.time,
                Resolution.DAILY
            )

            if history.empty:
                self.debug(
                    f"{self.time} | No historical data available."
                )
                self.signal = "CASH"
                return

            closes = history["close"].dropna()

            # Keep at most the latest 504 observations.
            closes = closes.tail(self.training_days)

            # Need enough observations to create
            # 20-day features.
            if len(closes) < self.lookback + 20:

                self.debug(
                    f"{self.time} | "
                    f"Insufficient training observations: "
                    f"{len(closes)}"
                )

                self.signal = "CASH"
                return

            # --------------------------------------------------------
            # Last completed daily close = yesterday's close
            # --------------------------------------------------------

            self.yesterday_close = float(
                closes.iloc[-1]
            )

            values = closes.values.astype(float)

            # --------------------------------------------------------
            # Create X and y
            #
            # X = previous 20 daily closing prices
            # y = following day's closing price
            # --------------------------------------------------------

            X = []
            y = []

            for i in range(
                self.lookback,
                len(values)
            ):

                features = values[
                    i - self.lookback:i
                ]

                target = values[i]

                X.append(features)
                y.append(target)

            X = np.array(X)
            y = np.array(y)

            # --------------------------------------------------------
            # XGBoost model
            # --------------------------------------------------------

            self.model = xgb.XGBRegressor(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="reg:squarederror",
                random_state=42
            )

            self.model.fit(X, y)

            # --------------------------------------------------------
            # Predict next/current trading day's closing price
            # using the latest 20 completed daily closes.
            # --------------------------------------------------------

            latest_features = values[
                -self.lookback:
            ].reshape(1, -1)

            self.predicted_close = float(
                self.model.predict(
                    latest_features
                )[0]
            )

            # --------------------------------------------------------
            # Calculate predicted return
            # --------------------------------------------------------

            predicted_return = (
                self.predicted_close /
                self.yesterday_close
                - 1
            )

            # --------------------------------------------------------
            # Generate trading signal
            # --------------------------------------------------------

            if predicted_return >= self.threshold:

                self.signal = "LONG"

            elif predicted_return <= -self.threshold:

                self.signal = "SHORT"

            else:

                self.signal = "CASH"

            # --------------------------------------------------------
            # Debug information
            # --------------------------------------------------------

            self.debug(
                f"{self.time.date()} | "
                f"Yesterday Close = "
                f"{self.yesterday_close:.2f} | "
                f"XGB Prediction = "
                f"{self.predicted_close:.2f} | "
                f"Predicted Return = "
                f"{predicted_return:.4%} | "
                f"Signal = {self.signal}"
            )

        except Exception as e:

            self.signal = "CASH"

            self.debug(
                f"{self.time} | "
                f"XGBoost error: {str(e)}"
            )

    # =================================================================
    # 10. PLACE MARKET-ON-OPEN ORDER
    # =================================================================

    def place_opening_order(self):

        # No valid model/signal
        if self.model is None:
            return

        # Cash signal → no trade
        if self.signal == "CASH":
            return

        # Do not open another position if already invested.
        if self.portfolio[self.aapl].invested:
            return

        # Prevent duplicate opening orders.
        if (
            self.long_entry_ticket is not None
            or self.short_entry_ticket is not None
        ):
            return

        if self.yesterday_close is None:
            return

        # ------------------------------------------------------------
        # Estimate quantity using $50,000 trading capital.
        #
        # We use a 5% price buffer because the actual MOO fill price
        # is not known until the opening auction.
        # ------------------------------------------------------------

        estimated_open_price = (
            self.yesterday_close * 1.05
        )

        quantity = int(
            self.trading_capital /
            estimated_open_price
        )

        if quantity <= 0:
            return

        # ============================================================
        # LONG
        # ============================================================

        if self.signal == "LONG":

            self.long_entry_ticket = (
                self.market_on_open_order(
                    self.aapl,
                    quantity,
                    tag="Q1IV LONG ENTRY"
                )
            )

            self.debug(
                f"{self.time} | "
                f"LONG MOO submitted | "
                f"Quantity = {quantity}"
            )

        # ============================================================
        # SHORT
        # ============================================================

        elif self.signal == "SHORT":

            self.short_entry_ticket = (
                self.market_on_open_order(
                    self.aapl,
                    -quantity,
                    tag="Q1IV SHORT ENTRY"
                )
            )

            self.debug(
                f"{self.time} | "
                f"SHORT MOO submitted | "
                f"Quantity = {-quantity}"
            )

    # =================================================================
    # 11. HANDLE ORDER FILLS
    # =================================================================

    def on_order_event(self, order_event):

        # Only act when the order is actually filled.
        if order_event.status != OrderStatus.FILLED:
            return

        # ============================================================
        # LONG ENTRY FILLED
        # ============================================================

        if (
            self.long_entry_ticket is not None
            and
            order_event.order_id ==
            self.long_entry_ticket.order_id
        ):

            self.long_entry_price = float(
                order_event.fill_price
            )

            quantity = abs(
                order_event.fill_quantity
            )

            # --------------------------------------------------------
            # Take-profit price:
            #
            # Entry price + 0.5%
            # --------------------------------------------------------

            # Rounded UP to the nearest cent (stock prices move in cents),
            # so the target is never below the +0.5% take-profit level.
            target_price = float(
                np.ceil(
                    self.long_entry_price *
                    (1 + self.threshold) * 100
                ) / 100
            )

            self.long_exit_ticket = (
                self.limit_order(
                    self.aapl,
                    -quantity,
                    target_price,
                    tag="Q1IV LONG TAKE PROFIT"
                )
            )

            self.debug(
                f"{self.time} | "
                f"LONG FILLED | "
                f"Entry = {self.long_entry_price:.2f} | "
                f"Target = {target_price:.2f}"
            )

        # ============================================================
        # LONG TAKE-PROFIT FILLED
        # ============================================================

        elif (
            self.long_exit_ticket is not None
            and
            order_event.order_id ==
            self.long_exit_ticket.order_id
        ):

            self.debug(
                f"{self.time} | "
                f"LONG TAKE PROFIT FILLED | "
                f"Exit = {order_event.fill_price:.2f}"
            )

            self.long_exit_ticket = None
            self.long_entry_ticket = None
            self.long_entry_price = None

        # ============================================================
        # SHORT ENTRY FILLED
        # ============================================================

        elif (
            self.short_entry_ticket is not None
            and
            order_event.order_id ==
            self.short_entry_ticket.order_id
        ):

            self.debug(
                f"{self.time} | "
                f"SHORT FILLED | "
                f"Entry = {order_event.fill_price:.2f}"
            )

    # =================================================================
    # 12. CLOSE SHORT POSITION SAME DAY
    # =================================================================

    def close_short_position(self):

        # If there is a short position, close it.
        if (
            self.portfolio[self.aapl].invested
            and
            self.portfolio[self.aapl].quantity < 0
        ):

            # Cancel any outstanding orders first.
            self.transactions.cancel_open_orders(
                self.aapl
            )

            # Close the short position.
            self.liquidate(
                self.aapl,
                tag="Q1IV SHORT SAME-DAY SETTLEMENT"
            )

            self.debug(
                f"{self.time} | "
                f"SHORT POSITION CLOSED "
                f"FOR SAME-DAY SETTLEMENT"
            )

        # Reset short entry ticket.
        self.short_entry_ticket = None

    # =================================================================
    # 13. END-OF-DAY CLEANUP
    # =================================================================

    def on_end_of_day(self, symbol):

        if symbol != self.aapl:
            return

        # The daily signal should only apply to that trading day.
        self.signal = "CASH"

        # ------------------------------------------------------------
        # IMPORTANT:
        #
        # We DO NOT reset long_entry_ticket or long_exit_ticket here.
        #
        # Why?
        #
        # The assignment says:
        # "hold and sell as soon as price goes above 0.5%
        # of buying price in future."
        #
        # Therefore, if the long target is not reached today,
        # the position and its limit sell order must remain active
        # into subsequent days.
        # ------------------------------------------------------------
