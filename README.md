# QuantConnect ML Trading & Portfolio Optimization

An Investment Theory project exploring machine learning-based stock price forecasting, systematic trading strategies, and portfolio optimization using Python and QuantConnect.

## Project Overview

This project combines predictive modelling and quantitative portfolio management to examine stock price behaviour, develop trading strategies, and compare portfolio construction approaches.

## Key Components

### 1. Stock Price Forecasting
- XGBoost and Long Short-Term Memory (LSTM) models for stock price forecasting.
- Forecasting analysis for Apple (AAPL) and Microsoft (MSFT).
- Model evaluation using Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).

### 2. Machine Learning Trading Strategy
- XGBoost-based trading strategy for Apple (AAPL).
- Prediction-driven trading signals using defined entry thresholds.
- Backtesting through QuantConnect.

### 3. Portfolio Analysis
- **Equal-Weight Buy-and-Hold:** Portfolio performance evaluation against a market benchmark.
- **Minimum-Variance Portfolio:** Long-only portfolio optimization with monthly rebalancing and a minimum 5% allocation per asset.

## Tools & Technologies

Python · QuantConnect · XGBoost · LSTM · Pandas · NumPy · Portfolio Optimization

## Repository Structure

- `stock-price-forecasting/` — Stock price forecasting models.
- `ml-trading-strategy/` — XGBoost-based AAPL trading strategy.
- `portfolio-analysis/` — Equal-weight and minimum-variance portfolio implementations.

## Project Context

Developed as part of an Investment Theory project, applying machine learning and quantitative methods to trading and portfolio management.

*Note: Backtest performance is historical and does not guarantee future investment returns.*
