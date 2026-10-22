# ML SETUP
# ============================================================

ML_STOCKS = ["AAPL", "MSFT"]

LOOKBACK = 20

def create_ml_dataset(price_series, lookback=20):

    data = price_series.dropna().copy()

    X = []
    y = []
    dates = []

    values = data.values

    # Target = the close on day i, predicted from the 20 closes before it
    # (i-lookback ... i-1). This is a true NEXT-DAY forecast.
    for i in range(lookback, len(values)):

        X.append(values[i-lookback:i])

        y.append(values[i])

        dates.append(data.index[i])

    X = np.array(X)
    y = np.array(y)

    return X, y, dates

# ----- next cell -----

# ============================================================
# XGBOOST
# ============================================================

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

xgb_results = {}

for ticker in ML_STOCKS:

    print("Processing:", ticker)

    X, y, dates = create_ml_dataset(
        close_prices[ticker],
        LOOKBACK
    )

    split = int(len(X) * 0.80)

    X_train = X[:split]
    X_test = X[split:]

    y_train = y[:split]
    y_test = y[split:]

    test_dates = dates[split:]

    # --------------------------------------------------------
    # XGBoost model
    # --------------------------------------------------------

    model = XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    xgb_results[ticker] = {
        "model": model,
        "actual": y_test,
        "predicted": predictions,
        "dates": test_dates,
        "MAE": mae,
        "RMSE": rmse
    }

    print("MAE :", round(mae, 4))
    print("RMSE:", round(rmse, 4))
    print()

# ----- next cell -----

# ============================================================
# XGBOOST — ACTUAL VS FORECAST
# ============================================================

for ticker in ML_STOCKS:

    result = xgb_results[ticker]

    plt.figure(figsize=(14, 6))

    plt.plot(
        result["dates"],
        result["actual"],
        label="Actual Closing Price"
    )

    plt.plot(
        result["dates"],
        result["predicted"],
        label="XGBoost Forecast"
    )

    plt.title(
        f"{ticker}: Actual vs XGBoost Predicted Closing Price"
    )

    plt.xlabel("Date")
    plt.ylabel("Closing Price")

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.show()

# ----- next cell -----

# ============================================================
# LSTM
# ============================================================

from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# ----- next cell -----

# ============================================================
# LSTM DATA PREPARATION
# ============================================================

def create_lstm_dataset(price_series, lookback=20):

    data = price_series.dropna().values.reshape(-1, 1)

    scaler = MinMaxScaler()

    scaled_data = scaler.fit_transform(data)

    X = []
    y = []

    for i in range(
        lookback,
        len(scaled_data)
    ):

        X.append(
            scaled_data[i-lookback:i, 0]
        )

        y.append(
            scaled_data[i, 0]
        )

    X = np.array(X)
    y = np.array(y)

    X = X.reshape(
        X.shape[0],
        X.shape[1],
        1
    )

    return X, y, scaler

# ----- next cell -----

# ============================================================
# TRAIN LSTM MODELS
# ============================================================

lstm_results = {}

for ticker in ML_STOCKS:

    print("Processing:", ticker)

    X, y, scaler = create_lstm_dataset(
        close_prices[ticker],
        LOOKBACK
    )

    split = int(len(X) * 0.80)

    X_train = X[:split]
    X_test = X[split:]

    y_train = y[:split]
    y_test = y[split:]

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------

    model = Sequential([
        LSTM(
            64,
            return_sequences=True,
            input_shape=(
                LOOKBACK,
                1
            )
        ),

        Dropout(0.2),

        LSTM(32),

        Dropout(0.2),

        Dense(1)
    ])

    model.compile(
        optimizer="adam",
        loss="mean_squared_error"
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    model.fit(
        X_train,
        y_train,
        epochs=20,
        batch_size=32,
        validation_split=0.1,
        shuffle=False,
        verbose=1
    )

    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    predictions_scaled = model.predict(
        X_test,
        verbose=0
    )

    predictions = scaler.inverse_transform(
        predictions_scaled
    ).flatten()

    actual = scaler.inverse_transform(
        y_test.reshape(-1, 1)
    ).flatten()

    mae = mean_absolute_error(
        actual,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predictions
        )
    )

    # Corresponding dates
    price_dates = close_prices[
        ticker
    ].dropna().index

    # Sample k predicts the close on day LOOKBACK + k, so the first
    # test sample is day LOOKBACK + split (same dates as the XGBoost test set)
    dates = price_dates[
        LOOKBACK + split:
        LOOKBACK + split + len(actual)
    ]

    lstm_results[ticker] = {
        "model": model,
        "actual": actual,
        "predicted": predictions,
        "dates": dates,
        "MAE": mae,
        "RMSE": rmse
    }

    print("MAE :", round(mae, 4))
    print("RMSE:", round(rmse, 4))
    print()

# ----- next cell -----

# ============================================================
# LSTM — ACTUAL VS FORECAST
# ============================================================

for ticker in ML_STOCKS:

    result = lstm_results[ticker]

    plt.figure(figsize=(14, 6))

    plt.plot(
        result["dates"],
        result["actual"],
        label="Actual Closing Price"
    )

    plt.plot(
        result["dates"],
        result["predicted"],
        label="LSTM Forecast"
    )

    plt.title(
        f"{ticker}: Actual vs LSTM Predicted Closing Price"
    )

    plt.xlabel("Date")
    plt.ylabel("Closing Price")

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.show()

# ----- next cell -----

# ============================================================
# XGBOOST VS LSTM
# ============================================================

ml_comparison = []

for ticker in ML_STOCKS:

    ml_comparison.append({
        "Security": ticker,
        "XGBoost MAE": xgb_results[ticker]["MAE"],
        "XGBoost RMSE": xgb_results[ticker]["RMSE"],
        "LSTM MAE": lstm_results[ticker]["MAE"],
        "LSTM RMSE": lstm_results[ticker]["RMSE"]
    })

ml_comparison = pd.DataFrame(ml_comparison)

display(ml_comparison)
