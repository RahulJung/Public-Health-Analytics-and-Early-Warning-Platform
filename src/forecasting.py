import json
import os
import subprocess
import sys

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np


def _regularize_series(series: pd.Series) -> tuple[pd.Series, pd.Timedelta]:
    #     # 1. Infer the typical date spacing from observed records.
    # 2. Reindex to a regular time grid required by forecasting models.
    # 3. Fill small gaps through interpolation/forward/backward fill.
    series = series.sort_index()
    if len(series.index) < 2:
        return series, pd.Timedelta(days=1)

    day_deltas = series.index.to_series().diff().dropna().dt.days
    median_days = int(max(1, round(day_deltas.median()))) if not day_deltas.empty else 1
    step = pd.Timedelta(days=median_days)
    full_index = pd.date_range(series.index.min(), series.index.max(), freq=step)
    return series.reindex(full_index).interpolate().ffill().bfill(), step


def _future_index(series: pd.Series, step: pd.Timedelta, periods: int) -> pd.DatetimeIndex:
    # Build the forward forecast date index using the inferred series cadence.
    return pd.date_range(series.index.max() + step, periods=periods, freq=step)


def _forecast_arima(train: pd.Series, test: pd.Series, series: pd.Series, step: pd.Timedelta, periods: int):
    # Fit a lightweight ARIMA model and split output into backtest and future periods.
    order = (1, 1, 1)
    model = ARIMA(train, order=order)
    fitted = model.fit()
    forecast_values = fitted.forecast(steps=len(test) + periods)
    predictions = pd.Series(np.asarray(forecast_values[: len(test)]), index=test.index)
    future = pd.Series(np.asarray(forecast_values[len(test) :]), index=_future_index(series, step, periods))
    return predictions, future, {"model": "ARIMA", "arima_order": order}


def _forecast_prophet(train: pd.Series, test: pd.Series, series: pd.Series, step: pd.Timedelta, periods: int):
    #     # 1. Run Prophet in a subprocess to isolate local runtime/library instability.
    # 2. Send training values and forecast dates through JSON.
    # 3. Parse yhat values back into test and future forecast series.
    forecast_index = list(test.index) + list(_future_index(series, step, periods))
    payload = {
        "train": [{"ds": date.isoformat(), "y": float(value)} for date, value in train.items()],
        "forecast_dates": [date.isoformat() for date in forecast_index],
    }
    prophet_code = """
import json
import sys
import pandas as pd
from prophet import Prophet

payload = json.loads(sys.stdin.read())
train_df = pd.DataFrame(payload["train"])
forecast_dates = pd.DataFrame({"ds": pd.to_datetime(payload["forecast_dates"])})
model = Prophet(daily_seasonality=False, weekly_seasonality=False, yearly_seasonality=True)
model.fit(train_df)
forecast = model.predict(forecast_dates)
print(json.dumps(forecast["yhat"].tolist()))
"""
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", "/private/tmp")
    result = subprocess.run(
        [sys.executable, "-c", prophet_code],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
        env=env,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        error = f"Prophet exited with code {result.returncode}."
        if detail and "Matplotlib created a temporary cache directory" not in detail:
            error = f"{error} {detail}"
        raise RuntimeError(error)

    yhat = np.asarray(json.loads(result.stdout))
    predictions = pd.Series(yhat[: len(test)], index=test.index)
    future = pd.Series(yhat[len(test) :], index=_future_index(series, step, periods))
    return predictions, future, {"model": "Prophet"}


def _forecast_exponential_smoothing(
    train: pd.Series,
    test: pd.Series,
    series: pd.Series,
    step: pd.Timedelta,
    periods: int,
):
    # Fit Holt-style exponential smoothing as a stable statistical fallback model.
    model = ExponentialSmoothing(train, trend="add", seasonal=None, initialization_method="estimated")
    fitted = model.fit(optimized=True)
    forecast_values = fitted.forecast(len(test) + periods)
    predictions = pd.Series(np.asarray(forecast_values[: len(test)]), index=test.index)
    future = pd.Series(np.asarray(forecast_values[len(test) :]), index=_future_index(series, step, periods))
    return predictions, future, {"model": "Exponential Smoothing"}


def _forecast_rolling_average(train: pd.Series, test: pd.Series, series: pd.Series, step: pd.Timedelta, periods: int):
    # Use the latest rolling average when all model-based forecasting methods fail.
    rolling_value = train.rolling(7, min_periods=1).mean().iloc[-1]
    predictions = pd.Series([rolling_value] * len(test), index=test.index)
    future = pd.Series([rolling_value] * periods, index=_future_index(series, step, periods))
    return predictions, future, {"model": "Rolling Average"}


def forecast_series(
    df: pd.DataFrame,
    state: str,
    syndrome: str,
    periods: int = 14,
    model_name: str = "ARIMA",
):
    #     # 1. Filter to one state/syndrome time series.
    # 2. Require enough history for a meaningful backtest and forecast.
    # 3. Regularize the dates, split train/test, and run the requested model.
    # 4. Fall back to simpler models if the requested model fails.
    # 5. Return one dataframe containing historical, backtest, and future forecast rows.
    subset = df[(df["state"] == state) & (df["syndrome"] == syndrome)].copy()
    subset = subset.sort_values("date")
    if len(subset) < 20:
        return pd.DataFrame(), {"error": "Not enough records for forecasting."}

    series = subset.groupby("date")["visit_percentage"].mean()
    series, step = _regularize_series(series)

    train = series.iloc[:-min(periods, max(3, len(series)//5))]
    test = series.iloc[len(train):]

    requested_model = model_name
    forecasters = {
        "ARIMA": _forecast_arima,
        "Prophet": _forecast_prophet,
        "Exponential Smoothing": _forecast_exponential_smoothing,
    }
    forecast_fn = forecasters.get(model_name, _forecast_arima)

    try:
        predictions, future, model_info = forecast_fn(train, test, series, step, periods)
    except Exception as exc:
        try:
            predictions, future, model_info = _forecast_exponential_smoothing(train, test, series, step, periods)
            model_info["fallback_reason"] = str(exc)
        except Exception as fallback_exc:
            predictions, future, model_info = _forecast_rolling_average(train, test, series, step, periods)
            model_info["fallback_reason"] = f"{exc}; {fallback_exc}"

    metrics = {
        "mae": float(mean_absolute_error(test, predictions)) if len(test) else None,
        "rmse": float(np.sqrt(mean_squared_error(test, predictions))) if len(test) else None,
        "mape": float((np.abs((test - predictions) / test.replace(0, np.nan))).mean() * 100) if len(test) else None,
        "requested_model": requested_model,
        **model_info,
    }

    historical = pd.DataFrame({"date": series.index, "actual": series.values, "forecast": np.nan, "type": "historical"})
    predicted = pd.DataFrame({"date": predictions.index, "actual": test.values, "forecast": predictions.values, "type": "test_forecast"})
    future_df = pd.DataFrame({"date": future.index, "actual": np.nan, "forecast": future.values, "type": "future_forecast"})
    return pd.concat([historical, predicted, future_df], ignore_index=True), metrics
