import json
import os
import subprocess
import sys

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np


def _clip_forecast(values: pd.Series | np.ndarray) -> np.ndarray:
    # Surveillance percentages should not be negative.
    return np.clip(np.asarray(values, dtype=float), 0, None)


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
    # Select the ARIMA order with the lowest holdout error instead of using a fixed order.
    candidate_orders = [(0, 1, 1), (1, 1, 0), (1, 1, 1), (2, 1, 1), (1, 0, 1), (2, 0, 1)]
    candidates = []
    for order in candidate_orders:
        try:
            fitted = ARIMA(train, order=order).fit()
            forecast_values = _clip_forecast(fitted.forecast(steps=len(test) + periods))
            predictions = pd.Series(forecast_values[: len(test)], index=test.index)
            future = pd.Series(forecast_values[len(test) :], index=_future_index(series, step, periods))
            candidates.append((predictions, future, {"model": "ARIMA", "arima_order": order}))
        except Exception:
            continue
    if not candidates:
        raise RuntimeError("No ARIMA candidate could be fitted.")
    return _select_best_candidate(candidates, test)


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
    yhat = _clip_forecast(yhat)
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
    # Compare several ETS-style variants and keep the one with the best holdout error.
    candidates = []
    model_specs = [
        {"trend": None, "damped_trend": False},
        {"trend": "add", "damped_trend": False},
        {"trend": "add", "damped_trend": True},
    ]
    for spec in model_specs:
        try:
            model = ExponentialSmoothing(train, seasonal=None, initialization_method="estimated", **spec)
            fitted = model.fit(optimized=True)
            forecast_values = _clip_forecast(fitted.forecast(len(test) + periods))
            predictions = pd.Series(forecast_values[: len(test)], index=test.index)
            future = pd.Series(forecast_values[len(test) :], index=_future_index(series, step, periods))
            candidates.append((predictions, future, {"model": "Exponential Smoothing", **spec}))
        except Exception:
            continue
    if not candidates:
        raise RuntimeError("No Exponential Smoothing candidate could be fitted.")
    return _select_best_candidate(candidates, test)


def _forecast_rolling_average(train: pd.Series, test: pd.Series, series: pd.Series, step: pd.Timedelta, periods: int):
    # Use the latest short rolling average when all model-based forecasting methods fail.
    rolling_value = train.rolling(min(7, len(train)), min_periods=1).mean().iloc[-1]
    predictions = pd.Series([rolling_value] * len(test), index=test.index)
    future = pd.Series([rolling_value] * periods, index=_future_index(series, step, periods))
    return predictions, future, {"model": "Rolling Average"}


def _forecast_naive(train: pd.Series, test: pd.Series, series: pd.Series, step: pd.Timedelta, periods: int):
    # A persistence baseline is often hard to beat for short noisy surveillance series.
    latest_value = float(train.iloc[-1])
    predictions = pd.Series([latest_value] * len(test), index=test.index)
    future = pd.Series([latest_value] * periods, index=_future_index(series, step, periods))
    return predictions, future, {"model": "Naive Baseline"}


def _forecast_drift(train: pd.Series, test: pd.Series, series: pd.Series, step: pd.Timedelta, periods: int):
    # Linear drift baseline using the first and latest training observations.
    if len(train) < 2:
        return _forecast_naive(train, test, series, step, periods)
    slope = (float(train.iloc[-1]) - float(train.iloc[0])) / max(1, len(train) - 1)
    values = _clip_forecast([float(train.iloc[-1]) + slope * i for i in range(1, len(test) + periods + 1)])
    predictions = pd.Series(values[: len(test)], index=test.index)
    future = pd.Series(values[len(test) :], index=_future_index(series, step, periods))
    return predictions, future, {"model": "Drift Baseline"}


def _forecast_moving_average(train: pd.Series, test: pd.Series, series: pd.Series, step: pd.Timedelta, periods: int):
    # Compare multiple rolling windows because syndrome series can be smooth or noisy.
    candidates = []
    for window in [3, 5, 7, 14]:
        if len(train) < window:
            continue
        value = float(train.rolling(window, min_periods=1).mean().iloc[-1])
        predictions = pd.Series([value] * len(test), index=test.index)
        future = pd.Series([value] * periods, index=_future_index(series, step, periods))
        candidates.append((predictions, future, {"model": "Moving Average", "window": window}))
    if not candidates:
        return _forecast_rolling_average(train, test, series, step, periods)
    return _select_best_candidate(candidates, test)


def _safe_mape(actual: pd.Series, predicted: pd.Series) -> float | None:
    denominator = actual.replace(0, np.nan).abs()
    valid = denominator > 1e-6
    if not valid.any():
        return None
    return float((np.abs((actual[valid] - predicted[valid]) / denominator[valid])).mean() * 100)


def _smape(actual: pd.Series, predicted: pd.Series) -> float | None:
    denominator = (actual.abs() + predicted.abs()) / 2
    valid = denominator > 1e-6
    if not valid.any():
        return None
    return float((np.abs(actual[valid] - predicted[valid]) / denominator[valid]).mean() * 100)


def _wmape(actual: pd.Series, predicted: pd.Series) -> float | None:
    denominator = float(actual.abs().sum())
    if denominator <= 1e-6:
        return None
    return float(np.abs(actual - predicted).sum() / denominator * 100)


def _candidate_score(actual: pd.Series, predicted: pd.Series) -> tuple[float, float]:
    # Prefer lower MAE, with sMAPE as a tie-breaker for scale-aware accuracy.
    mae = float(mean_absolute_error(actual, predicted))
    smape = _smape(actual, predicted)
    return mae, smape if smape is not None else float("inf")


def _select_best_candidate(candidates: list[tuple[pd.Series, pd.Series, dict]], test: pd.Series):
    scored = [
        (_candidate_score(test, predictions), predictions, future, info)
        for predictions, future, info in candidates
    ]
    _, predictions, future, info = min(scored, key=lambda item: item[0])
    return predictions, future, info


def _forecast_auto(train: pd.Series, test: pd.Series, series: pd.Series, step: pd.Timedelta, periods: int):
    # Choose the best available model family using the same holdout window used for displayed metrics.
    candidates = []
    for forecast_fn in [
        _forecast_arima,
        _forecast_exponential_smoothing,
        _forecast_moving_average,
        _forecast_naive,
        _forecast_drift,
    ]:
        try:
            candidates.append(forecast_fn(train, test, series, step, periods))
        except Exception:
            continue
    if not candidates:
        raise RuntimeError("No automatic forecasting candidate could be fitted.")
    predictions, future, info = _select_best_candidate(candidates, test)
    info["selection_method"] = "best holdout MAE with sMAPE tie-breaker"
    return predictions, future, info


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
    selected_state = state
    selected_syndrome = syndrome
    if len(subset) < 20:
        syndrome_subset = df[df["syndrome"] == syndrome].copy()
        counts = syndrome_subset.groupby("state").size().sort_values(ascending=False)
        if not counts.empty and counts.iloc[0] >= 20:
            selected_state = counts.index[0]
            subset = syndrome_subset[syndrome_subset["state"] == selected_state].copy()
        else:
            counts = df.groupby(["state", "syndrome"]).size().sort_values(ascending=False)
            if not counts.empty and counts.iloc[0] >= 20:
                selected_state, selected_syndrome = counts.index[0]
                subset = df[(df["state"] == selected_state) & (df["syndrome"] == selected_syndrome)].copy()
    subset = subset.sort_values("date")
    if len(subset) < 20:
        return pd.DataFrame(), {"error": "Not enough records for forecasting."}

    series = subset.groupby("date")["visit_percentage"].mean()
    series, step = _regularize_series(series)

    train = series.iloc[:-min(periods, max(3, len(series)//5))]
    test = series.iloc[len(train):]

    requested_model = model_name
    forecasters = {
        "Auto (Best Backtest)": _forecast_auto,
        "ARIMA": _forecast_arima,
        "Prophet": _forecast_prophet,
        "Exponential Smoothing": _forecast_exponential_smoothing,
        "Moving Average": _forecast_moving_average,
    }
    forecast_fn = forecasters.get(model_name, _forecast_auto)

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
        "mape": _safe_mape(test, predictions) if len(test) else None,
        "smape": _smape(test, predictions) if len(test) else None,
        "wmape": _wmape(test, predictions) if len(test) else None,
        "requested_model": requested_model,
        "state": selected_state,
        "syndrome": selected_syndrome,
        **model_info,
    }

    historical = pd.DataFrame({"date": series.index, "actual": series.values, "forecast": np.nan, "type": "historical"})
    predicted = pd.DataFrame({"date": predictions.index, "actual": test.values, "forecast": predictions.values, "type": "test_forecast"})
    future_df = pd.DataFrame({"date": future.index, "actual": np.nan, "forecast": future.values, "type": "future_forecast"})
    return pd.concat([historical, predicted, future_df], ignore_index=True), metrics
