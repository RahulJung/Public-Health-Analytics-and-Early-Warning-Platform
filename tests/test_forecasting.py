import pandas as pd

from src.forecasting import forecast_series


def test_forecast_series_preserves_weekly_cadence():
    # A weekly input series should produce future forecast dates at weekly intervals.
    df = pd.DataFrame({
        "date": pd.date_range("2026-01-03", periods=24, freq="7D"),
        "state": ["Georgia"] * 24,
        "syndrome": ["COVID-like illness"] * 24,
        "visit_percentage": [2 + i * 0.1 for i in range(24)],
    })

    forecast_df, metrics = forecast_series(df, "Georgia", "COVID-like illness", periods=4, model_name="ARIMA")

    future = forecast_df[forecast_df["type"] == "future_forecast"]
    assert len(future) == 4
    assert future["date"].diff().dropna().dt.days.eq(7).all()
    assert metrics["requested_model"] == "ARIMA"
    assert metrics["model"] in {"ARIMA", "Exponential Smoothing", "Rolling Average"}
    assert "mae" in metrics


def test_forecast_series_handles_unavailable_prophet():
    # Prophet selection should still return output through fallback if Prophet cannot run locally.
    df = pd.DataFrame({
        "date": pd.date_range("2026-01-03", periods=24, freq="7D"),
        "state": ["Georgia"] * 24,
        "syndrome": ["COVID-like illness"] * 24,
        "visit_percentage": [2 + i * 0.1 for i in range(24)],
    })

    forecast_df, metrics = forecast_series(df, "Georgia", "COVID-like illness", periods=4, model_name="Prophet")

    assert not forecast_df.empty
    assert metrics["requested_model"] == "Prophet"
    assert metrics["model"] in {"Prophet", "Exponential Smoothing", "Rolling Average"}
