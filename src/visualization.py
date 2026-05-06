import plotly.express as px
import plotly.graph_objects as go
import pandas as pd


def plot_trend_with_anomalies(df: pd.DataFrame, state: str, syndrome: str):
    #     # 1. Filter to one state/syndrome trend.
    # 2. Plot observed visit percentage and rolling baseline.
    # 3. Overlay anomaly markers for analyst review.
    subset = df[(df["state"] == state) & (df["syndrome"] == syndrome)].sort_values("date")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=subset["date"], y=subset["visit_percentage"], mode="lines", name="Visit %"))
    fig.add_trace(go.Scatter(x=subset["date"], y=subset["rolling_mean"], mode="lines", name="Rolling baseline"))
    anomalies = subset[subset["any_anomaly"]]
    fig.add_trace(go.Scatter(x=anomalies["date"], y=anomalies["visit_percentage"], mode="markers", name="Anomaly", marker={"size": 10}))
    fig.update_layout(title=f"{syndrome} ED Visit Trend - {state}", xaxis_title="Date", yaxis_title="Visit Percentage")
    return fig


def plot_risk_table(df: pd.DataFrame):
    # Return the highest-risk latest state/syndrome rows as a compact table.
    latest = df.sort_values("date").groupby(["state", "syndrome"], as_index=False).tail(1)
    latest = latest.sort_values("risk_score", ascending=False).head(25)
    return latest[["date", "state", "syndrome", "visit_percentage", "risk_score", "risk_level", "any_anomaly"]]


def plot_national_overview(df: pd.DataFrame):
    # Aggregate state-level records into national average syndrome trends.
    overview = df.groupby(["date", "syndrome"], as_index=False)["visit_percentage"].mean()
    return px.line(overview, x="date", y="visit_percentage", color="syndrome", title="Average ED Visit Percentage by Syndrome")


def plot_forecast(forecast_df: pd.DataFrame, title: str):
    #     # 1. Split forecast output into historical, backtest, and future rows.
    # 2. Plot each segment as a separate trace for clear interpretation.
    fig = go.Figure()
    historical = forecast_df[forecast_df["type"] == "historical"]
    test_forecast = forecast_df[forecast_df["type"] == "test_forecast"]
    future = forecast_df[forecast_df["type"] == "future_forecast"]
    fig.add_trace(go.Scatter(x=historical["date"], y=historical["actual"], mode="lines", name="Actual"))
    if not test_forecast.empty:
        fig.add_trace(go.Scatter(x=test_forecast["date"], y=test_forecast["forecast"], mode="lines", name="Backtest forecast"))
    if not future.empty:
        fig.add_trace(go.Scatter(x=future["date"], y=future["forecast"], mode="lines", name="Future forecast"))
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title="Visit Percentage")
    return fig
