import numpy as np
import pandas as pd


def _scale(series: pd.Series, max_value: float) -> pd.Series:
    # Normalize a nonnegative signal to a 0-1 range with an upper cap.
    values = series.fillna(0).clip(lower=0)
    return (values / max_value).clip(0, 1)


def assign_risk_scores(df: pd.DataFrame) -> pd.DataFrame:
    #     # 1. Convert statistical severity, growth, ML anomaly flags, and persistence into weighted components.
    # 2. Sum components into a bounded 0-100 risk score.
    # 3. Map numeric score ranges to Low/Moderate/High/Critical labels.
    df = df.copy()

    z_component = _scale(df.get("zscore_severity", pd.Series(0, index=df.index)), 5) * 35
    growth_component = _scale(df.get("pct_change_7", pd.Series(0, index=df.index)), 100) * 25
    isolation_component = df.get("isolation_anomaly", pd.Series(False, index=df.index)).astype(int) * 25

    df["recent_anomaly_count"] = (
        df.sort_values(["state", "syndrome", "date"])
        .groupby(["state", "syndrome"])["any_anomaly"]
        .transform(lambda x: x.rolling(7, min_periods=1).sum())
    )
    persistence_component = _scale(df["recent_anomaly_count"], 7) * 15

    df["risk_score"] = (z_component + growth_component + isolation_component + persistence_component).round(1).clip(0, 100)

    conditions = [
        df["risk_score"] >= 81,
        df["risk_score"] >= 61,
        df["risk_score"] >= 31,
    ]
    choices = ["Critical", "High", "Moderate"]
    df["risk_level"] = np.select(conditions, choices, default="Low")
    return df


def explain_alert(row: pd.Series) -> str:
    # Build a concise analyst-readable explanation from the triggered model components.
    reasons = []
    if bool(row.get("zscore_anomaly", False)):
        reasons.append(f"visit percentage is {row.get('z_score', 0):.2f} standard deviations from baseline")
    if bool(row.get("isolation_anomaly", False)):
        reasons.append("Isolation Forest identified the pattern as unusual")
    if row.get("pct_change_7", 0) > 0:
        reasons.append(f"7-period growth is {row.get('pct_change_7', 0):.1f}%")
    if not reasons:
        return "No major anomaly signals detected."
    return "; ".join(reasons) + "."
