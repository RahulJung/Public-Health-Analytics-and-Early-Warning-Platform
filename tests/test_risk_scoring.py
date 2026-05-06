import pandas as pd
from src.risk_scoring import assign_risk_scores


def test_assign_risk_scores_range():
    # Risk scoring should always produce bounded scores and a categorical risk label.
    df = pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=3),
        "state": ["GA", "GA", "GA"],
        "syndrome": ["ILI", "ILI", "ILI"],
        "zscore_severity": [0, 3, 5],
        "pct_change_7": [0, 50, 150],
        "isolation_anomaly": [False, True, True],
        "any_anomaly": [False, True, True],
    })
    scored = assign_risk_scores(df)
    assert scored["risk_score"].between(0, 100).all()
    assert "risk_level" in scored.columns
