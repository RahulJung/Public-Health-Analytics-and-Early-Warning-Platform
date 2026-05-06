import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


def detect_zscore_anomalies(df: pd.DataFrame, threshold: float = 2.5) -> pd.DataFrame:
    #     # 1. Copy the input so upstream data is not mutated.
    # 2. Flag records whose rolling-baseline z-score exceeds the threshold.
    # 3. Store absolute z-score as an interpretable severity value.
    df = df.copy()
    df["zscore_anomaly"] = df["z_score"].abs() >= threshold
    df["zscore_severity"] = df["z_score"].abs().fillna(0)
    return df


def detect_isolation_forest_anomalies(df: pd.DataFrame, contamination: float = 0.03) -> pd.DataFrame:
    #     # 1. Build the multivariate feature matrix used for unsupervised anomaly detection.
    # 2. Replace missing/infinite model inputs with zero after feature selection.
    # 3. If the series is too small, skip the model and mark no Isolation Forest anomalies.
    # 4. Standardize features, fit Isolation Forest, and convert decision scores into anomaly scores.
    df = df.copy()
    features = ["visit_percentage", "rolling_mean", "z_score", "pct_change_7", "trend_acceleration"]
    model_df = df[features].replace([np.inf, -np.inf], np.nan).fillna(0)

    if len(model_df) < 20:
        df["isolation_anomaly"] = False
        df["isolation_score"] = 0.0
        return df

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(model_df)
    model = IsolationForest(n_estimators=200, contamination=contamination, random_state=42)
    predictions = model.fit_predict(x_scaled)
    scores = model.decision_function(x_scaled)

    df["isolation_anomaly"] = predictions == -1
    df["isolation_score"] = -scores
    return df


def detect_anomalies(df: pd.DataFrame, zscore_threshold: float = 2.5, contamination: float = 0.03) -> pd.DataFrame:
    #     # 1. Run transparent statistical detection.
    # 2. Run unsupervised multivariate detection.
    # 3. Create combined flags for any model hit and cross-model agreement.
    df = detect_zscore_anomalies(df, threshold=zscore_threshold)
    df = detect_isolation_forest_anomalies(df, contamination=contamination)
    df["any_anomaly"] = df["zscore_anomaly"] | df["isolation_anomaly"]
    df["model_agreement"] = df["zscore_anomaly"] & df["isolation_anomaly"]
    return df
