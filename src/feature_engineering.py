import numpy as np
import pandas as pd


def add_features(df: pd.DataFrame, rolling_window: int = 7) -> pd.DataFrame:
    #     # 1. Sort records into state/syndrome time-series order.
    # 2. Compute lagged values and rolling baseline statistics within each group.
    # 3. Derive z-score, percent change, and acceleration features for model scoring.
    # 4. Replace infinite values caused by zero denominators with missing values.
    df = df.copy().sort_values(["state", "syndrome", "date"])
    group_cols = ["state", "syndrome"]

    df["lag_1"] = df.groupby(group_cols)["visit_percentage"].shift(1)
    df["lag_7"] = df.groupby(group_cols)["visit_percentage"].shift(7)
    df["rolling_mean"] = df.groupby(group_cols)["visit_percentage"].transform(
        lambda x: x.rolling(rolling_window, min_periods=3).mean()
    )
    df["rolling_std"] = df.groupby(group_cols)["visit_percentage"].transform(
        lambda x: x.rolling(rolling_window, min_periods=3).std()
    )
    df["rolling_std"] = df["rolling_std"].replace(0, np.nan)
    df["z_score"] = (df["visit_percentage"] - df["rolling_mean"]) / df["rolling_std"]
    df["pct_change_7"] = ((df["visit_percentage"] - df["lag_7"]) / df["lag_7"].replace(0, np.nan)) * 100
    df["trend_acceleration"] = df.groupby(group_cols)["visit_percentage"].diff().fillna(0)
    df = df.replace([np.inf, -np.inf], np.nan)
    return df
