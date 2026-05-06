from .config import load_config
from .data_ingestion import load_or_fetch_data
from .data_cleaning import clean_syndromic_data, save_processed_data
from .feature_engineering import add_features
from .anomaly_detection import detect_anomalies
from .risk_scoring import assign_risk_scores, explain_alert


def run_pipeline(force_refresh: bool = False):
    #     # 1. Load configuration and raw surveillance data.
    # 2. Clean and reshape data into the analytic state/date/syndrome grain.
    # 3. Add temporal features, detect anomalies, and compute risk scores.
    # 4. Generate analyst-readable alert explanations.
    # 5. Persist the processed dataset for the Streamlit dashboard.
    cfg = load_config()
    raw = load_or_fetch_data(force_refresh=force_refresh)
    clean = clean_syndromic_data(raw)
    featured = add_features(clean, rolling_window=int(cfg["model"]["rolling_window"]))
    detected = detect_anomalies(
        featured,
        zscore_threshold=float(cfg["model"]["zscore_threshold"]),
        contamination=float(cfg["model"]["contamination"]),
    )
    scored = assign_risk_scores(detected)
    scored["alert_explanation"] = scored.apply(explain_alert, axis=1)
    save_processed_data(scored)
    return scored


if __name__ == "__main__":
    # Execute the full pipeline from the command line for a refresh smoke test.
    output = run_pipeline(force_refresh=True)
    print(output.head())
    print(output.shape)
