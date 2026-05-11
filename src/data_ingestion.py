from pathlib import Path
import pandas as pd
import requests
from .config import ROOT_DIR, load_config
from .synthetic_data import generate_synthetic_syndromic_data, save_synthetic_syndromic_data


def fetch_cdc_data(api_url: str, max_records: int = 50000, app_token: str = "") -> pd.DataFrame:
    """Fetch public CDC NSSP ED trajectory records from the Socrata API."""
    #     # 1. Add optional Socrata app token when configured.
    # 2. Page through the API until max_records is reached or no more rows are returned.
    # 3. Return all records as a dataframe for cleaning.
    headers = {}
    if app_token:
        headers["X-App-Token"] = app_token

    records = []
    limit = 50000
    offset = 0

    while len(records) < max_records:
        params = {"$limit": min(limit, max_records - len(records)), "$offset": offset}
        response = requests.get(api_url, params=params, headers=headers, timeout=60)
        response.raise_for_status()
        batch = response.json()
        if not batch:
            break
        records.extend(batch)
        offset += len(batch)
        if len(batch) < limit:
            break

    return pd.DataFrame(records)


def load_or_fetch_data(force_refresh: bool = False) -> pd.DataFrame:
    #     # 1. Read raw cached data unless a refresh is requested.
    # 2. Otherwise fetch from the public API and cache the raw extract.
    # 3. If the API fails, fall back to the bundled sample dataset when available.
    cfg = load_config()
    raw_path = ROOT_DIR / cfg["data"]["raw_path"]
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    if raw_path.exists() and not force_refresh:
        df = pd.read_csv(raw_path, low_memory=False)
    else:
        try:
            df = fetch_cdc_data(
                cfg["data"]["cdc_api_url"],
                max_records=int(cfg["data"].get("max_records", 50000)),
                app_token=cfg["data"].get("app_token", ""),
            )
            if df.empty:
                raise ValueError("CDC API returned no records.")
            df.to_csv(raw_path, index=False)
        except Exception:
            sample_path = ROOT_DIR / "data" / "raw" / "sample_syndromic_data.csv"
            if sample_path.exists():
                df = pd.read_csv(sample_path)
            else:
                raise

    if not cfg["data"].get("include_synthetic", False):
        return df

    synthetic_path = ROOT_DIR / cfg["data"].get("synthetic_path", "data/raw/synthetic_syndromic_surveillance.csv")
    if synthetic_path.exists() and not force_refresh:
        synthetic = pd.read_csv(synthetic_path, low_memory=False)
    else:
        synthetic = generate_synthetic_syndromic_data()
        save_synthetic_syndromic_data(synthetic, synthetic_path)

    df = df.copy()
    df["source"] = "cdc_public"
    return pd.concat([df, synthetic], ignore_index=True, sort=False)


if __name__ == "__main__":
    # Run a direct ingestion smoke test and print sample rows/shape.
    data = load_or_fetch_data(force_refresh=True)
    print(data.head())
    print(data.shape)
