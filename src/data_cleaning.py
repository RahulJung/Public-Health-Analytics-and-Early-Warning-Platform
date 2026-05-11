import re
import pandas as pd
from .config import ROOT_DIR, load_config


def _normalize_column(name: str) -> str:
    # Convert source-specific column names into stable snake_case-like tokens.
    name = re.sub(r"[^0-9a-zA-Z]+", "_", str(name).strip().lower())
    return name.strip("_")


def _find_column(columns, candidates):
    # Search normalized source columns for likely aliases of a required field.
    normalized = {col: _normalize_column(col) for col in columns}
    for original, norm in normalized.items():
        for candidate in candidates:
            if candidate in norm:
                return original
    return None


def _reshape_cdc_ed_trajectories(df: pd.DataFrame) -> pd.DataFrame | None:
    #     # 1. Detect the CDC ED Trajectories schema.
    # 2. Melt COVID, influenza, and RSV percentage fields into a long syndrome table.
    # 3. Aggregate county/source rows into state-date-syndrome trend records.
    value_columns = {
        "percent_visits_covid": "COVID-like illness",
        "percent_visits_influenza": "Influenza-like illness",
        "percent_visits_rsv": "RSV-like illness",
    }
    available_values = [col for col in value_columns if col in df.columns]
    if "week_end" not in df.columns or "geography" not in df.columns or not available_values:
        return None

    long_df = df.melt(
        id_vars=["week_end", "geography"],
        value_vars=available_values,
        var_name="syndrome",
        value_name="_cdc_visit_percentage",
    )
    long_df["date"] = pd.to_datetime(long_df["week_end"], errors="coerce")
    long_df["state"] = long_df["geography"].astype(str).str.strip()
    long_df["syndrome"] = long_df["syndrome"].map(value_columns)
    long_df["visit_percentage"] = pd.to_numeric(long_df["_cdc_visit_percentage"], errors="coerce")
    long_df = long_df.dropna(subset=["date", "state", "syndrome", "visit_percentage"])

    return (
        long_df.groupby(["date", "state", "syndrome"], as_index=False)["visit_percentage"]
        .mean()
        .sort_values(["state", "syndrome", "date"])
    )


def clean_syndromic_data(df: pd.DataFrame) -> pd.DataFrame:
    #     # 1. Normalize incoming column names.
    # 2. Prefer the known CDC ED Trajectories reshape path when that schema is present.
    # 3. Otherwise infer date, geography, syndrome, and value columns from aliases.
    # 4. Validate required fields, coerce types, remove invalid rows, and add calendar fields.
    if df.empty:
        raise ValueError("Input dataframe is empty.")

    df = df.copy()
    df.columns = [_normalize_column(col) for col in df.columns]
    cdc_long = _reshape_cdc_ed_trajectories(df)
    if cdc_long is not None:
        long_columns = ["date", "state", "syndrome", "visit_percentage"]
        if all(col in df.columns for col in long_columns):
            long_df = df[long_columns].dropna(subset=long_columns)
            df = pd.concat([cdc_long, long_df], ignore_index=True, sort=False)
        else:
            df = cdc_long
    else:
        date_col = _find_column(df.columns, ["week", "date", "time_period", "reporting_period", "mmwr_week"])
        geo_col = _find_column(df.columns, ["state", "jurisdiction", "geography", "region", "location"])
        syndrome_col = _find_column(df.columns, ["pathogen", "syndrome", "condition", "indicator", "visit_type"])
        value_col = _find_column(df.columns, ["percent", "percentage", "pct", "proportion", "visit_percent", "value"])

        rename_map = {}
        if date_col:
            rename_map[date_col] = "date"
        if geo_col:
            rename_map[geo_col] = "state"
        if syndrome_col:
            rename_map[syndrome_col] = "syndrome"
        if value_col:
            rename_map[value_col] = "visit_percentage"
        df = df.rename(columns=rename_map)

    required = ["date", "state", "syndrome", "visit_percentage"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        object_cols = df.select_dtypes(include="object").columns.tolist()
        raise ValueError(
            f"Could not infer required columns: {missing}. Available columns: {list(df.columns)}. "
            f"Numeric columns: {numeric_cols}. Object columns: {object_cols}"
        )

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["state"] = df["state"].astype(str).str.strip()
    df["syndrome"] = df["syndrome"].astype(str).str.strip()
    df["visit_percentage"] = pd.to_numeric(df["visit_percentage"], errors="coerce")

    df = df.dropna(subset=["date", "state", "syndrome", "visit_percentage"])
    df = df.drop_duplicates(subset=["date", "state", "syndrome"], keep="last")
    df = df.sort_values(["state", "syndrome", "date"]).reset_index(drop=True)

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["week"] = df["date"].dt.isocalendar().week.astype(int)

    return df[["date", "state", "syndrome", "visit_percentage", "year", "month", "week"]]


def save_processed_data(df: pd.DataFrame) -> None:
    # Save the processed analytic table to the configured project output path.
    cfg = load_config()
    processed_path = ROOT_DIR / cfg["data"]["processed_path"]
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed_path, index=False)


if __name__ == "__main__":
    # Allow this module to be run directly for a quick clean-and-save smoke test.
    from .data_ingestion import load_or_fetch_data

    raw = load_or_fetch_data()
    clean = clean_syndromic_data(raw)
    save_processed_data(clean)
    print(clean.head())
