import pandas as pd
from src.data_cleaning import clean_syndromic_data


def test_clean_syndromic_data_basic():
    # Verify a simple already-long dataset is typed, cleaned, and enriched with calendar fields.
    raw = pd.DataFrame({
        "date": ["2026-01-01", "2026-01-02"],
        "state": ["Georgia", "Georgia"],
        "syndrome": ["COVID-like illness", "COVID-like illness"],
        "visit_percentage": ["2.1", "2.3"],
    })
    clean = clean_syndromic_data(raw)
    assert len(clean) == 2
    assert clean["visit_percentage"].dtype.kind in "fi"
    assert "week" in clean.columns


def test_clean_syndromic_data_reshapes_cdc_ed_trajectory_schema():
    # Verify CDC wide respiratory percentage columns melt into three syndrome rows.
    raw = pd.DataFrame({
        "week_end": ["2026-01-03T00:00:00.000", "2026-01-03T00:00:00.000"],
        "geography": ["Georgia", "Georgia"],
        "county": ["Fulton", "Cobb"],
        "percent_visits_covid": [2.0, 4.0],
        "percent_visits_influenza": [6.0, 8.0],
        "percent_visits_rsv": [1.0, 3.0],
    })

    clean = clean_syndromic_data(raw)

    assert len(clean) == 3
    assert set(clean["syndrome"]) == {
        "COVID-like illness",
        "Influenza-like illness",
        "RSV-like illness",
    }
    covid = clean[clean["syndrome"] == "COVID-like illness"].iloc[0]
    assert covid["state"] == "Georgia"
    assert covid["visit_percentage"] == 3.0
