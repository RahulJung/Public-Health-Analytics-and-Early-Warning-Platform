from src.synthetic_data import generate_synthetic_syndromic_data
from src.data_cleaning import clean_syndromic_data


def test_synthetic_data_adds_applied_surveillance_syndromes():
    df = generate_synthetic_syndromic_data(
        start_date="2025-01-01",
        end_date="2025-01-07",
        states=["Georgia", "Pennsylvania"],
        seed=7,
        missing_rate=0,
    )

    expected = {
        "Gastrointestinal illness",
        "Heat-related illness",
        "Suspected opioid overdose",
        "Mental health crisis",
        "Suicide-related behavior",
        "Firearm injury",
        "Rash and fever syndrome",
        "Neurological symptoms",
    }
    assert set(df["syndrome"].unique()) == expected
    assert {"date", "state", "syndrome", "visit_percentage"}.issubset(df.columns)
    assert df["visit_percentage"].notna().all()

    clean = clean_syndromic_data(df)
    assert len(clean) == len(df)
    assert set(clean["syndrome"].unique()) == expected
