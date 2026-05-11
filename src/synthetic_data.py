from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ROOT_DIR


DEFAULT_STATES = [
    "Alabama",
    "Arizona",
    "California",
    "Colorado",
    "Florida",
    "Georgia",
    "Illinois",
    "Louisiana",
    "Massachusetts",
    "Michigan",
    "New York",
    "North Carolina",
    "Ohio",
    "Oregon",
    "Pennsylvania",
    "Texas",
    "Virginia",
    "Washington",
]


SYNDROME_PROFILES = {
    "Gastrointestinal illness": {"baseline": 1.2, "season_amp": 0.25, "peak_day": 210, "dispersion": 45},
    "Heat-related illness": {"baseline": 0.18, "season_amp": 1.8, "peak_day": 205, "dispersion": 18},
    "Suspected opioid overdose": {"baseline": 0.32, "season_amp": 0.12, "peak_day": 180, "dispersion": 24},
    "Mental health crisis": {"baseline": 1.05, "season_amp": 0.16, "peak_day": 45, "dispersion": 36},
    "Suicide-related behavior": {"baseline": 0.22, "season_amp": 0.2, "peak_day": 70, "dispersion": 20},
    "Firearm injury": {"baseline": 0.11, "season_amp": 0.22, "peak_day": 190, "dispersion": 16},
    "Rash and fever syndrome": {"baseline": 0.35, "season_amp": 0.25, "peak_day": 120, "dispersion": 24},
    "Neurological symptoms": {"baseline": 0.28, "season_amp": 0.12, "peak_day": 250, "dispersion": 22},
}


OUTBREAKS = [
    ("OB-GI-2024-001", "Colorado", "Gastrointestinal illness", "2024-06-05", 18, 3.0),
    ("OB-HEAT-2025-002", "Georgia", "Heat-related illness", "2025-07-18", 20, 5.5),
    ("OB-OD-2025-003", "Pennsylvania", "Suspected opioid overdose", "2025-03-12", 14, 4.2),
    ("OB-MH-2025-004", "New York", "Mental health crisis", "2025-10-03", 30, 2.4),
    ("OB-FIREARM-2026-005", "Texas", "Firearm injury", "2026-02-08", 12, 4.8),
]


def _seasonal_factor(day_of_year: int, peak_day: int, amplitude: float) -> float:
    phase = 2 * np.pi * (day_of_year - peak_day) / 365.25
    return max(0.15, 1 + amplitude * np.cos(phase))


def _weekday_factor(day_of_week: int) -> float:
    if day_of_week == 5:
        return 0.88
    if day_of_week == 6:
        return 0.78
    if day_of_week == 0:
        return 1.08
    return 1.0


def _negative_binomial(rng: np.random.Generator, mean: float, dispersion: float) -> int:
    mean = max(float(mean), 0.001)
    dispersion = max(float(dispersion), 0.001)
    probability = dispersion / (dispersion + mean)
    return int(rng.negative_binomial(dispersion, probability))


def _outbreak_multiplier(date: pd.Timestamp, state: str, syndrome: str) -> tuple[float, bool, str]:
    for event_id, event_state, event_syndrome, start_date, duration_days, peak_multiplier in OUTBREAKS:
        if event_state != state or event_syndrome != syndrome:
            continue
        start = pd.Timestamp(start_date)
        end = start + pd.Timedelta(days=duration_days - 1)
        if start <= date <= end:
            position = (date - start).days / max(duration_days - 1, 1)
            pulse = np.sin(np.pi * position)
            return 1 + (peak_multiplier - 1) * pulse, True, event_id
    return 1.0, False, ""


def generate_synthetic_syndromic_data(
    start_date: str = "2023-01-01",
    end_date: str = "2026-04-30",
    states: list[str] | None = None,
    syndromes: list[str] | None = None,
    seed: int = 42,
    missing_rate: float = 0.004,
) -> pd.DataFrame:
    """Generate transparent aggregate synthetic ED surveillance trends."""
    rng = np.random.default_rng(seed)
    states = states or DEFAULT_STATES
    syndromes = syndromes or list(SYNDROME_PROFILES)
    unknown = [name for name in syndromes if name not in SYNDROME_PROFILES]
    if unknown:
        raise ValueError(f"Unknown syndrome profile(s): {unknown}")

    dates = pd.date_range(start_date, end_date, freq="D")
    state_volume = {state: rng.integers(700, 4200) for state in states}
    state_effect = {state: rng.lognormal(mean=0, sigma=0.16) for state in states}
    rows = []

    for state in states:
        for date in dates:
            dow_factor = _weekday_factor(date.dayofweek)
            visit_mean = state_volume[state] * dow_factor * rng.lognormal(mean=0, sigma=0.05)
            total_ed_visits = max(50, _negative_binomial(rng, visit_mean, dispersion=320))

            for syndrome in syndromes:
                profile = SYNDROME_PROFILES[syndrome]
                seasonal = _seasonal_factor(date.dayofyear, profile["peak_day"], profile["season_amp"])
                outbreak_factor, outbreak_flag, event_id = _outbreak_multiplier(date, state, syndrome)
                secular = 1 + 0.03 * ((date.year - dates[0].year) / max(dates[-1].year - dates[0].year, 1))
                expected_pct = profile["baseline"] * seasonal * state_effect[state] * dow_factor * secular * outbreak_factor
                expected_pct = float(np.clip(expected_pct, 0.01, 35.0))
                syndrome_visits = _negative_binomial(
                    rng,
                    total_ed_visits * expected_pct / 100,
                    dispersion=profile["dispersion"],
                )
                visit_percentage = round((syndrome_visits / total_ed_visits) * 100, 3)
                if rng.random() < missing_rate:
                    visit_percentage = np.nan

                rows.append({
                    "date": date.date().isoformat(),
                    "state": state,
                    "syndrome": syndrome,
                    "visit_percentage": visit_percentage,
                    "source": "synthetic",
                    "total_ed_visits": total_ed_visits,
                    "syndrome_visits": syndrome_visits,
                    "synthetic_outbreak": outbreak_flag,
                    "synthetic_event_id": event_id,
                })

    return pd.DataFrame(rows)


def save_synthetic_syndromic_data(df: pd.DataFrame, output_path: str | Path) -> Path:
    path = Path(output_path)
    if not path.is_absolute():
        path = ROOT_DIR / path
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic aggregate syndromic surveillance data.")
    parser.add_argument("--output", default="data/raw/synthetic_syndromic_surveillance.csv")
    parser.add_argument("--start-date", default="2023-01-01")
    parser.add_argument("--end-date", default="2026-04-30")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    df = generate_synthetic_syndromic_data(start_date=args.start_date, end_date=args.end_date, seed=args.seed)
    output_path = save_synthetic_syndromic_data(df, args.output)
    print(f"Wrote {len(df):,} synthetic records to {output_path}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"States: {df['state'].nunique()}, syndromes: {df['syndrome'].nunique()}")


if __name__ == "__main__":
    main()
