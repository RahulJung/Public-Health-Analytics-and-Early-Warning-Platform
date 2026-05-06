# Data Dictionary

This dictionary documents the CDC-derived emergency department visit trajectory data fetched into `data/raw/cdc_nssp_ed_trajectories_raw.csv` and the processed dataset used by the Streamlit dashboard in `data/processed/cdc_nssp_ed_trajectories_processed.csv`.

The data used in this project is public aggregate or synthetic surveillance data. It is not patient-level clinical data.

## Current Data Inventory

| Dataset | Path | Current Size |
|---|---|---:|
| Raw source extract | `data/raw/cdc_nssp_ed_trajectories_raw.csv` | 50,000 rows |
| Processed analytic dataset | `data/processed/cdc_nssp_ed_trajectories_processed.csv` | 7,842 rows |

Processed dataset summary:

| Metric | Current Value |
|---|---:|
| Date range | 2022-10-01 to 2024-01-06 |
| States / regions | 51 |
| Syndrome categories | 3 |
| Data completeness | 98.16% |
| Missing-value rate | 1.84% |
| Duplicate records | 0 |

## Raw CDC-Derived Data

Source: CDC Socrata API endpoint configured in `config.yaml`.

| Field | Type | Description |
|---|---:|---|
| week_end | object/date | Week ending date for the surveillance reporting period. |
| geography | object | State, jurisdiction, or region name. This becomes `state` in the processed data. |
| county | object | County name associated with the source record. |
| ed_trends_covid | object | CDC trend category or availability status for COVID-related ED visits. |
| ed_trends_influenza | object | CDC trend category or availability status for influenza-related ED visits. |
| ed_trends_rsv | object | CDC trend category or availability status for RSV-related ED visits. |
| hsa | object | Health service area name. |
| hsa_counties | object | Counties included in the health service area. |
| hsa_nci_id | integer | Health service area identifier. |
| fips | integer | County FIPS code. |
| trend_source | object | Geographic or source level used for the trend record, such as HSA. |
| buildnumber | object/date | CDC data build or publication date. |
| percent_visits_combined | float | Percentage of ED visits for the combined respiratory illness grouping. |
| percent_visits_covid | float | Percentage of ED visits associated with COVID-like illness. |
| percent_visits_influenza | float | Percentage of ED visits associated with influenza-like illness. |
| percent_visits_rsv | float | Percentage of ED visits associated with RSV-like illness. |
| percent_visits_smoothed | float | Smoothed percentage for the combined respiratory illness grouping. |
| percent_visits_smoothed_covid | float | Smoothed percentage for COVID-like illness. |
| percent_visits_smoothed_1 | float | Smoothed percentage for influenza-like illness in the fetched CDC schema. |
| percent_visits_smoothed_rsv | float | Smoothed percentage for RSV-like illness. |

## Processing Notes

The fetched CDC-derived file is wide-format: COVID, influenza, and RSV percentages are stored in separate columns. The cleaning step reshapes these columns into one row per `date`, `state`, and `syndrome`.

Current syndrome mapping:

| Raw Field | Processed Syndrome |
|---|---|
| percent_visits_covid | COVID-like illness |
| percent_visits_influenza | Influenza-like illness |
| percent_visits_rsv | RSV-like illness |

County-level rows are averaged by `date`, `state`, and `syndrome` to create the processed state-level trend dataset used by the dashboard and models.

## Processed Analytic Dataset

| Field | Type | Description |
|---|---:|---|
| date | date | Reporting date derived from `week_end`. |
| state | object | State, jurisdiction, or region derived from `geography`. |
| syndrome | object | Public health indicator being monitored: COVID-like illness, influenza-like illness, or RSV-like illness. |
| visit_percentage | float | Percentage of ED visits associated with the syndrome. |
| year | integer | Calendar year extracted from `date`. |
| month | integer | Calendar month extracted from `date`. |
| week | integer | ISO week number extracted from `date`. |
| lag_1 | float | Previous record's `visit_percentage` for the same state and syndrome. |
| lag_7 | float | `visit_percentage` seven records earlier for the same state and syndrome. |
| rolling_mean | float | Rolling baseline average of `visit_percentage`. |
| rolling_std | float | Rolling baseline standard deviation of `visit_percentage`. |
| z_score | float | Standardized distance from the rolling baseline. |
| pct_change_7 | float | Percent change from seven records earlier. |
| trend_acceleration | float | Period-over-period change in `visit_percentage`. |
| zscore_anomaly | boolean | True when the absolute z-score exceeds the configured threshold. |
| zscore_severity | float | Absolute z-score used as a statistical severity signal. |
| isolation_anomaly | boolean | True when Isolation Forest classifies the record as unusual. |
| isolation_score | float | Isolation Forest anomaly score, with larger values indicating more unusual records. |
| any_anomaly | boolean | True when either z-score detection or Isolation Forest flags the record. |
| model_agreement | boolean | True when both anomaly detection methods flag the record. |
| recent_anomaly_count | float | Rolling count of recent anomaly flags for the same state and syndrome. |
| risk_score | float | 0-100 public-health-style risk score combining severity, growth, anomaly status, and persistence. |
| risk_level | object | Risk label derived from `risk_score`: Low, Moderate, High, or Critical. |
| alert_explanation | object | Human-readable explanation of anomaly and risk signals. |

## Risk Level Values

| Risk Level | Score Range | Current Record Count |
|---|---:|---:|
| Low | 0-30 | 6,158 |
| Moderate | 31-60 | 1,502 |
| High | 61-80 | 182 |
| Critical | 81-100 | 0 |

## RAG Context Sources

The RAG assistant builds retrieval context from local documentation and a live surveillance snapshot.

Current indexed local files include:

- `README.md`
- `reports/model_summary.md`
- `reports/data_dictionary.md`
- `reports/public_health_use_cases.md`

Generated runtime context includes:

- Latest data date
- Risk-level distribution
- High-risk state summaries
- Anomaly counts
- Syndrome trend summaries
- Data quality indicators

## Synthetic Data Boundary

The synthetic HL7 generator creates simulated ADT-style messages for testing and demonstration. These messages are not real patient records and should not be connected to real identifiers or operational clinical data.

## Data Limitations

- The current data is public aggregate or synthetic and historical.
- Visit counts are unavailable in the processed percentage-only source.
- County-level records are averaged to state-level trends.
- Missing values occur primarily in lag and rolling-window fields at the beginning of each state-syndrome series.
- The data should not be interpreted as current operational surveillance intelligence.
