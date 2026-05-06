# Model and Analytics Summary

## Objective

The Public Health Signal Detection and Response Dashboard detects unusual emergency department visit patterns using public aggregate or synthetic syndromic surveillance data. The goal is to demonstrate an analyst-facing surveillance analytics workflow for respiratory illness indicators, geographic risk monitoring, forecasting, data quality review, and grounded natural-language query support.

The system is not an official surveillance product. It is designed as an applied public health informatics and analytics modernization prototype.

## Input Data

The raw CDC-derived dataset is stored at:

```text
data/raw/cdc_nssp_ed_trajectories_raw.csv
```

The processed analytic dataset is stored at:

```text
data/processed/cdc_nssp_ed_trajectories_processed.csv
```

Current processed dataset size: 7,842 records.

Current date range: 2022-10-01 to 2024-01-06.

The fetched source file is wide-format, with separate percentage columns for each monitored illness:

| Raw Field | Processed Syndrome |
|---|---|
| percent_visits_covid | COVID-like illness |
| percent_visits_influenza | Influenza-like illness |
| percent_visits_rsv | RSV-like illness |

During cleaning, these fields are reshaped into one row per `date`, `state`, and `syndrome`. County-level records are averaged into state-level syndrome trends before feature engineering.

## Feature Engineering

Features are calculated separately for each state and syndrome time series.

| Feature | Description |
|---|---|
| lag_1 | Previous record's visit percentage. |
| lag_7 | Visit percentage seven records earlier. |
| rolling_mean | Rolling baseline average using the configured rolling window. |
| rolling_std | Rolling standard deviation using the configured rolling window. |
| z_score | Difference between current value and rolling baseline, scaled by rolling standard deviation. |
| pct_change_7 | Percent change from seven records earlier. |
| trend_acceleration | Period-over-period change in visit percentage. |

Configured rolling window: 7 records.

The feature layer is intended to provide temporal context before model scoring. This avoids interpreting every observation independently and allows the system to evaluate whether current activity is unusual relative to recent state-syndrome behavior.

## Anomaly Detection Models

### Z-Score Detector

The z-score detector flags observations whose absolute z-score exceeds the configured threshold.

Current threshold: 2.5.

This method is simple, transparent, and useful for analyst-facing dashboards because reviewers can interpret how far the current value is from the recent rolling baseline.

### Isolation Forest

Isolation Forest is an unsupervised anomaly detection model that identifies unusual observations using multiple engineered features:

- `visit_percentage`
- `rolling_mean`
- `z_score`
- `pct_change_7`
- `trend_acceleration`

Before fitting the model, features are standardized with `StandardScaler`. Missing and infinite values are replaced with zero for model input. If fewer than 20 records are available, the Isolation Forest step is skipped and no Isolation Forest anomalies are assigned.

Current Isolation Forest settings:

| Parameter | Value |
|---|---:|
| n_estimators | 200 |
| contamination | 0.03 |
| random_state | 42 |

The system also creates:

| Field | Description |
|---|---|
| any_anomaly | True if either z-score or Isolation Forest flags the record. |
| model_agreement | True if both methods flag the same record. |

Current processed output contains 236 records flagged as anomalies.

Current model agreement count: 0 records.

Model agreement is intentionally reported as an observed metric, not as a quality claim. A production validation plan would require adjudicated labels, retrospective event review, and analyst feedback.

## Forecasting

Short-term forecasting supports ARIMA and Prophet, with fallback behavior for local runtime stability.

The default forecasting model is ARIMA from `statsmodels`. Prophet is available from the dashboard when the `prophet` package is installed and can fit the selected time series. Exponential Smoothing remains as a fallback so the dashboard can still produce a forecast if ARIMA or Prophet cannot fit a specific state/syndrome series. A rolling-average forecast is used only as a last-resort fallback.

The forecast step:

1. Filters to the selected state and syndrome.
2. Groups values by date.
3. Regularizes the series using the observed median time interval.
4. Splits recent observations into a test window.
5. Fits the selected forecast model.
6. Produces backtest predictions and future forecasts.

Supported forecasting models:

| Model | Role | Notes |
|---|---|---|
| ARIMA | Default | Uses an ARIMA(1, 1, 1) model for short-term statistical forecasting. |
| Prophet | Optional dashboard selection | Uses Prophet with yearly seasonality enabled and daily/weekly seasonality disabled. |
| Exponential Smoothing | Fallback and selectable baseline | Uses additive trend Holt-Winters Exponential Smoothing. |
| Rolling Average | Last-resort fallback | Used only if the selected model and Exponential Smoothing both fail. |

Configured forecast horizon: 14 periods.

Reported forecast metrics:

| Metric | Description |
|---|---|
| Model | Forecasting model that actually produced the output. |
| MAE | Mean absolute error on the test window. |
| RMSE | Root mean squared error on the test window. |
| MAPE | Mean absolute percentage error on the test window. |

Forecasting outputs are prototype analytics intended for review and planning context. They should not be interpreted as official predictions.

## Risk Scoring

The risk score combines statistical severity, recent growth, machine-learning anomaly status, and anomaly persistence into a 0-100 score.

| Component | Weight | Description |
|---|---:|---|
| Z-score severity | 35 | Scaled absolute z-score, capped at a severity value of 5. |
| Seven-record growth | 25 | Positive `pct_change_7`, capped at 100 percent growth. |
| Isolation Forest anomaly | 25 | Adds risk when Isolation Forest flags the record. |
| Recent persistence | 15 | Rolling count of anomaly flags across the last 7 records. |

Risk level thresholds:

| Risk Level | Score Range |
|---|---:|
| Low | 0-30 |
| Moderate | 31-60 |
| High | 61-80 |
| Critical | 81-100 |

Current processed risk-level distribution:

| Risk Level | Records |
|---|---:|
| Low | 6,158 |
| Moderate | 1,502 |
| High | 182 |
| Critical | 0 |

Top state/region maximum risk scores in the current processed dataset:

| State / Region | Max Risk Score |
|---|---:|
| Oklahoma | 76.3 |
| Nebraska | 74.7 |
| Idaho | 73.2 |
| Montana | 72.8 |
| Vermont | 72.6 |

## Alert Explanations

Each row receives an analyst-readable explanation. Explanations can include:

- Whether the visit percentage is far from baseline by z-score.
- Whether Isolation Forest identified the pattern as unusual.
- Whether the seven-record percentage change is positive.

Rows without major signals receive:

```text
No major anomaly signals detected.
```

The alert language is intentionally framed as analyst review support, not automated public health direction.

## RAG-Based Public Health Query Assistant

The RAG-based public health search assistant lets analysts ask natural-language questions about surveillance trends, data quality, syndrome definitions, model behavior, or operational changes.

The assistant uses a retrieval-augmented generation workflow:

1. Build a live surveillance snapshot from the loaded processed dataset.
2. Load local project documentation.
3. Chunk the documentation into searchable passages.
4. Retrieve relevant context using TF-IDF and cosine similarity.
5. Send the question and retrieved context to an LLM when `OPENAI_API_KEY` is configured.
6. Fall back to retrieval-only answers when LLM generation is unavailable.

The retriever indexes:

- `README.md`
- `reports/model_summary.md`
- `reports/data_dictionary.md`
- `reports/public_health_use_cases.md`
- A generated `live_surveillance_snapshot` containing latest risk, anomaly, trend, and data quality context.

The LLM prompt instructs the model to answer only from retrieved context, state when evidence is insufficient, and avoid diagnosis or official public health directives.

## Synthetic HL7 Generator

The synthetic HL7 generator creates ADT-style emergency department messages for selected state and syndrome records. Messages include core `MSH`, `PID`, `PV1`, and `OBX` segments, including chief complaint text and a syndromic surveillance category.

This is a simulation layer for demonstrations and test workflows. It does not generate real patient data.

## Chief Complaint NLP Classifier

The chief complaint classifier uses transparent keyword matching to classify free-text chief complaints into:

- COVID-like illness
- Influenza-like illness
- RSV-like illness
- Unclassified

The classifier returns a predicted syndrome, confidence score, per-syndrome scores, and matched terms. It is intentionally simple and explainable; a trained clinical NLP model would be a future enhancement.

## Geospatial Risk Map

The geospatial view aggregates latest state-level syndrome records and maps them using built-in state centroid coordinates. Marker color and size represent the latest maximum risk score by state.

This layer supports geographic risk ranking and visual triage. The current implementation is state-level and should not be interpreted as county-level or facility-level surveillance.

## Data Quality Metrics

Current processed dataset quality metrics:

| Metric | Value |
|---|---:|
| Completeness | 98.16% |
| Missing-value rate | 1.84% |
| Duplicate records | 0 |
| Valid date coverage | 100% |
| States / regions represented | 51 |

Missing values are expected primarily in lag and rolling-window fields at the beginning of each state-syndrome time series.

## Limitations

- The data is public, aggregated, and historical, not real-time clinical data.
- The model does not replace epidemiological review.
- The current pipeline focuses on COVID-like illness, influenza-like illness, and RSV-like illness.
- County-level raw records are averaged to state-level trends, which may hide local hotspots.
- Z-score and Isolation Forest outputs depend on data completeness and recent baseline quality.
- Precision and recall require adjudicated ground truth labels and are not asserted from this unlabeled dataset.
- Forecasts are short-term statistical estimates and should not be interpreted as clinical or operational predictions.
- The HL7 generator creates synthetic messages only and should not be used with real patient identifiers.
- The chief complaint classifier is rule-based and should be validated before operational use.
- The RAG assistant depends on retrieved local context and should not be used as a substitute for epidemiological review, clinical reasoning, or official public health guidance.
- LLM generation requires an `OPENAI_API_KEY`; without it, the app uses retrieval-only mode.

## Human-in-the-Loop Design

The project is designed to support analysts and epidemiologists, not replace them. Alerts should be reviewed by public health professionals before any operational use, especially when decisions may involve resource allocation, public communication, or outbreak response.
