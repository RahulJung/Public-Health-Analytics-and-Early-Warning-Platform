# Public Health Signal Detection and Response Dashboard

An applied public health informatics prototype for syndromic surveillance analytics, anomaly detection, forecasting, geospatial risk monitoring, synthetic HL7 simulation, and retrieval-augmented public health search.

This project demonstrates how public aggregate or synthetic emergency department surveillance data can be transformed into analyst-facing operational intelligence. The dashboard is designed as a public health analytics modernization prototype, not as an official public health, clinical, or government decision system.

## Project Purpose

Public health surveillance teams need timely, explainable tools for identifying unusual illness patterns, monitoring geographic risk, validating data quality, and communicating emerging signals. This repository implements a modular analytics workflow that turns syndromic surveillance data into:

- Rolling baseline comparisons
- Statistical anomaly detection
- Isolation Forest anomaly scoring
- Short-term forecasting
- Composite risk scoring
- Geographic risk maps
- Alert intelligence tables
- Data quality indicators
- RAG-based natural-language surveillance search
- Synthetic HL7-style test message generation

The current implementation focuses on respiratory syndrome categories derived from public CDC NSSP Emergency Department Visit Trajectories data.

## Current Dataset Snapshot

The processed dataset is stored at `data/processed/cdc_nssp_ed_trajectories_processed.csv`.

| Metric | Current Value |
|---|---:|
| Raw source rows | 50,000 |
| Processed analytic records | 7,842 |
| Date range | 2022-10-01 to 2024-01-06 |
| States / regions | 51 |
| Syndrome categories | 3 |
| Anomaly-flagged records | 236 |
| High-risk records | 182 |
| Critical-risk records | 0 |
| Duplicate records | 0 |
| Data completeness | 98.16% |

Monitored syndrome categories:

- COVID-like illness
- Influenza-like illness
- RSV-like illness

## Public Health Context

Syndromic surveillance uses near real-time healthcare encounter data, such as emergency department visits and chief complaints, to identify unusual patterns before confirmed diagnoses are available. This capability supports early outbreak detection, respiratory illness monitoring, overdose surveillance concepts, environmental health monitoring, emergency preparedness, and cross-jurisdiction situational awareness.

The project is conceptually aligned with public health surveillance modernization priorities reflected in:

- CDC National Syndromic Surveillance Program
- CDC ESSENCE workflows
- CDC BioSense Platform concepts
- HHS public health preparedness and data modernization priorities
- WHO epidemic intelligence and public health surveillance concepts

No government agency endorses or operates this prototype.

## System Architecture

```text
Public / Synthetic Surveillance Data
        |
        v
Data Ingestion
        |
        v
ETL and Quality Control
        |
        v
Feature Engineering
        |
        v
Anomaly Detection + Forecasting
        |
        v
Risk Scoring + Geospatial Aggregation
        |
        v
RAG/NLP Query Assistant
        |
        v
Streamlit Dashboard + Alert Intelligence
```

Core architecture layers:

| Layer | Role |
|---|---|
| Data ingestion | Loads raw public/synthetic surveillance extracts and cached processed files. |
| ETL and quality control | Standardizes dates, geography, syndrome fields, numeric values, missingness, and duplicates. |
| Feature engineering | Computes lag features, rolling baselines, z-scores, percent change, and trend acceleration. |
| Analytics engine | Runs z-score detection, Isolation Forest anomaly scoring, forecasting, and model output generation. |
| Risk scoring | Converts severity, growth, anomaly flags, and persistence into a 0-100 risk score. |
| Geospatial analytics | Aggregates state-level risk and displays risk maps and rankings. |
| RAG/NLP assistant | Retrieves relevant project and surveillance context for grounded natural-language answers. |
| Dashboard layer | Presents filters, KPIs, charts, maps, alert tables, data quality metrics, and methodology sections. |

## Models and Analytics

### Signal Detection

- Rolling baseline comparison
- Z-score anomaly detection
- Isolation Forest anomaly detection
- Percent change from historical baseline
- Composite risk scoring
- Analyst-readable alert explanations

### Forecasting

The forecasting module supports:

- ARIMA
- Prophet, when available and stable in the local environment
- Exponential Smoothing fallback
- Rolling-average last-resort fallback

Forecasting outputs are prototype analytics intended for trend review and early-warning demonstration. They are not operational predictions.

### RAG-Based Public Health Query Assistant

The RAG assistant lets users ask plain-language questions about:

- Current surveillance trends
- High-risk states or regions
- Anomaly scores and alert explanations
- Data quality limitations
- Syndrome definitions
- Model behavior and methodology
- Project documentation

The assistant uses local project documentation and a live surveillance snapshot. If `OPENAI_API_KEY` is configured, it can generate grounded LLM responses from retrieved context. If no key is available, it falls back to retrieval-only responses.

The assistant is not designed to provide medical advice, diagnosis, treatment guidance, or official public health direction.

### Synthetic HL7 and NLP Extensions

The prototype includes:

- Synthetic HL7 ADT-style message generation for testing and demonstration
- Rule-based chief complaint classification for transparent syndrome mapping
- Future design space for privacy-preserving synthetic trend simulation and GAN-style synthetic data generation

Synthetic data is used only for research, testing, and demonstration.

## Streamlit Dashboard

The app is implemented in `app.py` and organized into top-level tabs:

- Overview
- Signals
- Analytics Engine
- Geography
- Forecasting
- Data Quality
- Methodology
- Evidence

Dashboard capabilities:

- Professional public health surveillance landing page
- KPI metric cards
- Public health surveillance context section
- Heat-map style geospatial risk visualization
- Alert intelligence table
- Model output summaries
- Forecast charts
- Data quality scorecards
- RAG-based public health query assistant
- Synthetic HL7 and NLP demonstrations
- References and disclaimer

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Optional: create a local `.env` file for LLM-backed RAG generation:

```text
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4.1-mini
```

Run the dashboard:

```bash
streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal, usually:

```text
http://localhost:8501
```

## Repository Structure

```text
app.py                         Streamlit dashboard
config.yaml                    Data and model configuration
src/                           Data, modeling, forecasting, RAG, HL7, and visualization modules
data/raw/                      Raw public/synthetic source files
data/processed/                Processed analytic surveillance dataset
reports/data_dictionary.md     Data fields and processing definitions
reports/model_summary.md       Model and analytics documentation
reports/public_health_use_cases.md
docs/Public_Health_Signal_Detection_Technical_White_Paper.docx
docs/modernizing_public_health_surveillance_with_ai_medium_article.txt
docs/whitepaper_assets/        Charts generated for documentation
tests/                         Unit tests
```

## Documentation

Key documentation files:

- `docs/Public_Health_Signal_Detection_Technical_White_Paper.docx`  
  Research-grade technical white paper describing the architecture, methodology, public health context, evaluation framework, operational readiness, and future roadmap.

- `docs/modernizing_public_health_surveillance_with_ai_medium_article.txt`  
  Medium-style public-facing article on modernizing public health surveillance with AI.

- `reports/model_summary.md`  
  Technical summary of feature engineering, anomaly detection, forecasting, risk scoring, RAG, NLP, HL7, and geospatial methods.

- `reports/data_dictionary.md`  
  Raw and processed field dictionary for the CDC-derived surveillance dataset.

- `reports/public_health_use_cases.md`  
  Applied public health use cases supported by the prototype architecture.

## Limitations

- The dataset is public aggregate or synthetic and historical, not live operational surveillance data.
- The app is not affiliated with, endorsed by, or operated by HHS, CDC, WHO, NSSP, BioSense, ESSENCE, or any government agency.
- Model thresholds are prototype settings and require validation before operational use.
- Precision and recall require adjudicated ground truth labels and are not asserted from the unlabeled public/synthetic dataset.
- Forecasts are short-term analytic demonstrations, not official predictions.
- RAG responses depend on retrieved local context and should be reviewed by a human analyst.
- The system must not be used for clinical decision-making, diagnosis, treatment, or operational public health response.

## Future Roadmap

- Real-time or scheduled surveillance data refresh
- HL7/FHIR ingestion patterns
- County-level or jurisdiction-level risk views
- API-based alert publishing
- Analyst feedback and alert disposition workflow
- Model monitoring and drift detection
- More advanced chief complaint NLP
- Ontology-aware retrieval using public health and clinical vocabularies
- Privacy-preserving synthetic data simulation
- Cloud-native deployment with authentication, audit logging, and role-based access

## Official References

- CDC National Syndromic Surveillance Program: https://www.cdc.gov/nssp/
- CDC BioSense Platform: https://www.cdc.gov/nssp/php/about/about-nssp-and-the-biosense-platform.html
- CDC ESSENCE: https://www.cdc.gov/nssp/php/onboarding-toolkits/essence.html
- CDC Data Modernization: https://www.cdc.gov/data-modernization/php/about/index.html
- HHS Emergency Preparedness and Response: https://www.hhs.gov/programs/emergency-preparedness/index.html
- WHO Epidemic Intelligence from Open Sources: https://www.who.int/initiatives/eios

## Disclaimer

This prototype is an independent public health analytics research project using public aggregate or synthetic data. It is not an official system of HHS, CDC, WHO, NSSP, BioSense, ESSENCE, or any government agency. It is not intended for clinical decision-making, diagnosis, treatment, or operational public health response.
