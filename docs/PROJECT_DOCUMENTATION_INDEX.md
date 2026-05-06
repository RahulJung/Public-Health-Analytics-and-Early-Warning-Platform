# Project Documentation Index

This folder contains project-level documentation for the Public Health Signal Detection and Response Dashboard.

## Primary Documents

| Document | Purpose |
|---|---|
| `Public_Health_Signal_Detection_Technical_White_Paper.docx` | Research-grade technical white paper covering public health context, architecture, AI/ML methodology, forecasting, RAG/NLP design, synthetic HL7 concepts, operational readiness, limitations, and roadmap. |
| `modernizing_public_health_surveillance_with_ai_medium_article.txt` | Medium-style public-facing article explaining how AI can support public health surveillance modernization. |
| `NIW_PROJECT_SUMMARY.md` | Project summary retained for continuity with earlier documentation, now written as a public health surveillance analytics project summary. |

## Supporting Reports

| Report | Path | Purpose |
|---|---|---|
| Data dictionary | `../reports/data_dictionary.md` | Defines raw and processed dataset fields, model output columns, RAG context sources, and data limitations. |
| Model summary | `../reports/model_summary.md` | Documents feature engineering, anomaly detection, forecasting, risk scoring, RAG, NLP, synthetic HL7, geospatial analytics, and model limitations. |
| Public health use cases | `../reports/public_health_use_cases.md` | Describes applied public health workflows supported by the prototype architecture. |

## Generated White Paper Assets

Charts used in the technical white paper are stored in:

```text
docs/whitepaper_assets/
```

These include risk distribution, anomaly-by-syndrome, temporal forecast, state risk ranking, architecture, and data quality figures.

## Current Data Metrics

| Metric | Current Value |
|---|---:|
| Raw source rows | 50,000 |
| Processed analytic records | 7,842 |
| Date range | 2022-10-01 to 2024-01-06 |
| States / regions | 51 |
| Syndrome categories | 3 |
| Anomaly-flagged records | 236 |
| High-risk records | 182 |
| Data completeness | 98.16% |

## Disclaimer

This project uses public aggregate or synthetic data for demonstration, research, and portfolio purposes. It is not an official system of HHS, CDC, WHO, NSSP, BioSense, ESSENCE, or any government agency. It is not intended for clinical decision-making, diagnosis, treatment, or operational public health response.
