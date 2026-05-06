# Public Health Use Cases

This document summarizes applied public health use cases supported by the Public Health Signal Detection and Response Dashboard. The current implementation uses public aggregate or synthetic respiratory syndrome surveillance data, but the architecture is designed to generalize to other syndrome definitions and approved data feeds.

## 1. Respiratory Illness Monitoring

The dashboard monitors COVID-like illness, influenza-like illness, RSV-like illness, and related respiratory indicators. It can identify unusual increases in emergency department visit percentages by state or region and display changes through trends, anomaly flags, forecasts, and risk scores.

Relevant workflows:

- Seasonal respiratory illness monitoring
- Early surge detection
- Cross-state comparison
- Baseline deviation review
- Forecast-informed planning

## 2. Outbreak and Emerging Signal Detection

The anomaly detection workflow can surface unusual syndrome activity that may require analyst review. Statistical z-score detection highlights deviations from rolling baseline behavior, while Isolation Forest identifies unusual multivariate patterns across visit percentage, baseline, growth, and acceleration features.

The output is designed for human review and does not represent an automated outbreak determination.

## 3. Opioid and Overdose Surveillance Concept

The same architecture can be adapted to suspected overdose emergency department visit data, where anomaly detection may help flag county, state, or regional spikes that require public health investigation.

Potential adaptations:

- Overdose-related syndrome definitions
- Chief complaint NLP expansion
- Geographic hotspot views
- Alert prioritization by growth and persistence

## 4. Heat Illness and Environmental Exposure

The system can be extended to monitor heat-related illness, wildfire smoke exposure, carbon monoxide exposure, or other environmental health syndromes during extreme weather events and disasters.

Potential outputs:

- Heat-map style geographic risk views
- Time-series monitoring during emergency events
- Data quality checks for reporting disruptions
- Analyst summaries for situational awareness

## 5. Emergency Preparedness and Response Intelligence

During emergency events, public health teams need fast, understandable surveillance summaries. The dashboard can support preparedness workflows by providing:

- Current risk summaries
- Geographic ranking tables
- Syndrome trend charts
- Short-term forecast context
- Data quality indicators
- Plain-language RAG summaries for briefing preparation

## 6. Data Quality Surveillance

Surveillance signals must be interpreted in the context of data quality. The project reports completeness, missing-value rate, duplicate count, date range, geography coverage, and data validity indicators.

Potential data quality use cases:

- Detect abnormal drops in reporting completeness
- Identify missing or stale data windows
- Review duplicate submissions
- Evaluate whether a signal may reflect a reporting artifact
- Communicate data limitations alongside model outputs

## 7. Synthetic HL7/EHR Simulation

The system can generate synthetic HL7 ADT-style emergency department messages for demonstration, testing, and pipeline validation. These messages include simulated patient identifiers, visit metadata, chief complaint text, and syndromic surveillance category fields.

This is useful for:

- Testing ingestion concepts without PHI
- Demonstrating HL7-style public health data flow
- Stress-testing parsing and classification logic
- Training and sandbox workflows

The synthetic generator must not be used with real patient identifiers.

## 8. Chief Complaint NLP Classification

The chief complaint classifier maps free-text symptoms to respiratory syndrome categories. The current implementation is rule-based and transparent, making it useful for demonstrations and analyst review before moving to trained clinical NLP models.

Current output includes:

- Predicted syndrome category
- Confidence score
- Per-syndrome match scores
- Matched keywords

Future versions could incorporate validated clinical NLP models, terminology mapping, and ontology-aware classification.

## 9. RAG-Based Public Health Search Assistant

The RAG assistant retrieves relevant sections from local project documentation and a live surveillance snapshot, then can use an LLM to generate a grounded analyst-facing answer when `OPENAI_API_KEY` is configured.

Example questions:

- Which states show unusual respiratory activity?
- What regions have the highest anomaly scores?
- Summarize current high-risk signals for review.
- Are data quality limitations affecting interpretation?
- Explain why a state was flagged as high risk.
- What model outputs contributed to the current alert table?

If LLM generation is unavailable, the app still returns the most relevant retrieved context.

The assistant should not provide medical advice, diagnosis, treatment guidance, or official public health directives.

## 10. Geospatial Risk Mapping

The geospatial map summarizes state-level risk scores and anomaly activity. It helps analysts scan where elevated syndrome activity may require closer review.

Current geospatial capabilities:

- State-level risk aggregation
- Marker size and color based on risk score
- Top high-risk region ranking
- Hover details for analyst review

Future extensions could include county-level boundaries, jurisdiction-level dashboards, facility participation indicators, and privacy-aware small-cell suppression.

## 11. Analyst Briefing and Policy Support

The dashboard can support public health briefing workflows by turning model outputs into concise situational summaries. The RAG assistant and alert intelligence table can help non-technical stakeholders understand:

- What changed
- Where the signal appears
- How unusual the pattern is
- Which syndrome category is involved
- What data quality limitations exist
- What should be reviewed next

This is decision-support context, not automated decision-making.

## 12. Model Testing and Surveillance Engineering Sandbox

The project can serve as a sandbox for evaluating surveillance analytics methods before applying them to approved operational feeds.

Potential engineering uses:

- Test anomaly thresholds
- Compare forecasting methods
- Validate synthetic data scenarios
- Evaluate RAG retrieval quality
- Prototype alert workflows
- Document model governance requirements

## Current Boundaries

- Current data is public aggregate or synthetic and historical.
- Current syndrome scope is respiratory-focused.
- Current maps are state-level.
- Current NLP classification is rule-based.
- Current RAG answers depend on retrieved local context.
- The system is not an official public health, clinical, or government decision system.
