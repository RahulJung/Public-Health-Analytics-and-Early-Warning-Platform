import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# # 1. Resolve the project root so app.py can import local src modules.
# 2. Load environment variables such as OPENAI_API_KEY from .env.
# 3. Build the Streamlit dashboard from processed surveillance data.
ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))
load_dotenv(ROOT / ".env")

from src.config import ROOT_DIR, load_config
from src.pipeline import run_pipeline
from src.forecasting import forecast_series
from src.visualization import plot_trend_with_anomalies, plot_national_overview, plot_forecast
from src.geospatial import STATE_CENTROIDS, latest_geospatial_risk, plot_geospatial_risk
from src.hl7_generator import generate_hl7_batch
from src.nlp_classifier import classify_chief_complaint
from src.rag_assistant import answer_rag_question


APP_TITLE = "Public Health Signal Detection and Response Dashboard"
DISCLAIMER = (
    "This prototype is an independent research and portfolio project using public aggregate or synthetic data. "
    "It is not an official system of HHS, CDC, WHO, NSSP, BioSense, ESSENCE, or any government agency. "
    "It is not intended for clinical decision-making or operational public health response."
)
OFFICIAL_REFERENCES = {
    "HHS Emergency Preparedness & Response": "https://www.hhs.gov/programs/emergency-preparedness/index.html",
    "CDC National Syndromic Surveillance Program": "https://www.cdc.gov/nssp/",
    "CDC BioSense Platform": "https://www.cdc.gov/nssp/php/about/about-nssp-and-the-biosense-platform.html",
    "CDC ESSENCE": "https://www.cdc.gov/nssp/php/onboarding-toolkits/essence.html",
    "CDC Data Modernization": "https://www.cdc.gov/data-modernization/php/about/index.html",
    "WHO Public Health Surveillance": "https://www.who.int/westernpacific/menu/mega-menu/emergencies/surveillance",
    "WHO Epidemic Intelligence": "https://www.who.int/initiatives/eios",
    "USAGov Disasters and Emergencies": "https://www.usa.gov/disasters-and-emergencies",
    "CDC Public Health Efficiency": "https://www.cdc.gov/public-health-data-strategy/php/story/using-ai-to-improve-public-health-efficiency.html",
}


st.set_page_config(page_title=APP_TITLE, layout="wide")


# Define one global visual system for the surveillance product UI.
st.markdown(
    """
    <style>
    .block-container {padding-top: 1.4rem; background: #f6f8f7;}
    h1, h2, h3, h4 {color: #1f2933;}
    .hero {
        background: #ffffff;
        border: 1px solid #d8dfdd;
        border-top: 5px solid #0f766e;
        border-radius: 8px;
        padding: 24px 28px;
        margin-bottom: 16px;
        box-shadow: 0 1px 2px rgba(31, 41, 51, 0.08);
    }
    .hero h1 {
        margin: 0 0 8px 0;
        color: #1f2933;
        font-size: 2.25rem;
        line-height: 1.15;
    }
    .hero p {
        color: #3e4c59;
        font-size: 1.02rem;
        line-height: 1.55;
        margin: 0 0 12px 0;
    }
    .badge {
        display: inline-block;
        background: #e7f5f2;
        color: #115e59;
        border: 1px solid #a7d7cf;
        border-radius: 999px;
        padding: 5px 11px;
        margin-right: 7px;
        font-size: 0.82rem;
        font-weight: 700;
    }
    .section-heading {
        color: #1f2933;
        border-left: 5px solid #0f766e;
        padding-left: 10px;
        margin: 18px 0 10px 0;
        font-size: 1.2rem;
        font-weight: 750;
    }
    .summary-card {
        background: #ffffff;
        border: 1px solid #d8dfdd;
        border-radius: 8px;
        padding: 16px;
        min-height: 140px;
        box-shadow: 0 1px 2px rgba(31, 41, 51, 0.06);
    }
    .summary-card h4 {
        margin: 0 0 8px 0;
        color: #1f2933;
        font-size: 1rem;
    }
    .summary-card p, .small-note {
        color: #52605c;
        font-size: 0.94rem;
        line-height: 1.45;
        margin: 0;
    }
    .context-heading {
        color: #1f2933;
        border-left: 6px solid #0f766e;
        padding-left: 12px;
        margin: 22px 0 8px 0;
        font-size: 1.45rem;
        font-weight: 800;
    }
    .context-intro {
        max-width: 880px;
        color: #3e4c59;
        line-height: 1.45;
        margin-bottom: 8px;
    }
    .context-card {
        background: #ffffff;
        border: 1px solid #d8dfdd;
        border-radius: 8px;
        padding: 18px 18px;
        min-height: 188px;
        box-shadow: 0 2px 8px rgba(31, 41, 51, 0.07);
        margin-top: 12px;
    }
    .context-card h4 {
        color: #1f2933;
        font-size: 1.05rem;
        margin: 0 0 8px 0;
    }
    .context-card p, .context-card li {
        color: #52605c;
        font-size: 0.94rem;
        line-height: 1.45;
    }
    .context-card ul {
        margin: 0;
        padding-left: 1.1rem;
    }
    .disclaimer {
        background: #fff7ed;
        color: #7c2d12;
        border: 1px solid #fed7aa;
        border-radius: 8px;
        padding: 14px 16px;
        margin-top: 16px;
        font-weight: 600;
    }
    .reference-box {
        background: #ffffff;
        border: 1px solid #d8dfdd;
        border-radius: 8px;
        padding: 14px 16px;
        margin-top: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


cfg = load_config()
processed_path = ROOT_DIR / cfg["data"]["processed_path"]


# Sidebar: data source controls.
with st.sidebar:
    st.header("Surveillance Controls")
    st.subheader("Data Source")
    refresh = st.button("Fetch / Refresh CDC Data")
    st.caption("CDC public NSSP Emergency Department Visit Trajectories API.")


@st.cache_data(show_spinner=True)
def load_data(refresh_flag: bool = False):
    #     # If the user requests a refresh or no processed file exists, run the ETL/model pipeline.
    # Otherwise, load the cached processed CSV for fast dashboard startup.
    if refresh_flag or not processed_path.exists():
        return run_pipeline(force_refresh=refresh_flag)
    return pd.read_csv(processed_path, parse_dates=["date"])


try:
    df = load_data(refresh)
except Exception as exc:
    st.error("Unable to load surveillance data.")
    st.exception(exc)
    st.stop()

if df.empty:
    st.warning("No data available.")
    st.stop()

states = sorted(df["state"].dropna().unique())
syndromes = sorted(df["syndrome"].dropna().unique())
min_date = pd.to_datetime(df["date"]).min().date()
max_date = pd.to_datetime(df["date"]).max().date()


# Sidebar: compact grouped filters.
with st.sidebar:
    st.subheader("Geography")
    state = st.selectbox("State / Region", states, index=0)

    st.subheader("Syndrome Category")
    syndrome = st.selectbox("Syndrome / Condition", syndromes, index=0)

    st.subheader("Time Period")
    date_range = st.date_input("Date Range", value=(min_date, max_date), min_value=min_date, max_value=max_date)

    st.subheader("Model Settings")
    forecast_models = ["ARIMA", "Prophet", "Exponential Smoothing"]
    default_forecast_model = cfg["model"].get("forecast_model", "ARIMA")
    default_forecast_index = forecast_models.index(default_forecast_model) if default_forecast_model in forecast_models else 0
    forecast_model = st.selectbox("Forecast Model", forecast_models, index=default_forecast_index)

    st.subheader("Display Options")
    top_n = st.slider("Rows to Show", min_value=5, max_value=25, value=10)


if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
else:
    start_date, end_date = pd.to_datetime(min_date), pd.to_datetime(max_date)

filtered_df = df[(df["date"] >= start_date) & (df["date"] <= end_date)].copy()
if filtered_df.empty:
    st.warning("No records match the selected time period.")
    st.stop()

latest_date = filtered_df["date"].max()
latest_rows = filtered_df[filtered_df["date"] == latest_date]
latest_alerts = latest_rows[latest_rows["any_anomaly"]]
high_risk = latest_rows[latest_rows["risk_level"].isin(["High", "Critical"])]
highest_risk_row = latest_rows.sort_values("risk_score", ascending=False).head(1)
highest_risk_state = highest_risk_row["state"].iloc[0] if not highest_risk_row.empty else "N/A"


def completeness_score(data: pd.DataFrame) -> float:
    # Score only the core fields required for dashboard interpretation.
    required = ["date", "state", "syndrome", "visit_percentage"]
    return 1 - (data[required].isna().sum().sum() / (len(data) * len(required)))


def action_for_risk(risk_level: str) -> str:
    # Convert a model risk label into analyst-facing review guidance.
    mapping = {
        "Critical": "Immediate epidemiologist review recommended.",
        "High": "Validate signal and monitor nearby regions.",
        "Moderate": "Continue trend monitoring.",
        "Low": "No immediate action.",
    }
    return mapping.get(risk_level, "Continue routine monitoring.")


def alert_intelligence(data: pd.DataFrame, risk_filter: list[str] | None = None) -> pd.DataFrame:
    #     # 1. Keep the latest row for every state/syndrome pair.
    # 2. Optionally filter to selected risk levels.
    # 3. Add analyst-facing fields and recommended review actions.
    # 4. Return a compact alert table sorted by priority.
    latest = data.sort_values("date").groupby(["state", "syndrome"], as_index=False).tail(1).copy()
    if risk_filter:
        latest = latest[latest["risk_level"].isin(risk_filter)]
    latest["Visit Count"] = "N/A"
    latest["Anomaly Score"] = latest[["zscore_severity", "isolation_score"]].fillna(0).max(axis=1).round(2)
    latest["Percent Change From Baseline"] = latest["pct_change_7"].round(1)
    latest["Recommended Public Health Action"] = latest["risk_level"].map(action_for_risk)
    latest = latest.sort_values(["risk_score", "Anomaly Score"], ascending=False)
    return latest.rename(columns={
        "state": "Region / State",
        "syndrome": "Syndrome / Category",
        "risk_level": "Risk Level",
    })[
        [
            "Region / State",
            "Syndrome / Category",
            "Visit Count",
            "Anomaly Score",
            "Percent Change From Baseline",
            "Risk Level",
            "Recommended Public Health Action",
        ]
    ]


def model_output_table(data: pd.DataFrame) -> pd.DataFrame:
    #     # 1. Summarize latest model outputs per state/syndrome.
    # 2. Derive interpretable values for anomaly score, baseline, observed value, and deviation.
    # 3. Sort by risk so analysts see highest-priority rows first.
    latest = data.sort_values("date").groupby(["state", "syndrome"], as_index=False).tail(1).copy()
    latest["anomaly_score"] = latest[["zscore_severity", "isolation_score"]].fillna(0).max(axis=1).round(2)
    latest["baseline_visits"] = latest["rolling_mean"].round(2)
    latest["observed_visits"] = latest["visit_percentage"].round(2)
    latest["percent_deviation"] = (
        ((latest["visit_percentage"] - latest["rolling_mean"]) / latest["rolling_mean"].replace(0, pd.NA)) * 100
    ).round(1)
    latest["confidence_priority_score"] = latest["risk_score"].round(1)
    return latest.sort_values("risk_score", ascending=False).rename(columns={
        "state": "region/state",
        "syndrome": "syndrome/category",
        "risk_level": "risk level",
    })[
        [
            "region/state",
            "syndrome/category",
            "anomaly_score",
            "baseline_visits",
            "observed_visits",
            "percent_deviation",
            "risk level",
            "confidence_priority_score",
        ]
    ]


def expected_forecast_direction(forecast_data: pd.DataFrame) -> str:
    # Compare the first and last future forecast values to label short-term direction.
    future = forecast_data[forecast_data["type"] == "future_forecast"].dropna(subset=["forecast"])
    if len(future) < 2:
        return "Insufficient forecast horizon"
    delta = future["forecast"].iloc[-1] - future["forecast"].iloc[0]
    if delta > 0.05:
        return "Increasing"
    if delta < -0.05:
        return "Decreasing"
    return "Stable"


def local_query_assistant(question: str, data: pd.DataFrame) -> str:
    #     # 1. Use simple intent matching for the Analytics Engine quick assistant.
    # 2. Refuse medical advice questions.
    # 3. Answer from current model output tables only.
    normalized = question.lower()
    outputs = model_output_table(data)
    top = outputs.head(5)

    if any(term in normalized for term in ["medical advice", "diagnosis", "treatment", "should i"]):
        return "The assistant summarizes app data and model outputs only. It does not provide medical advice, diagnosis, or official public health guidance."
    if "georgia" in normalized and "flag" in normalized:
        georgia = outputs[outputs["region/state"].str.lower() == "georgia"].head(3)
        if georgia.empty:
            return "Georgia is not among the highest-priority latest model outputs in the selected data window."
        return "Georgia signal summary:\n" + georgia.to_string(index=False)
    if any(term in normalized for term in ["highest", "anomaly score", "regions", "states"]):
        return "Highest current anomaly or risk outputs:\n" + top.to_string(index=False)
    if any(term in normalized for term in ["unusual", "respiratory", "signals", "summarize"]):
        signal_rows = outputs[outputs["risk level"].isin(["Critical", "High", "Moderate"])].head(8)
        if signal_rows.empty:
            return "No moderate, high, or critical latest signals are present in the selected data window."
        return "Current signals for public health review:\n" + signal_rows.to_string(index=False)
    if "overdose" in normalized:
        return "The current public dataset loaded in this app tracks COVID-like illness, influenza-like illness, and RSV-like illness. Overdose-related visits are a future extension and are not available in the current model outputs."
    return "I can summarize current signals, highest anomaly scores, state risk rankings, data quality, and methodology using the app's public/synthetic dataset and generated model outputs."


def data_quality_table(data: pd.DataFrame) -> pd.DataFrame:
    #     # 1. Measure missingness, duplicate keys, geography validity, and timeliness.
    # 2. Return a dashboard-ready quality table for analyst interpretation.
    missing_rate = data["visit_percentage"].isna().mean()
    duplicate_count = int(data.duplicated(subset=["date", "state", "syndrome"]).sum())
    invalid_geography_count = int((~data["state"].isin(STATE_CENTROIDS.keys())).sum())
    latest = pd.to_datetime(data["date"]).max()
    days_since_latest = (pd.Timestamp.today().normalize() - latest.normalize()).days
    completeness = completeness_score(data)
    timeliness = "Current" if days_since_latest <= 14 else "Historical / stale public extract"
    return pd.DataFrame({
        "Quality Measure": [
            "Missing value rate",
            "Duplicate count",
            "Date range",
            "Invalid geography count",
            "Completeness score",
            "Timeliness indicator",
        ],
        "Value": [
            f"{missing_rate:.2%}",
            str(duplicate_count),
            f"{data['date'].min().date()} to {data['date'].max().date()}",
            str(invalid_geography_count),
            f"{completeness:.1%}",
            f"{timeliness} ({days_since_latest} days since latest record)",
        ],
    })


def render_references():
    # Render official public health reference links in one reusable footer component.
    links = " | ".join(f'<a href="{url}" target="_blank">{label}</a>' for label, url in OFFICIAL_REFERENCES.items())
    st.markdown(f'<div class="reference-box">{links}</div>', unsafe_allow_html=True)


# Render the top hero section before the tabbed dashboard.
st.markdown(
    """
    <div class="hero">
        <h1>Public Health Signal Detection and Response Dashboard</h1>
        <p>A syndromic surveillance analytics prototype for detecting unusual emergency department visit patterns, monitoring geographic risk, and supporting public health response planning.</p>
        <p><strong>Built around public health surveillance concepts used for early detection, situational awareness, and emergency preparedness.</strong></p>
        <span class="badge">Syndromic Surveillance</span>
        <span class="badge">Signal Detection</span>
        <span class="badge">Public Health Data Modernization</span>
    </div>
    """,
    unsafe_allow_html=True,
)


overview_tab, signals_tab, analytics_tab, geography_tab, forecasting_tab, quality_tab, methodology_tab, evidence_tab = st.tabs([
    "Overview",
    "Signals",
    "Analytics Engine",
    "Geography",
    "Forecasting",
    "Data Quality",
    "Methodology",
    "Evidence",
])


with overview_tab:
    # Explain the surveillance context and show headline KPIs and map preview.
    st.markdown('<div class="context-heading">Public Health Surveillance Context</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="context-intro">This section explains the surveillance concepts behind the dashboard and how the application turns public emergency department trend data into reviewable signals.</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<span class="badge">Syndromic Surveillance</span><span class="badge">Early Detection</span><span class="badge">Situational Awareness</span><span class="badge">Emergency Preparedness</span>',
        unsafe_allow_html=True,
    )

    ctx_left, ctx_right = st.columns([0.6, 0.4], gap="medium")
    with ctx_left:
        st.markdown(
            '<div class="context-card"><h4>What is Syndromic Surveillance</h4><p>Syndromic surveillance uses near real-time data from emergency departments, urgent care, and other healthcare systems to identify unusual illness patterns before diagnoses are confirmed. By monitoring symptom groupings such as respiratory distress, overdose indicators, or heat-related illness, public health agencies can detect emerging health threats earlier than traditional reporting systems.</p></div>',
            unsafe_allow_html=True,
        )
    with ctx_right:
        st.markdown(
            '<div class="context-card"><h4>Where It Is Applied</h4><ul><li>Outbreak and seasonal illness monitoring</li><li>Environmental health and heat events</li><li>Substance-related harm surveillance</li><li>Emergency preparedness and response</li><li>Cross-jurisdiction situational awareness</li></ul></div>',
            unsafe_allow_html=True,
        )

    st.markdown("")
    with st.container():
        st.markdown(
            '<div class="context-card"><h4>How This Application Supports Surveillance</h4><p>This application combines data processing, syndrome grouping, baseline comparison, anomaly detection, trend analysis, and geospatial visualization in one interface. It highlights unusual emergency department visit patterns, identifies potential signals of concern, and provides contextual outputs that support timely review, monitoring, and response planning.</p></div>',
            unsafe_allow_html=True,
        )

    # Display one row of key operational metrics for the selected filter window.
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Total Records Analyzed", f"{len(filtered_df):,}")
    k2.metric("Latest Data Date", str(pd.to_datetime(latest_date).date()))
    k3.metric("Active Signals", f"{len(latest_alerts):,}")
    k4.metric("Highest Risk State", highest_risk_state)
    k5.metric("Syndromes Monitored", f"{filtered_df['syndrome'].nunique():,}")
    k6.metric("Data Completeness", f"{completeness_score(filtered_df):.1%}")

    # Use the landing heat map as the primary geographic signal preview.
    st.markdown('<div class="section-heading">Geographic Signal Heat Map</div>', unsafe_allow_html=True)
    st.write(
        "The heat map shows the latest relative risk intensity by state or region. Larger and darker markers indicate "
        "higher current model-assigned risk scores. Analysts can use this view to quickly identify where unusual "
        "syndrome activity may require closer review, comparison with nearby regions, or additional data quality checks."
    )
    st.plotly_chart(plot_geospatial_risk(filtered_df), use_container_width=True)

    # Summarize the data pipeline and practical use cases after the map.
    st.markdown('<div class="section-heading">Project Summary</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="summary-card"><h4>Surveillance Scope</h4><p>Monitors COVID-like illness, influenza-like illness, and RSV-like illness using public emergency department trajectory data.</p></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="summary-card"><h4>Signal Workflow</h4><p>Transforms raw CDC records into cleaned state-level trends, engineered features, alerts, forecasts, and risk rankings.</p></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="summary-card"><h4>Response Planning</h4><p>Prioritizes regions for review with alert intelligence, geospatial risk, and natural-language search support.</p></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-heading">Data Pipeline Summary</div>', unsafe_allow_html=True)
    pipeline_rows = pd.DataFrame({
        "Stage": ["Ingest", "Clean", "Group", "Engineer", "Detect", "Score", "Present"],
        "Output": [
            "CDC public API or cached raw CSV",
            "Standardized dates, geography, syndrome, and visit percentage",
            "State/date/syndrome time series",
            "Lag, rolling baseline, Z-score, growth, acceleration",
            "Z-score and Isolation Forest anomaly flags",
            "0-100 risk score and risk level",
            "Dashboard charts, maps, tables, NLP/search tools",
        ],
    })
    st.dataframe(pipeline_rows, use_container_width=True, hide_index=True)

    st.markdown('<div class="section-heading">Top Surveillance Use Cases</div>', unsafe_allow_html=True)
    use_cases = pd.DataFrame({
        "Use Case": ["Respiratory illness monitoring", "Regional signal triage", "Data quality surveillance"],
        "Operational Value": [
            "Track unusual increases in COVID-like, influenza-like, and RSV-like ED visit percentages.",
            "Use risk rankings and maps to prioritize analyst review across states and regions.",
            "Monitor missingness, duplicate keys, timeliness, and completeness before interpreting signals.",
        ],
    })
    st.dataframe(use_cases, use_container_width=True, hide_index=True)


with signals_tab:
    # Let analysts filter current signal rows by risk level and review recommended actions.
    st.markdown('<div class="section-heading">Detected Signals</div>', unsafe_allow_html=True)
    selected_risks = st.multiselect(
        "Risk Level Filter",
        ["Critical", "High", "Moderate", "Low"],
        default=["Critical", "High", "Moderate"],
    )
    signals = alert_intelligence(filtered_df, selected_risks).head(top_n)
    st.dataframe(signals, use_container_width=True, hide_index=True)
    st.caption("Visit counts are unavailable in this public percentage-only source, so the table retains the field and marks it as N/A.")


with analytics_tab:
    # Show model components, latest model outputs, forecast context, and assistant workflows.
    st.markdown('<div class="section-heading">Signal Detection Models</div>', unsafe_allow_html=True)
    a1, a2, a3, a4, a5 = st.columns(5)
    with a1:
        st.markdown('<div class="summary-card"><h4>Rolling Baseline</h4><p>Compares observed visit percentage against recent state/syndrome rolling averages.</p></div>', unsafe_allow_html=True)
    with a2:
        st.markdown('<div class="summary-card"><h4>Z-Score Detection</h4><p>Flags observations that deviate materially from the rolling baseline.</p></div>', unsafe_allow_html=True)
    with a3:
        st.markdown('<div class="summary-card"><h4>Isolation Forest</h4><p>Uses multiple engineered features to identify unusual multivariate patterns.</p></div>', unsafe_allow_html=True)
    with a4:
        st.markdown('<div class="summary-card"><h4>Percent Change</h4><p>Measures recent movement from historical baseline using lagged comparisons.</p></div>', unsafe_allow_html=True)
    with a5:
        st.markdown('<div class="summary-card"><h4>Risk Scoring</h4><p>Combines severity, growth, anomaly flags, and persistence into a priority score.</p></div>', unsafe_allow_html=True)

    st.markdown("#### Latest Model Outputs")
    st.dataframe(model_output_table(filtered_df).head(top_n), use_container_width=True, hide_index=True)

    st.info(
        "This prototype uses interpretable model outputs such as anomaly score, baseline deviation, percent change, "
        "and risk level to support analyst review. The system does not replace epidemiological judgment or operational "
        "public health decision-making."
    )

    st.markdown('<div class="section-heading">Forecasting Models</div>', unsafe_allow_html=True)
    forecast_df, metrics = forecast_series(filtered_df, state=state, syndrome=syndrome, periods=7, model_name=forecast_model)
    if forecast_df.empty:
        st.warning(metrics.get("error", "Forecast unavailable."))
    else:
        st.plotly_chart(plot_forecast(forecast_df, f"Observed and Forecasted Trend: {state} - {syndrome}"), use_container_width=True)
        fm1, fm2, fm3 = st.columns(3)
        fm1.metric("Short-Term Model", metrics.get("model", "N/A"))
        fm2.metric("Expected 7-Day Direction", expected_forecast_direction(forecast_df))
        fm3.metric("Confidence Interval", "Not available")
        st.caption("ARIMA or Prophet can be used for short-term syndrome forecasting. LSTM sequence modeling is a future placeholder for longer historical datasets.")

    st.markdown('<div class="section-heading">Public Health Query Assistant</div>', unsafe_allow_html=True)
    st.caption("The assistant summarizes app data and model outputs only. It does not provide medical advice, diagnosis, or official public health guidance.")
    example_questions = [
        "Which states show unusual respiratory activity this week?",
        "Are overdose-related visits increasing?",
        "What regions have the highest anomaly scores?",
        "Summarize current signals for public health review.",
        "Explain why Georgia was flagged as high risk.",
    ]
    if "analytics_query" not in st.session_state:
        st.session_state["analytics_query"] = example_questions[0]
    qcols = st.columns(3)
    for col, prompt in zip([qcols[0], qcols[1], qcols[2], qcols[0], qcols[1]], example_questions):
        if col.button(prompt, use_container_width=True):
            st.session_state["analytics_query"] = prompt
    with st.form("analytics_query_form"):
        analytics_question = st.text_area("Ask about the app's surveillance data and model outputs", key="analytics_query", height=80)
        ask_submitted = st.form_submit_button("Ask Assistant", use_container_width=True)
    if ask_submitted:
        st.write(local_query_assistant(analytics_question, filtered_df))

    st.markdown('<div class="section-heading">RAG-Based Policy Briefing Assistant</div>', unsafe_allow_html=True)
    st.write(
        "This assistant is designed to help public health policy makers and program leaders ask plain-language "
        "questions about current surveillance signals, data quality, model outputs, and operational context. It can "
        "summarize what the dashboard is showing, identify regions that may need review, and explain why a signal was "
        "prioritized using retrieved project documentation and generated model outputs."
    )
    p1, p2, p3 = st.columns(3)
    with p1:
        st.markdown('<div class="summary-card"><h4>Policy Briefing</h4><p>Summarizes current signals, risk levels, and geographic patterns for leadership review.</p></div>', unsafe_allow_html=True)
    with p2:
        st.markdown('<div class="summary-card"><h4>Resource Planning</h4><p>Highlights regions and syndrome categories that may warrant closer analyst monitoring.</p></div>', unsafe_allow_html=True)
    with p3:
        st.markdown('<div class="summary-card"><h4>Data Confidence</h4><p>Connects signal interpretation with data quality, timeliness, and model transparency outputs.</p></div>', unsafe_allow_html=True)

    policy_questions = [
        "Summarize the current surveillance signals for a policy briefing.",
        "Which regions should be prioritized for additional review?",
        "What data quality issues should leadership know about?",
    ]
    if "policy_rag_query" not in st.session_state:
        st.session_state["policy_rag_query"] = policy_questions[0]
    policy_cols = st.columns(3)
    for col, prompt in zip(policy_cols, policy_questions):
        if col.button(prompt, use_container_width=True):
            st.session_state["policy_rag_query"] = prompt

    doc_paths = [
        ROOT_DIR / "README.md",
        ROOT_DIR / "reports" / "model_summary.md",
        ROOT_DIR / "reports" / "data_dictionary.md",
        ROOT_DIR / "reports" / "public_health_use_cases.md",
    ]
    with st.form("policy_rag_form"):
        policy_query = st.text_area("Ask a policy or program planning question", key="policy_rag_query", height=80)
        policy_submitted = st.form_submit_button("Generate Policy-Focused Summary", use_container_width=True)
    if policy_submitted and policy_query.strip():
        answer, sources, rag_meta = answer_rag_question(
            policy_query,
            doc_paths,
            surveillance_df=filtered_df,
            use_llm=True,
            model=cfg["model"].get("rag_model", "gpt-4.1-mini"),
        )
        if rag_meta["mode"] == "llm_rag":
            st.success(f"Generated with {rag_meta['model']} using retrieved local context.")
        elif rag_meta.get("llm_error"):
            st.warning(f"LLM generation unavailable: {rag_meta['llm_error']}")
        st.write(answer)
        if not sources.empty:
            policy_sources = sources[["source", "chunk_id", "score"]].copy()
            policy_sources["score"] = policy_sources["score"].round(3)
            st.dataframe(policy_sources, use_container_width=True, hide_index=True)

    st.markdown('<div class="section-heading">RAG / NLP Design</div>', unsafe_allow_html=True)
    st.write(
        "The NLP assistant can be extended using retrieval-augmented generation to search model output summaries, "
        "anomaly tables, data quality reports, methodology documentation, and public health reference notes."
    )

    st.markdown('<div class="section-heading">Synthetic Data Generation</div>', unsafe_allow_html=True)
    st.write(
        "GANs are better suited here as a synthetic data generation method rather than as the NLP assistant itself. "
        "A future module could use GAN-based or other generative simulation techniques to create realistic but "
        "non-identifiable emergency department trend patterns for stress-testing anomaly detection and forecasting models."
    )


with geography_tab:
    # Pair the geospatial risk map with state/region risk rankings.
    st.markdown('<div class="section-heading">Geographic Risk</div>', unsafe_allow_html=True)
    st.plotly_chart(plot_geospatial_risk(filtered_df), use_container_width=True)
    ranking = latest_geospatial_risk(filtered_df).sort_values("max_risk_score", ascending=False)
    left, right = st.columns([1, 1])
    with left:
        st.markdown("#### State / Region Risk Ranking")
        st.dataframe(ranking.head(top_n), use_container_width=True, hide_index=True)
    with right:
        st.markdown("#### Top 5 High-Risk Regions")
        st.dataframe(ranking.head(5), use_container_width=True, hide_index=True)


with forecasting_tab:
    # Plot observed trend, anomaly markers, and the selected short-term forecast model.
    st.markdown('<div class="section-heading">Trend Forecast</div>', unsafe_allow_html=True)
    st.plotly_chart(plot_trend_with_anomalies(filtered_df, state, syndrome), use_container_width=True)
    forecast_df, metrics = forecast_series(filtered_df, state=state, syndrome=syndrome, periods=7, model_name=forecast_model)
    if forecast_df.empty:
        st.warning(metrics.get("error", "Forecast unavailable."))
    else:
        st.plotly_chart(plot_forecast(forecast_df, f"7-Day Prototype Forecast: {state} - {syndrome}"), use_container_width=True)
        f1, f2, f3, f4 = st.columns(4)
        f1.metric("Forecast Model", metrics.get("model", "N/A"))
        f2.metric("MAE", f"{metrics.get('mae', 0):.4f}" if metrics.get("mae") is not None else "N/A")
        f3.metric("RMSE", f"{metrics.get('rmse', 0):.4f}" if metrics.get("rmse") is not None else "N/A")
        f4.metric("MAPE", f"{metrics.get('mape', 0):.2f}%" if metrics.get("mape") is not None else "N/A")
        st.info("Forecasting is implemented as a prototype analytics module. Forecast horizon is seven forward periods and follows the cadence available in the source time series.")


with quality_tab:
    # Surface data quality checks before analysts interpret model outputs.
    st.markdown('<div class="section-heading">Data Quality</div>', unsafe_allow_html=True)
    st.dataframe(data_quality_table(filtered_df), use_container_width=True, hide_index=True)
    st.markdown("#### Processed Dataset Coverage")
    coverage = pd.DataFrame({
        "Metric": ["Records", "States / Regions", "Syndromes", "Start Date", "End Date"],
        "Value": [
            f"{len(filtered_df):,}",
            f"{filtered_df['state'].nunique():,}",
            f"{filtered_df['syndrome'].nunique():,}",
            str(filtered_df["date"].min().date()),
            str(filtered_df["date"].max().date()),
        ],
    })
    st.dataframe(coverage, use_container_width=True, hide_index=True)


with methodology_tab:
    # Keep detailed methods in expanders so the main dashboard stays concise.
    st.markdown('<div class="section-heading">Methods</div>', unsafe_allow_html=True)
    with st.expander("Data ingestion", expanded=True):
        st.write("The app loads cached processed data or fetches public CDC NSSP ED trajectory records through the configured Socrata API endpoint.")
    with st.expander("Cleaning and standardization"):
        st.write("Raw CDC wide-format percentage fields are normalized into a long state/date/syndrome structure and invalid or missing required values are removed.")
    with st.expander("Syndrome grouping"):
        st.write("COVID, influenza, and RSV percentage columns are mapped to COVID-like illness, influenza-like illness, and RSV-like illness categories.")
    with st.expander("Feature engineering"):
        st.write("Features include lag values, rolling mean, rolling standard deviation, Z-score, seven-record percent change, and trend acceleration.")
    with st.expander("Anomaly detection"):
        st.write("Z-score detection flags values far from baseline. Isolation Forest adds an unsupervised anomaly signal across engineered features.")
    with st.expander("Forecasting"):
        st.write("ARIMA is the default model. Prophet and Exponential Smoothing are selectable. Exponential Smoothing and rolling average are fallback methods.")
    with st.expander("Risk scoring"):
        st.write("Risk score combines Z-score severity, positive growth, Isolation Forest anomaly status, and recent anomaly persistence into a 0-100 score.")
    with st.expander("Visualization"):
        st.write("The app uses Plotly and Streamlit to render geospatial risk, national trends, state trends, alerts, forecasts, and data quality summaries.")
    with st.expander("NLP, search, and synthetic HL7 modules"):
        tool_tab1, tool_tab2, tool_tab3 = st.tabs(["Chief Complaint NLP", "RAG Search", "Synthetic HL7"])
        with tool_tab1:
            # Demonstrate transparent chief complaint syndrome classification.
            complaint = st.text_area("Chief Complaint", value="fever cough shortness of breath for three days")
            result = classify_chief_complaint(complaint)
            n1, n2 = st.columns(2)
            n1.metric("Predicted Syndrome", result["predicted_syndrome"])
            n2.metric("Confidence", f"{result['confidence']:.0%}")
            st.dataframe(
                pd.DataFrame({
                    "syndrome": list(result["scores"].keys()),
                    "score": list(result["scores"].values()),
                    "matched_terms": [", ".join(result["matched_terms"][key]) for key in result["scores"]],
                }),
                use_container_width=True,
                hide_index=True,
            )
        with tool_tab2:
            # Run retrieval over local docs and optional LLM generation for grounded answers.
            suggestions = [
                "Which states have the highest latest risk scores?",
                "What data quality issues are present in the current dataset?",
                "How does the risk score work?",
            ]
            if "rag_query" not in st.session_state:
                st.session_state["rag_query"] = suggestions[0]
            cols = st.columns(3)
            for col, suggestion in zip(cols, suggestions):
                if col.button(suggestion, use_container_width=True):
                    st.session_state["rag_query"] = suggestion
            use_llm = st.toggle("Generate answer with LLM", value=True)
            llm_model = st.text_input("LLM Model", value=cfg["model"].get("rag_model", "gpt-4.1-mini"))
            doc_paths = [
                ROOT_DIR / "README.md",
                ROOT_DIR / "reports" / "model_summary.md",
                ROOT_DIR / "reports" / "data_dictionary.md",
                ROOT_DIR / "reports" / "public_health_use_cases.md",
            ]
            with st.form("rag_search_form"):
                query = st.text_area("Ask a public health surveillance question", key="rag_query", height=80)
                submitted = st.form_submit_button("Search and Generate Answer", use_container_width=True)
            if submitted and query.strip():
                answer, sources, rag_meta = answer_rag_question(query, doc_paths, surveillance_df=filtered_df, use_llm=use_llm, model=llm_model)
                if rag_meta["mode"] == "llm_rag":
                    st.success(f"Generated with {rag_meta['model']} using retrieved local context.")
                elif rag_meta.get("llm_error"):
                    st.warning(f"LLM generation unavailable: {rag_meta['llm_error']}")
                st.write(answer)
                if not sources.empty:
                    source_view = sources[["source", "chunk_id", "score", "text"]].copy()
                    source_view["score"] = source_view["score"].round(3)
                    source_view["text"] = source_view["text"].str.slice(0, 400)
                    st.dataframe(source_view, use_container_width=True, hide_index=True)
        with tool_tab3:
            # Generate synthetic HL7-style messages for selected surveillance rows.
            hl7_count = st.slider("Messages", min_value=1, max_value=20, value=5)
            hl7_df = generate_hl7_batch(filtered_df, state=state, syndrome=syndrome, count=hl7_count)
            if hl7_df.empty:
                st.warning("No records available for the selected state and syndrome.")
            else:
                st.dataframe(hl7_df[["patient_id", "date", "state", "syndrome", "chief_complaint"]], use_container_width=True, hide_index=True)
                selected_message = st.selectbox("HL7 Message", hl7_df["patient_id"].tolist())
                message = hl7_df.loc[hl7_df["patient_id"] == selected_message, "hl7_message"].iloc[0]
                st.code(message.replace("\r", "\n"), language="text")


with evidence_tab:
    # Summarize technical evidence areas without claiming government affiliation.
    st.markdown('<div class="section-heading">Project Evidence</div>', unsafe_allow_html=True)
    ev1, ev2, ev3, ev4, ev5 = st.columns(5)
    with ev1:
        st.markdown('<div class="summary-card"><h4>Machine Learning Signal Detection</h4><p>Uses Z-score and Isolation Forest outputs to identify unusual state/syndrome patterns.</p></div>', unsafe_allow_html=True)
    with ev2:
        st.markdown('<div class="summary-card"><h4>Forecasting for Early Warning</h4><p>Provides short-term syndrome projections using ARIMA, Prophet, or baseline forecasting.</p></div>', unsafe_allow_html=True)
    with ev3:
        st.markdown('<div class="summary-card"><h4>NLP-Based Data Exploration</h4><p>Supports plain-language questions about surveillance outputs and methodology.</p></div>', unsafe_allow_html=True)
    with ev4:
        st.markdown('<div class="summary-card"><h4>Synthetic Data Simulation</h4><p>Includes synthetic HL7 examples and a roadmap for simulated ED trend generation.</p></div>', unsafe_allow_html=True)
    with ev5:
        st.markdown('<div class="summary-card"><h4>Decision Support Logic</h4><p>Maps risk levels to recommended analyst review actions without replacing expert judgment.</p></div>', unsafe_allow_html=True)

    project_evidence = pd.DataFrame({
        "Evidence Area": [
            "Public health problem addressed",
            "Engineering implementation",
            "Working analytics outputs",
            "Data modernization alignment",
            "Future scalability",
        ],
        "Project Evidence": [
            "Addresses surveillance, emergency preparedness, and earlier review of potential health-threat signals.",
            "Demonstrates ingestion, ETL, feature engineering, anomaly detection, forecasting, risk scoring, mapping, and deployment.",
            "Produces measurable records analyzed, active signals, risk rankings, forecast metrics, and data quality indicators.",
            "Uses public data, structured processing, interoperability concepts, and analyst-facing outputs consistent with modernization goals.",
            "Can be extended with scheduled feeds, API alerts, jurisdiction-level views, expanded NLP, and additional public health datasets.",
        ],
    })
    st.dataframe(project_evidence, use_container_width=True, hide_index=True)

    st.markdown('<div class="section-heading">Use Cases</div>', unsafe_allow_html=True)
    use_case_evidence = pd.DataFrame({
        "Use Case": [
            "Respiratory illness monitoring",
            "Regional signal triage",
            "Preparedness analytics",
            "Data quality review",
        ],
        "Evidence Generated": [
            "State-level syndrome trends and anomaly flags.",
            "Geographic risk ranking and recommended action table.",
            "Forecasting, baseline comparison, and alert context.",
            "Completeness, missingness, duplicate, geography, and timeliness checks.",
        ],
    })
    st.dataframe(use_case_evidence, use_container_width=True, hide_index=True)


# Render alignment cards below the tabs as a compact cross-cutting summary.
st.markdown('<div class="section-heading">Policy and Public Health Alignment</div>', unsafe_allow_html=True)
pa1, pa2, pa3, pa4 = st.columns(4)
with pa1:
    st.markdown('<div class="summary-card"><h4>Early Detection</h4><p>Surfaces unusual emergency department visit patterns through baseline comparison and anomaly review.</p></div>', unsafe_allow_html=True)
with pa2:
    st.markdown('<div class="summary-card"><h4>Data Modernization</h4><p>Transforms public data into structured, reusable, analyst-facing surveillance outputs.</p></div>', unsafe_allow_html=True)
with pa3:
    st.markdown('<div class="summary-card"><h4>Emergency Preparedness</h4><p>Supports preparedness workflows by summarizing signals, regions, trends, and data quality indicators.</p></div>', unsafe_allow_html=True)
with pa4:
    st.markdown('<div class="summary-card"><h4>Cross-Jurisdiction Situational Awareness</h4><p>Compares state-level risk and trends to support broader monitoring across regions.</p></div>', unsafe_allow_html=True)


# Close every page render with official references and the prototype disclaimer.
st.markdown('<div class="section-heading">References & Disclaimer</div>', unsafe_allow_html=True)
render_references()
st.markdown(f'<div class="disclaimer">{DISCLAIMER}</div>', unsafe_allow_html=True)
