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


APP_TITLE = "Public Syndromic Surveillance Platform"
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
    .stApp {background: #f4f7f6;}
    .block-container {padding-top: 1.1rem; max-width: 1240px;}
    h1, h2, h3, h4 {color: #1f2933;}
    .hero {
        background: linear-gradient(135deg, #ffffff 0%, #edf8f5 100%);
        border: 1px solid #d8dfdd;
        border-radius: 12px;
        padding: 28px 34px;
        margin: 0 auto 18px auto;
        max-width: 1160px;
        box-shadow: 0 8px 24px rgba(31, 41, 51, 0.07);
    }
    .hero h1 {
        margin: 0 0 8px 0;
        color: #1f2933;
        font-size: 2.55rem;
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
        border-radius: 6px;
        padding: 5px 9px;
        margin: 0 6px 6px 0;
        font-size: 0.78rem;
        font-weight: 700;
    }
    .hero-actions {
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
        margin: 16px 0 12px 0;
    }
    .hero-button {
        border-radius: 8px;
        padding: 10px 14px;
        font-weight: 800;
        font-size: 0.92rem;
        display: inline-block;
    }
    .hero-button.primary {
        background: #0f766e;
        color: #ffffff;
        border: 1px solid #0f766e;
    }
    .hero-button.secondary {
        background: #ffffff;
        color: #115e59;
        border: 1px solid #b7d8d2;
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
        min-height: 128px;
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
    div[data-testid="stTabs"] div[role="tablist"] {
        flex-wrap: wrap;
        gap: 6px 8px;
    }
    div[data-testid="stTabs"] button[role="tab"] {
        font-size: 1.08rem;
        font-weight: 800;
        color: #334e4a;
        padding: 12px 14px;
        border-radius: 8px 8px 0 0;
    }
    div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
        color: #0f766e;
        border-bottom: 3px solid #0f766e;
        background: #e7f5f2;
    }
    div[data-testid="stTabs"] button[role="tab"] p {
        font-size: 1.08rem;
        font-weight: 800;
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
series_counts = df.groupby(["state", "syndrome"]).size().sort_values(ascending=False)
default_state, default_syndrome = series_counts.index[0] if not series_counts.empty else (states[0], syndromes[0])


# Sidebar: compact grouped filters.
with st.sidebar:
    st.subheader("Geography")
    default_state_index = states.index(default_state) if default_state in states else 0
    state = st.selectbox("State / Region", states, index=default_state_index)

    st.subheader("Syndrome Category")
    default_syndrome_index = syndromes.index(default_syndrome) if default_syndrome in syndromes else 0
    syndrome = st.selectbox("Syndrome / Condition", syndromes, index=default_syndrome_index)

    st.subheader("Time Period")
    date_range = st.date_input("Date Range", value=(min_date, max_date), min_value=min_date, max_value=max_date)

    st.subheader("Model Settings")
    forecast_models = ["Auto (Best Backtest)", "ARIMA", "Prophet", "Exponential Smoothing", "Moving Average"]
    default_forecast_model = cfg["model"].get("forecast_model", "Auto (Best Backtest)")
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


def style_geography_table(table: pd.DataFrame):
    # Highlight risk rows and make geography ranking tables easier to scan.
    risk_colors = {
        "Critical": "background-color: #fee2e2; color: #7f1d1d; font-weight: 700;",
        "High": "background-color: #ffedd5; color: #7c2d12; font-weight: 700;",
        "Moderate": "background-color: #fef3c7; color: #713f12; font-weight: 700;",
        "Low": "background-color: #dcfce7; color: #14532d; font-weight: 700;",
    }

    def highlight_row(row: pd.Series) -> list[str]:
        row_style = risk_colors.get(row.get("Risk Level", ""), "")
        return [
            row_style if column in {"Risk Level", "Risk Score", "Primary Risk Driver"} else ""
            for column in row.index
        ]

    return (
        table.style
        .format({"Observed Value": "{:.3f}", "Risk Score": "{:.1f}"})
        .apply(highlight_row, axis=1)
        .set_properties(**{
            "border-color": "#d8dfdd",
            "font-size": "0.92rem",
        })
        .set_table_styles([
            {
                "selector": "thead th",
                "props": [
                    ("background-color", "#0f766e"),
                    ("color", "#ffffff"),
                    ("font-weight", "800"),
                    ("border-color", "#0f766e"),
                ],
            },
            {
                "selector": "tbody td",
                "props": [
                    ("border-color", "#e5e7eb"),
                    ("padding", "8px 10px"),
                ],
            },
        ])
    )


def style_signal_table(table: pd.DataFrame):
    # Apply the same public-health severity palette to signal review tables.
    risk_colors = {
        "Critical": "background-color: #fee2e2; color: #7f1d1d; font-weight: 700;",
        "High": "background-color: #ffedd5; color: #7c2d12; font-weight: 700;",
        "Moderate": "background-color: #fef3c7; color: #713f12; font-weight: 700;",
        "Low": "background-color: #dcfce7; color: #14532d; font-weight: 700;",
    }

    def highlight_row(row: pd.Series) -> list[str]:
        row_style = risk_colors.get(row.get("Risk Level", ""), "")
        return [
            row_style if column in {"Risk Level", "Risk Score", "Syndrome / Category"} else ""
            for column in row.index
        ]

    return (
        table.style
        .format({
            "Observed Value": "{:.3f}",
            "Baseline": "{:.3f}",
            "Deviation": "{:.1f}%",
            "Risk Score": "{:.1f}",
            "Anomaly Score": "{:.2f}",
        }, na_rep="N/A")
        .apply(highlight_row, axis=1)
        .set_properties(**{
            "border-color": "#d8dfdd",
            "font-size": "0.92rem",
        })
        .set_table_styles([
            {
                "selector": "thead th",
                "props": [
                    ("background-color", "#0f766e"),
                    ("color", "#ffffff"),
                    ("font-weight", "800"),
                    ("border-color", "#0f766e"),
                ],
            },
            {
                "selector": "tbody td",
                "props": [
                    ("border-color", "#e5e7eb"),
                    ("padding", "8px 10px"),
                ],
            },
        ])
    )


def style_model_output_table(table: pd.DataFrame):
    # Style model diagnostics so technical reviewers can scan risk and deviation quickly.
    risk_colors = {
        "Critical": "background-color: #fee2e2; color: #7f1d1d; font-weight: 700;",
        "High": "background-color: #ffedd5; color: #7c2d12; font-weight: 700;",
        "Moderate": "background-color: #fef3c7; color: #713f12; font-weight: 700;",
        "Low": "background-color: #dcfce7; color: #14532d; font-weight: 700;",
    }

    def highlight_row(row: pd.Series) -> list[str]:
        row_style = risk_colors.get(row.get("Risk Level", ""), "")
        return [
            row_style if column in {"Risk Level", "Risk Score", "Syndrome / Category"} else ""
            for column in row.index
        ]

    return (
        table.style
        .format({
            "Anomaly Score": "{:.2f}",
            "Baseline": "{:.3f}",
            "Observed Value": "{:.3f}",
            "Percent Deviation": "{:.1f}%",
            "Risk Score": "{:.1f}",
        }, na_rep="N/A")
        .apply(highlight_row, axis=1)
        .set_properties(**{
            "border-color": "#d8dfdd",
            "font-size": "0.92rem",
        })
        .set_table_styles([
            {
                "selector": "thead th",
                "props": [
                    ("background-color", "#0f766e"),
                    ("color", "#ffffff"),
                    ("font-weight", "800"),
                    ("border-color", "#0f766e"),
                ],
            },
            {
                "selector": "tbody td",
                "props": [
                    ("border-color", "#e5e7eb"),
                    ("padding", "8px 10px"),
                ],
            },
        ])
    )


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
    latest["Observed Value"] = latest["visit_percentage"].round(3)
    latest["Baseline"] = latest["rolling_mean"].round(3)
    latest["Deviation"] = (
        ((latest["visit_percentage"] - latest["rolling_mean"]) / latest["rolling_mean"].replace(0, pd.NA)) * 100
    ).round(1)
    latest["Risk Score"] = latest["risk_score"].round(1)
    latest["Review Note"] = latest["risk_level"].map(action_for_risk)
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
            "Observed Value",
            "Baseline",
            "Deviation",
            "Anomaly Score",
            "Risk Score",
            "Risk Level",
            "Review Note",
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
        "state": "Region / State",
        "syndrome": "Syndrome / Category",
        "risk_level": "Risk Level",
        "anomaly_score": "Anomaly Score",
        "baseline_visits": "Baseline",
        "observed_visits": "Observed Value",
        "percent_deviation": "Percent Deviation",
        "confidence_priority_score": "Risk Score",
    })[
        [
            "Region / State",
            "Syndrome / Category",
            "Anomaly Score",
            "Baseline",
            "Observed Value",
            "Percent Deviation",
            "Risk Level",
            "Risk Score",
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


def rag_doc_paths() -> list[Path]:
    return [
        ROOT_DIR / "README.md",
        ROOT_DIR / "reports" / "model_summary.md",
        ROOT_DIR / "reports" / "data_dictionary.md",
        ROOT_DIR / "reports" / "public_health_use_cases.md",
    ]


CDC_HHS_GUIDANCE_SUMMARY = (
    "CDC NSSP/ESSENCE guidance describes syndromic surveillance workflows where public health professionals analyze, "
    "visualize, share, and interpret emergency department and other health care data. CDC data modernization guidance "
    "emphasizes data that can detect and monitor, investigate and respond, inform and disseminate, and be response-ready. "
    "HHS emergency preparedness materials frame public health preparedness around prevention, preparation, response, and recovery."
)


# Render the top hero section before the tabbed dashboard.
st.markdown(
    """
    <div class="hero">
        <h1>Public Syndromic Surveillance Platform</h1>
        <p><strong>Explainable public health signal intelligence using CDC public aggregate data plus synthetic validation data.</strong></p>
        <p>Transforms emergency department visit patterns into reviewable signals for anomaly detection, geographic risk awareness, forecasting, and responsible analyst workflows.</p>
        <div class="hero-actions">
            <span class="hero-button primary">Explore Signals</span>
            <span class="hero-button secondary">Review Methods</span>
        </div>
        <span class="badge">Syndromic Surveillance</span>
        <span class="badge">CDC + Synthetic Data</span>
        <span class="badge">Explainable Signal Detection</span>
        <span class="badge">Geographic Risk Review</span>
        <span class="badge">Forecasting</span>
    </div>
    """,
    unsafe_allow_html=True,
)


overview_tab, signals_tab, analytics_tab, query_tab, policy_tab, geography_tab, forecasting_tab, quality_tab, methodology_tab, evidence_tab = st.tabs([
    "Overview",
    "Signal Review",
    "Analytics Engine",
    "Query Assistant",
    "Policy Briefing",
    "Geography",
    "Forecasting",
    "Data Quality",
    "Methods",
    "Technical Whitepaper",
])


with overview_tab:
    # Explain the surveillance context and guide reviewers through the app.
    st.markdown('<div class="context-heading">Platform Overview</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="context-intro">The Public Syndromic Surveillance Platform is a public health informatics prototype for turning emergency department visit patterns into explainable signal intelligence. It demonstrates how public aggregate and synthetic data can be cleaned, scored, mapped, forecasted, and summarized for analyst review. For the full research-style explanation, use the <strong>Technical Whitepaper</strong> tab as the primary reference.</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<span class="badge">Syndromic Surveillance</span><span class="badge">Explainable Analytics</span><span class="badge">Analyst Review</span><span class="badge">Public Aggregate + Synthetic Data</span>',
        unsafe_allow_html=True,
    )

    ctx_left, ctx_right = st.columns([0.6, 0.4], gap="medium")
    with ctx_left:
        st.markdown(
            '<div class="context-card"><h4>What Public Health Problem This Addresses</h4><p>Public health teams often need to recognize unusual illness or injury patterns before confirmed case data are complete. Emergency department visit trends and syndrome categories can provide earlier awareness of respiratory illness, overdose-related harms, heat illness, gastrointestinal illness, injury patterns, mental health concerns, and other emerging conditions that may warrant review.</p></div>',
            unsafe_allow_html=True,
        )
    with ctx_right:
        st.markdown(
            '<div class="context-card"><h4>What The Platform Does</h4><ul><li>Transforms public aggregate and synthetic records into standardized analytic features</li><li>Compares current activity against rolling baselines</li><li>Flags unusual patterns using statistical and unsupervised methods</li><li>Ranks state and syndrome signals for review</li><li>Provides maps, forecasts, data quality checks, and grounded assistant summaries</li></ul></div>',
            unsafe_allow_html=True,
        )

    ov1, ov2, ov3, ov4 = st.columns(4)
    with ov1:
        st.markdown('<div class="summary-card"><h4>Early Signal Review</h4><p>Highlights unusual state/syndrome activity before treating it as a confirmed public health event.</p></div>', unsafe_allow_html=True)
    with ov2:
        st.markdown('<div class="summary-card"><h4>Explainable Scoring</h4><p>Shows baseline, deviation, anomaly score, risk score, risk level, and review notes.</p></div>', unsafe_allow_html=True)
    with ov3:
        st.markdown('<div class="summary-card"><h4>Geographic Awareness</h4><p>Identifies where elevated signal scores are concentrated and which syndrome is driving risk.</p></div>', unsafe_allow_html=True)
    with ov4:
        st.markdown('<div class="summary-card"><h4>Responsible AI Assistance</h4><p>Uses grounded retrieval to explain data, methods, quality issues, and policy briefing context.</p></div>', unsafe_allow_html=True)

    st.markdown("")
    with st.container():
        st.markdown(
            '<div class="context-card"><h4>Important Reference For Reviewers</h4><p>The <strong>Technical Whitepaper</strong> tab is the best place to evaluate the prototype in depth. It explains the public health problem, system architecture, data boundary, model choices, RAG assistant design, evaluation approach, governance principles, limitations, and repository evidence in a research-paper format.</p></div>',
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

    # Summarize the data pipeline and practical use cases.
    st.markdown('<div class="section-heading">Project Summary</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="summary-card"><h4>Expanded Surveillance Scope</h4><p>Combines CDC respiratory indicators with synthetic signals for overdose, heat, gastrointestinal, mental health, injury, and other syndromes.</p></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="summary-card"><h4>Explainable Signal Workflow</h4><p>Transforms surveillance-style records into cleaned trends, engineered features, anomaly flags, forecasts, and risk rankings.</p></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="summary-card"><h4>Technical Whitepaper</h4><p>Provides the detailed research-style explanation of methods, model rationale, RAG design, evaluation, governance, and limitations.</p></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-heading">Suggested Review Path</div>', unsafe_allow_html=True)
    review_path = pd.DataFrame({
        "Step": [
            "1. Start here",
            "2. Review the whitepaper",
            "3. Inspect current signals",
            "4. Review geography and forecasts",
            "5. Check data quality and methods",
        ],
        "Where to Go": [
            "Overview",
            "Technical Whitepaper",
            "Signals and Analytics Engine",
            "Geography and Forecasting",
            "Data Quality and Methodology",
        ],
        "What to Look For": [
            "Purpose, scope, data boundary, and key metrics.",
            "Research-style explanation of architecture, models, RAG, governance, limitations, and repository evidence.",
            "Priority state/syndrome signals, observed value, baseline, deviation, anomaly score, and risk level.",
            "Geographic concentration, primary risk drivers, trend direction, and short-term forecast context.",
            "Completeness, missingness, duplicates, feature engineering, anomaly detection, and risk scoring logic.",
        ],
    })
    st.dataframe(review_path, use_container_width=True, hide_index=True)

    st.markdown('<div class="section-heading">Responsible-Use Boundary</div>', unsafe_allow_html=True)
    st.warning(
        "This prototype supports research, demonstration, and technical evaluation only. It does not provide medical "
        "advice, clinical diagnosis, confirmed outbreak detection, official public health guidance, or operational "
        "response instructions. Outputs are intended to support human analyst review."
    )

    st.markdown('<div class="section-heading">Data Pipeline Summary</div>', unsafe_allow_html=True)
    pipeline_rows = pd.DataFrame({
        "Stage": ["Ingest", "Clean", "Group", "Engineer", "Detect", "Score", "Present"],
        "Output": [
            "CDC public API, cached raw CSV, and synthetic syndrome simulation",
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
        "Use Case": ["Respiratory illness monitoring", "Overdose and injury surveillance", "Heat and environmental health", "Data quality surveillance"],
        "Operational Value": [
            "Track unusual increases in COVID-like, influenza-like, and RSV-like ED visit percentages.",
            "Use synthetic suspected overdose and firearm injury signals to stress-test review workflows.",
            "Evaluate heat-related and gastrointestinal increases with seasonal and injected event patterns.",
            "Monitor missingness, duplicate keys, timeliness, and completeness before interpreting signals.",
        ],
    })
    st.dataframe(use_cases, use_container_width=True, hide_index=True)


with signals_tab:
    # Let analysts filter current signal rows by risk level and review recommended actions.
    st.markdown('<div class="section-heading">Detected Signals</div>', unsafe_allow_html=True)
    st.write(
        "This tab lists the current state-and-syndrome patterns that should be reviewed first. A signal means the "
        "latest pattern is unusual relative to recent baseline behavior or model-derived anomaly features. It does "
        "not confirm an outbreak, diagnosis, public health event, or response action."
    )
    st.info(
        "How to use this tab: start with Critical and High signals, compare Observed Value against Baseline, check "
        "Deviation and Anomaly Score, then use the Review Note as a triage cue for analyst follow-up."
    )

    s1, s2, s3 = st.columns(3)
    with s1:
        st.markdown(
            '<div class="summary-card"><h4>Signal Priority</h4><p>Rows are sorted by risk score so the highest-priority state/syndrome patterns appear first.</p></div>',
            unsafe_allow_html=True,
        )
    with s2:
        st.markdown(
            '<div class="summary-card"><h4>Explainable Scoring</h4><p>Risk reflects baseline deviation, recent growth, anomaly model flags, and repeated recent anomalies.</p></div>',
            unsafe_allow_html=True,
        )
    with s3:
        st.markdown(
            '<div class="summary-card"><h4>Analyst Review</h4><p>Review notes support triage and monitoring; they are not medical advice or official response instructions.</p></div>',
            unsafe_allow_html=True,
        )

    selected_risks = st.multiselect(
        "Risk Level Filter",
        ["Critical", "High", "Moderate", "Low"],
        default=["Critical", "High", "Moderate"],
    )
    signals = alert_intelligence(filtered_df, selected_risks).head(top_n)
    st.markdown("### **Column Guide and Signal Interpretation**")
    st.markdown(
        """
        - **Region / State**: geography where the latest state/syndrome record was evaluated.
        - **Syndrome / Category**: surveillance syndrome being reviewed, such as respiratory illness, overdose, heat illness, or injury.
        - **Observed Value**: latest surveillance value for the state/syndrome pair.
        - **Baseline**: recent rolling expected value for the same state/syndrome pair.
        - **Deviation**: percentage difference between the observed value and baseline.
        - **Anomaly Score**: combined indicator using statistical severity and unsupervised anomaly scoring.
        - **Risk Score**: 0-100 prioritization score combining deviation severity, growth, model flags, and persistence.
        - **Risk Level**: Low, Moderate, High, or Critical label derived from the risk score.
        - **Review Note**: suggested analyst review posture based on the risk level.

        **Interpretation:** higher risk rows should be reviewed first, especially when observed values are above baseline
        and anomaly scores are elevated. These signals support prioritization only and require human review.
        """
    )
    st.markdown("#### Priority Signal Review Table")
    st.dataframe(
        style_signal_table(signals),
        use_container_width=True,
        hide_index=True,
        height=460,
    )
    st.caption("Visit counts are unavailable in this public percentage-only source, so the table retains the field and marks it as N/A.")


with analytics_tab:
    # Show model components, latest model outputs, forecast context, and assistant workflows.
    st.markdown('<div class="section-heading">Signal Detection Models</div>', unsafe_allow_html=True)
    st.write(
        "This tab explains the analytics stack behind the platform. It shows how raw surveillance-style values are "
        "converted into baselines, anomaly indicators, risk scores, and short-term trend projections that support "
        "analyst review."
    )
    st.info(
        "How to use this tab: review the model components first, then inspect Latest Model Outputs to see which "
        "state/syndrome pairs have the strongest deviation, anomaly, and risk-scoring evidence. Use Forecasting Models "
        "as short-term context, not as a standalone decision rule."
    )

    e1, e2, e3 = st.columns(3)
    with e1:
        st.markdown(
            '<div class="summary-card"><h4>Baseline First</h4><p>The system compares current activity with recent expected behavior before assigning priority.</p></div>',
            unsafe_allow_html=True,
        )
    with e2:
        st.markdown(
            '<div class="summary-card"><h4>Multiple Signals</h4><p>Z-scores, growth, Isolation Forest flags, and persistence are reviewed together rather than in isolation.</p></div>',
            unsafe_allow_html=True,
        )
    with e3:
        st.markdown(
            '<div class="summary-card"><h4>Explainable Outputs</h4><p>Every score is surfaced as a review aid with visible baseline, deviation, and risk-level context.</p></div>',
            unsafe_allow_html=True,
        )

    st.markdown("### **Analytics Workflow**")
    st.markdown(
        """
        1. **Rolling baseline** estimates recent expected behavior for each state/syndrome pair.
        2. **Z-score detection** measures how far the latest value is from that baseline.
        3. **Isolation Forest** checks whether the multivariate pattern looks unusual across engineered features.
        4. **Risk scoring** combines severity, growth, model flags, and persistence into a 0-100 priority score.
        5. **Forecasting** provides short-term directional context when enough history is available.
        """
    )

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

    model_outputs = model_output_table(filtered_df).head(top_n)
    st.markdown("### **Column Guide and Model Output Interpretation**")
    st.markdown(
        """
        - **Region / State**: state or region for the latest evaluated syndrome record.
        - **Syndrome / Category**: surveillance category being scored.
        - **Anomaly Score**: combined model diagnostic using z-score severity and unsupervised anomaly scoring.
        - **Baseline**: recent rolling expected value for that same state/syndrome pair.
        - **Observed Value**: latest observed surveillance value.
        - **Percent Deviation**: how far the observed value is above or below baseline.
        - **Risk Level**: Low, Moderate, High, or Critical label derived from the 0-100 risk score.
        - **Risk Score**: model-derived priority score used to sort review candidates.

        **Interpretation:** high-risk rows with large positive deviation and high anomaly score should be reviewed first.
        Low scores do not prove absence of risk; they only indicate lower priority under the current model and data window.
        """
    )

    st.markdown("#### Latest Model Outputs")
    st.dataframe(
        style_model_output_table(model_outputs),
        use_container_width=True,
        hide_index=True,
        height=430,
    )

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
        forecast_state = metrics.get("state", state)
        forecast_syndrome = metrics.get("syndrome", syndrome)
        if forecast_state != state or forecast_syndrome != syndrome:
            st.info(f"Selected series has limited history. Showing forecast for {forecast_state} - {forecast_syndrome}.")
        st.plotly_chart(
            plot_forecast(forecast_df, f"Observed and Forecasted Trend: {forecast_state} - {forecast_syndrome}"),
            use_container_width=True,
            key="analytics_forecast",
        )
        fm1, fm2, fm3 = st.columns(3)
        fm1.metric("Short-Term Model", metrics.get("model", "N/A"))
        fm2.metric("Expected 7-Day Direction", expected_forecast_direction(forecast_df))
        fm3.metric("Confidence Interval", "Not available")
        st.markdown(
            """
            **Forecast interpretation:** this chart provides directional context for the selected state and syndrome. The
            forecast model may be automatically selected from candidate statistical baselines using holdout performance.
            Forecast outputs should be compared with anomaly status, data quality, and subject-matter review before use.
            """
        )

    st.markdown('<div class="section-heading">RAG / NLP Design</div>', unsafe_allow_html=True)
    r1, r2 = st.columns(2)
    with r1:
        st.markdown(
            '<div class="summary-card"><h4>Grounded Retrieval</h4><p>The assistant retrieves local documentation, live surveillance snapshots, and methodology context before drafting answers.</p></div>',
            unsafe_allow_html=True,
        )
    with r2:
        st.markdown(
            '<div class="summary-card"><h4>Human-Readable Summaries</h4><p>Responses are structured for analysts and reviewers with caveats, evidence, and next-review steps.</p></div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-heading">Synthetic Data Generation</div>', unsafe_allow_html=True)
    st.markdown(
        """
        Synthetic data expands the prototype beyond the limited public respiratory dataset so the analytics workflow can
        be stress-tested across overdose, injury, heat, gastrointestinal, mental health, and other public health use cases.
        GAN-style generation could be explored later, but the current implementation favors transparent simulation so
        reviewers can understand what was added and why.
        """
    )


with query_tab:
    st.markdown('<div class="section-heading">Public Health Query Assistant</div>', unsafe_allow_html=True)
    st.write(
        "Purpose: this assistant helps reviewers ask plain-language questions about surveillance trends, signal flags, "
        "charts, syndrome categories, data quality, model methods, and limitations. It retrieves project documentation "
        "and the live surveillance snapshot, then explains the answer in human-readable terms."
    )
    st.info(
        "How to use it: ask one focused question at a time. Good questions include: "
        "'What should I review first?', 'Why is this state high risk?', 'Explain the map', "
        "'What does the risk score mean?', or 'Are there data quality concerns?'"
    )
    st.caption(CDC_HHS_GUIDANCE_SUMMARY)

    query_examples = [
        "Summarize the highest-priority current signals in plain language.",
        "Explain the geographic risk map and what a reviewer should look for.",
        "What data quality issues should I check before interpreting signals?",
    ]
    if "public_health_query_text" not in st.session_state:
        st.session_state["public_health_query_text"] = query_examples[0]
    qcols = st.columns(3)
    for col, example in zip(qcols, query_examples):
        if col.button(example, use_container_width=True):
            st.session_state["public_health_query_text"] = example

    with st.form("public_health_query_form"):
        public_health_query = st.text_area(
            "Ask a public health surveillance question",
            key="public_health_query_text",
            height=110,
        )
        query_submitted = st.form_submit_button("Generate Answer", use_container_width=True)

    if query_submitted and public_health_query.strip():
        answer, sources, rag_meta = answer_rag_question(
            public_health_query,
            rag_doc_paths(),
            surveillance_df=filtered_df,
            guidance_context=CDC_HHS_GUIDANCE_SUMMARY,
            use_llm=True,
            model=cfg["model"].get("rag_model", "gpt-4.1-mini"),
            response_mode="query",
        )
        if rag_meta["mode"] == "llm_rag":
            st.success(f"Generated with {rag_meta['model']} using retrieved platform context.")
        elif rag_meta.get("llm_error"):
            st.warning(f"LLM generation unavailable: {rag_meta['llm_error']}")
        st.markdown(answer)
        if not sources.empty:
            with st.expander("Retrieved context used for grounding", expanded=False):
                source_view = sources[["source", "chunk_id", "score", "text"]].copy()
                source_view["score"] = source_view["score"].round(3)
                source_view["text"] = source_view["text"].str.slice(0, 500)
                st.dataframe(source_view, use_container_width=True, hide_index=True)


with policy_tab:
    st.markdown('<div class="section-heading">Policy Briefing Assistant</div>', unsafe_allow_html=True)
    st.write(
        "Purpose: this assistant turns surveillance outputs into a leadership-ready briefing format. It is designed for "
        "planning conversations, program review, preparedness context, and governance discussion. It does not issue "
        "official guidance or response instructions."
    )
    st.info(
        "How to use it: ask for a short briefing, review priorities, policy planning considerations, resource planning "
        "questions, or governance caveats. The answer should be treated as a draft analytic summary for human review."
    )
    st.caption(CDC_HHS_GUIDANCE_SUMMARY)

    policy_examples = [
        "Create a policy briefing on current surveillance signals and review priorities.",
        "Which regions or syndrome categories should leadership monitor more closely?",
        "What governance, validation, and responsible-use caveats should accompany this analysis?",
    ]
    if "policy_briefing_query_text" not in st.session_state:
        st.session_state["policy_briefing_query_text"] = policy_examples[0]
    pcols = st.columns(3)
    for col, example in zip(pcols, policy_examples):
        if col.button(example, use_container_width=True):
            st.session_state["policy_briefing_query_text"] = example

    with st.form("policy_briefing_form"):
        policy_query = st.text_area(
            "Ask for a policy or leadership briefing",
            key="policy_briefing_query_text",
            height=110,
        )
        policy_submitted = st.form_submit_button("Generate Briefing", use_container_width=True)

    if policy_submitted and policy_query.strip():
        answer, sources, rag_meta = answer_rag_question(
            policy_query,
            rag_doc_paths(),
            surveillance_df=filtered_df,
            guidance_context=CDC_HHS_GUIDANCE_SUMMARY,
            use_llm=True,
            model=cfg["model"].get("rag_model", "gpt-4.1-mini"),
            response_mode="policy",
        )
        if rag_meta["mode"] == "llm_rag":
            st.success(f"Generated with {rag_meta['model']} using retrieved platform context.")
        elif rag_meta.get("llm_error"):
            st.warning(f"LLM generation unavailable: {rag_meta['llm_error']}")
        st.markdown(answer)
        if not sources.empty:
            with st.expander("Retrieved context used for grounding", expanded=False):
                source_view = sources[["source", "chunk_id", "score", "text"]].copy()
                source_view["score"] = source_view["score"].round(3)
                source_view["text"] = source_view["text"].str.slice(0, 500)
                st.dataframe(source_view, use_container_width=True, hide_index=True)


with geography_tab:
    # Pair the geospatial risk map with state/region risk rankings.
    st.markdown('<div class="section-heading">Geographic Risk</div>', unsafe_allow_html=True)
    st.write(
        "This view helps reviewers understand where elevated surveillance signals are concentrated across states or "
        "regions. It is designed for geographic prioritization and situational awareness, not diagnosis, outbreak "
        "confirmation, or official response guidance."
    )
    st.info(
        "How to read this tab: the map and rankings use all syndrome categories available in the selected date range. "
        "For each state, the displayed risk score is the highest latest risk score across its syndrome categories. "
        "The table identifies the primary syndrome driving that state-level score."
    )

    g1, g2, g3 = st.columns(3)
    with g1:
        st.markdown(
            '<div class="summary-card"><h4>What Is Ranked?</h4><p>Each row summarizes the latest available state-level signal profile within the selected date range.</p></div>',
            unsafe_allow_html=True,
        )
    with g2:
        st.markdown(
            '<div class="summary-card"><h4>What Drives Risk?</h4><p>Risk combines baseline deviation, recent growth, unsupervised anomaly flags, and repeated recent anomalies.</p></div>',
            unsafe_allow_html=True,
        )
    with g3:
        st.markdown(
            '<div class="summary-card"><h4>What Syndrome?</h4><p>Primary Risk Driver shows which syndrome category has the highest current risk score for the state.</p></div>',
            unsafe_allow_html=True,
        )

    st.plotly_chart(plot_geospatial_risk(filtered_df), use_container_width=True, key="geography_geospatial_risk")
    ranking = latest_geospatial_risk(filtered_df).sort_values("max_risk_score", ascending=False)
    ranking_view = ranking[[
        "state",
        "latest_date",
        "primary_syndrome",
        "primary_observed_value",
        "max_risk_score",
        "primary_risk_level",
        "high_risk_count",
        "anomaly_count",
    ]].rename(columns={
        "state": "Region / State",
        "latest_date": "Latest Date",
        "primary_syndrome": "Primary Risk Driver",
        "primary_observed_value": "Observed Value",
        "max_risk_score": "Risk Score",
        "primary_risk_level": "Risk Level",
        "high_risk_count": "High / Critical Syndrome Count",
        "anomaly_count": "Anomaly Count",
    })
    ranking_view["Observed Value"] = ranking_view["Observed Value"].round(3)
    ranking_view["Risk Score"] = ranking_view["Risk Score"].round(1)

    st.markdown("### **Column Guide and Risk Interpretation**")
    st.markdown(
        """
        - **Region / State**: geography represented by the map marker and ranking row.
        - **Latest Date**: most recent record used for that state within the selected date range.
        - **Primary Risk Driver**: syndrome category with the highest latest risk score for that state.
        - **Observed Value**: latest visit percentage or surveillance value for the primary risk driver.
        - **Risk Score**: 0-100 prioritization score based on statistical deviation, growth, model anomaly flag, and persistence.
        - **Risk Level**: Low, Moderate, High, or Critical label derived from the score range.
        - **High / Critical Syndrome Count**: number of syndrome categories in that state currently labeled High or Critical.
        - **Anomaly Count**: number of syndrome categories in that state with a latest anomaly flag.

        **Interpretation:** a higher score means the state has at least one syndrome category that looks unusual relative
        to its recent baseline and model-derived features. It does not mean a confirmed outbreak, diagnosis, or response action.
        """
    )

    st.markdown("#### Top 5 Geographic Review Priorities")
    st.dataframe(
        style_geography_table(ranking_view.head(5)),
        use_container_width=True,
        hide_index=True,
        height=250,
    )

    st.markdown("#### State / Region Risk Ranking")
    st.dataframe(
        style_geography_table(ranking_view.head(top_n)),
        use_container_width=True,
        hide_index=True,
        height=430,
    )


with forecasting_tab:
    # Plot observed trend, anomaly markers, and the selected short-term forecast model.
    st.markdown('<div class="section-heading">Trend Forecast</div>', unsafe_allow_html=True)
    st.write(
        "This tab helps reviewers understand the recent direction of a selected syndrome trend and compare observed "
        "activity with a short-term prototype forecast. It is meant for planning and analyst review, not authoritative "
        "prediction or operational decision-making."
    )
    st.info(
        f"Current selection: **{state}** and **{syndrome}** for records dated **{start_date.date()}** through "
        f"**{end_date.date()}**. The first chart shows observed trend behavior and anomaly markers. The second chart "
        "shows a seven-period forward projection when enough history is available."
    )

    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        st.markdown(
            '<div class="summary-card"><h4>Observed Trend</h4><p>Shows how the selected syndrome has moved over time within the selected region and date range.</p></div>',
            unsafe_allow_html=True,
        )
    with fc2:
        st.markdown(
            '<div class="summary-card"><h4>Anomaly Context</h4><p>Highlights points where recent activity appears unusual relative to baseline or model-derived features.</p></div>',
            unsafe_allow_html=True,
        )
    with fc3:
        st.markdown(
            '<div class="summary-card"><h4>Short-Term Projection</h4><p>Extends the recent pattern forward for seven periods as a directional planning aid.</p></div>',
            unsafe_allow_html=True,
        )

    st.markdown("### **How to Interpret the Forecasting Outputs**")
    st.markdown(
        """
        - **Observed line**: historical surveillance value for the selected state and syndrome.
        - **Anomaly markers**: records flagged by statistical or unsupervised anomaly detection.
        - **Forecast line**: short-term projection generated from available historical values.
        - **Forecast horizon**: seven forward periods, based on the cadence of the source time series.
        - **Model metrics**: error measures calculated from available fitted or back-tested values when supported.

        **Interpretation:** an upward projection suggests the selected syndrome may warrant closer monitoring. A flat or
        downward projection suggests lower near-term growth pressure. Forecasts should always be checked against anomaly
        status, data completeness, recent reporting changes, and public health context.
        """
    )

    st.plotly_chart(
        plot_trend_with_anomalies(filtered_df, state, syndrome),
        use_container_width=True,
        key="forecasting_trend_with_anomalies",
    )
    forecast_df, metrics = forecast_series(filtered_df, state=state, syndrome=syndrome, periods=7, model_name=forecast_model)
    if forecast_df.empty:
        st.warning(metrics.get("error", "Forecast unavailable."))
    else:
        forecast_state = metrics.get("state", state)
        forecast_syndrome = metrics.get("syndrome", syndrome)
        if forecast_state != state or forecast_syndrome != syndrome:
            st.info(f"Selected series has limited history. Showing forecast for {forecast_state} - {forecast_syndrome}.")
        st.plotly_chart(
            plot_forecast(forecast_df, f"7-Day Prototype Forecast: {forecast_state} - {forecast_syndrome}"),
            use_container_width=True,
            key="forecasting_forecast",
        )
        f1, f2, f3, f4, f5 = st.columns(5)
        f1.metric("Forecast Model", metrics.get("model", "N/A"))
        f2.metric("MAE", f"{metrics.get('mae', 0):.4f}" if metrics.get("mae") is not None else "N/A")
        f3.metric("RMSE", f"{metrics.get('rmse', 0):.4f}" if metrics.get("rmse") is not None else "N/A")
        f4.metric("sMAPE", f"{metrics.get('smape', 0):.2f}%" if metrics.get("smape") is not None else "N/A")
        f5.metric("WMAPE", f"{metrics.get('wmape', 0):.2f}%" if metrics.get("wmape") is not None else "N/A")
        model_details = []
        if metrics.get("requested_model"):
            model_details.append(f"Requested: {metrics['requested_model']}")
        if metrics.get("arima_order"):
            model_details.append(f"ARIMA order: {metrics['arima_order']}")
        if metrics.get("window"):
            model_details.append(f"Moving average window: {metrics['window']}")
        if metrics.get("selection_method"):
            model_details.append(f"Selection: {metrics['selection_method']}")
        if metrics.get("fallback_reason"):
            model_details.append("Fallback was used because the requested model could not complete.")
        if model_details:
            st.caption(" | ".join(model_details))
        st.markdown("### **Model Metrics and Limitations**")
        st.markdown(
            """
            - **MAE**: average absolute difference between observed and estimated values. Lower is better.
            - **RMSE**: error measure that penalizes larger misses more heavily. Lower is better.
            - **sMAPE**: symmetric percentage error. It is more stable than standard MAPE when values are small.
            - **WMAPE**: weighted percentage error across the holdout period. It is often easier to interpret for surveillance percentages.
            - **Auto model selection**: compares candidate ARIMA, exponential smoothing, moving average, naive, and drift baselines against the same holdout window, then chooses the lowest-error option.
            - **Fallback behavior**: if the selected state/syndrome has limited history, the platform may show the closest
              available series with enough records for forecasting.

            **Limitation:** forecasts are short-term analytic projections based on available historical patterns. They are
            directional planning aids, not confirmed future events, clinical predictions, or official public health guidance.
            """
        )


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
    # Present the project as a research-style technical whitepaper for reviewers.
    st.markdown('<div class="section-heading">Technical Whitepaper</div>', unsafe_allow_html=True)
    st.markdown(
        """
        # Public Syndromic Surveillance Platform
        **Explainable Public Health Signal Intelligence Using Public Aggregate and Synthetic Data**

        ## Abstract
        The Public Syndromic Surveillance Platform is an applied public health informatics prototype that demonstrates
        how surveillance-style emergency department trend data can be transformed into explainable signal intelligence
        for analyst review. The system combines data ingestion, cleaning, feature engineering, statistical anomaly
        detection, unsupervised machine learning, risk scoring, geospatial review, short-term forecasting, data quality
        assessment, and retrieval-augmented natural-language assistance.

        The purpose of the platform is not to make clinical determinations or issue official response guidance. Its
        purpose is to show how modern surveillance analytics can support earlier review of unusual patterns, transparent
        prioritization, and responsible interpretation using public aggregate and synthetic data boundaries.
        """
    )

    w1, w2, w3, w4 = st.columns(4)
    with w1:
        st.markdown('<div class="summary-card"><h4>Research Problem</h4><p>Public health teams need earlier awareness of unusual visit patterns before final confirmed case data are available.</p></div>', unsafe_allow_html=True)
    with w2:
        st.markdown('<div class="summary-card"><h4>Technical Contribution</h4><p>The platform connects ETL, explainable models, maps, forecasts, quality checks, and RAG assistance in one workflow.</p></div>', unsafe_allow_html=True)
    with w3:
        st.markdown('<div class="summary-card"><h4>Responsible Use</h4><p>Outputs are framed as prioritization aids for human analyst review, with explicit limitations and data boundaries.</p></div>', unsafe_allow_html=True)
    with w4:
        st.markdown('<div class="summary-card"><h4>Repository Evidence</h4><p>The implementation is backed by modular source files for ingestion, cleaning, features, models, scoring, maps, and RAG.</p></div>', unsafe_allow_html=True)

    st.markdown("## 1. Public Health Problem and Motivation")
    st.markdown(
        """
        Syndromic surveillance is valuable because it can surface unusual symptom or visit patterns before traditional
        confirmed diagnosis workflows are complete. Emergency department visit trends, chief complaint concepts, and
        syndrome groupings can provide earlier situational awareness for respiratory illness, overdose-related harms,
        heat illness, gastrointestinal illness, injury patterns, mental health presentations, and other emerging concerns.

        A practical surveillance analytics system must do more than display a chart. It needs to help analysts answer:

        - Which state, region, or syndrome should be reviewed first?
        - Is the current value unusual relative to its own recent baseline?
        - Is the signal isolated, persistent, growing, or geographically concentrated?
        - Are model outputs explainable enough for review?
        - Is the data complete enough to support interpretation?
        - Can non-technical reviewers understand what the system does and what it does not do?

        This prototype addresses those questions by organizing the surveillance workflow into reviewable analytic layers
        rather than presenting a single opaque model output.
        """
    )

    st.markdown("## 2. System Architecture")
    st.markdown(
        """
        The platform follows a modular public health informatics pipeline:

        **Data ingestion -> ETL and quality checks -> feature engineering -> anomaly detection -> risk scoring -> geospatial review -> forecasting -> analyst-facing presentation -> RAG-assisted explanation**

        Each stage produces an interpretable artifact. Raw records are converted into standardized analytic records.
        Features are derived at the state/syndrome/date level. Statistical and machine learning signals are calculated
        from those features. Risk scoring converts model outputs into a bounded prioritization score. The user interface
        exposes each stage through tables, charts, maps, explanations, and assistant workflows.
        """
    )

    architecture_evidence = pd.DataFrame({
        "Pipeline Stage": [
            "Data ingestion",
            "Cleaning and standardization",
            "Feature engineering",
            "Anomaly detection",
            "Risk scoring",
            "Geospatial review",
            "Forecasting",
            "RAG assistance",
            "Presentation layer",
        ],
        "Implementation Evidence": [
            "src/data_ingestion.py loads public CDC data, cached raw files, and optional synthetic syndrome data.",
            "src/data_cleaning.py standardizes dates, geography, syndrome categories, and visit percentage fields.",
            "src/feature_engineering.py creates lag, rolling baseline, z-score, percent change, and acceleration features.",
            "src/anomaly_detection.py implements z-score and Isolation Forest anomaly detection.",
            "src/risk_scoring.py converts model signals into a 0-100 score and Low/Moderate/High/Critical levels.",
            "src/geospatial.py aggregates state-level risk and identifies the primary syndrome risk driver.",
            "src/forecasting.py compares ARIMA, exponential smoothing, moving-average, naive, and drift candidates.",
            "src/rag_assistant.py retrieves documentation and live surveillance snapshots for grounded answers.",
            "app.py renders the Streamlit workflow, tabs, filters, charts, tables, and assistant interfaces.",
        ],
    })
    st.dataframe(architecture_evidence, use_container_width=True, hide_index=True)

    st.markdown("## 3. Data Boundary and Dataset Strategy")
    st.markdown(
        """
        The application uses public aggregate data and synthetic validation data only. It does not use live operational
        surveillance feeds, patient-level records, protected health information, or clinical records.

        Public aggregate data provides realism for respiratory surveillance-style trends, but it can be limited in
        syndrome breadth and historical depth. Synthetic data is therefore used to expand evaluation coverage across
        additional applied surveillance use cases. The synthetic records are not intended to represent real patients or
        real events. They provide controlled patterns that allow the pipeline to be tested for overdose, heat illness,
        gastrointestinal illness, mental health crisis, suicide-related behavior, firearm injury, rash/fever, and
        neurological symptom workflows.

        This design choice improves prototype evaluation because the system can be assessed across multiple public
        health scenarios without crossing privacy boundaries or implying access to operational feeds.
        """
    )

    st.markdown("## 4. Model Selection and Rationale")
    st.markdown(
        """
        The platform uses a layered modeling strategy rather than relying on a single black-box model. This is intentional.
        Public health surveillance review requires transparency, robustness, and interpretability. Analysts need to know
        why a signal was prioritized, what baseline it was compared against, and whether the signal is persistent or
        potentially model-driven noise.

        ### Rolling Baseline
        The rolling baseline estimates recent expected behavior for each state/syndrome pair. It is used because
        surveillance trends are local to geography, syndrome, seasonality, and reporting behavior. Comparing a value to
        a national average would be less useful than comparing it to its own recent history.

        ### Z-Score Detection
        Z-score detection measures how far the observed value is from the recent rolling baseline in standard deviation
        units. It is included because it is transparent, easy to explain, and useful for identifying large deviations.
        A reviewer can understand that a high z-score means the current value is unusually far from recent expected
        behavior.

        ### Isolation Forest
        Isolation Forest is used as an unsupervised anomaly detection model because public health surveillance often lacks
        labeled examples of every event type. It can identify unusual multivariate patterns across engineered features
        such as observed value, rolling baseline, z-score, percent change, and trend acceleration. It is not treated as
        a definitive event detector. It is treated as one model signal among several.

        ### Risk Scoring
        Risk scoring combines statistical severity, positive growth, unsupervised anomaly flags, and recent persistence
        into a bounded 0-100 score. This makes model output easier to prioritize and compare. The score is converted into
        Low, Moderate, High, and Critical labels so non-technical reviewers can understand the triage level while still
        seeing the underlying numeric value.

        ### Forecasting
        Forecasting is used for short-term directional context. The platform now supports automatic backtest-based model
        selection across ARIMA candidates, exponential smoothing variants, moving-average baselines, naive baseline, and
        drift baseline. This improves reliability because noisy surveillance series do not always fit one model family.
        The selected model is chosen using holdout error, with MAE and sMAPE used to support practical comparison.

        Forecasting remains a planning aid. It does not predict confirmed events or replace analyst judgment.
        """
    )

    model_rationale = pd.DataFrame({
        "Model / Method": [
            "Rolling baseline",
            "Z-score",
            "Isolation Forest",
            "Risk score",
            "Auto forecasting",
            "Geospatial aggregation",
        ],
        "Why It Is Used": [
            "Provides a recent expected value for each state/syndrome pair.",
            "Offers a transparent measure of deviation from baseline.",
            "Detects unusual multivariate patterns without requiring labeled outbreak data.",
            "Combines several model signals into one prioritization measure for review.",
            "Chooses the best short-term statistical option based on holdout performance.",
            "Shows where elevated scores are concentrated and which syndrome is driving the state-level risk.",
        ],
        "Responsible Interpretation": [
            "Baseline is context, not proof of normal or abnormal conditions.",
            "Large deviations require review for reporting changes, artifacts, and public health context.",
            "Unsupervised flags are screening signals, not confirmations.",
            "Risk levels prioritize review; they do not mandate response.",
            "Forecasts are directional and short-term only.",
            "Geographic concentration supports situational awareness, not diagnosis.",
        ],
    })
    st.dataframe(model_rationale, use_container_width=True, hide_index=True)

    st.markdown("## 5. Model and LLM Inventory")
    st.markdown(
        """
        The platform uses multiple model types because public health surveillance requires different analytic functions:
        baseline estimation, anomaly screening, prioritization, short-term projection, retrieval, and natural-language
        explanation. A single model would be less transparent and less appropriate for this workflow.

        The current default LLM-backed assistant model is **gpt-4.1-mini**, configured through `config.yaml` as
        `model.rag_model`. If an `OPENAI_API_KEY` is not available, the assistant automatically falls back to a
        retrieval-only answer built from local documentation and the live surveillance snapshot. This fallback behavior
        keeps the application usable without requiring an external LLM call.
        """
    )

    model_inventory = pd.DataFrame({
        "Component": [
            "Rolling baseline",
            "Z-score anomaly detection",
            "Isolation Forest",
            "Risk scoring model",
            "Auto forecasting selector",
            "ARIMA",
            "Exponential Smoothing",
            "Moving Average / Naive / Drift baselines",
            "TF-IDF retrieval",
            "LLM response generation",
            "Retrieval-only fallback",
            "Synthetic data simulation",
        ],
        "Model / Method Used": [
            "Seven-period rolling mean and rolling standard deviation by state/syndrome.",
            "Absolute z-score thresholding using configurable threshold from config.yaml.",
            "scikit-learn IsolationForest with standardized engineered features and configurable contamination.",
            "Weighted deterministic score combining z-score severity, growth, Isolation Forest flag, and persistence.",
            "Backtest-based selector comparing candidate forecasting methods on the holdout window.",
            "statsmodels ARIMA with multiple candidate orders, selected by holdout error.",
            "statsmodels ExponentialSmoothing with level/trend/damped trend variants.",
            "Simple statistical baselines used when noisy or short series outperform complex models.",
            "scikit-learn TfidfVectorizer with cosine similarity over local documentation and runtime context.",
            "OpenAI Responses API using gpt-4.1-mini by default for grounded natural-language summaries.",
            "Formatted extractive/summary response from top retrieved local context chunks.",
            "Transparent rule-based synthetic syndrome time-series generation for validation coverage.",
        ],
        "Reason For Use": [
            "Provides an interpretable recent expected value for each local surveillance series.",
            "Gives reviewers a transparent measure of how unusual the current value is.",
            "Finds unusual multivariate patterns when labeled event examples are unavailable.",
            "Turns multiple model signals into a reviewable priority score without hiding the inputs.",
            "Avoids assuming one forecasting model is best across all syndromes and geographies.",
            "Useful for short-term time-series projection when autoregressive structure is present.",
            "Works well for smoothed level/trend behavior and provides a stable statistical fallback.",
            "Often competitive for noisy surveillance percentages and easier to interpret.",
            "Grounds assistant answers in repository evidence and live app context before generation.",
            "Improves readability, synthesis, and briefing quality for mixed technical/non-technical reviewers.",
            "Preserves transparency and usability when an LLM is unavailable or disabled.",
            "Expands testing across applied surveillance use cases without using patient-level data.",
        ],
        "Output Interpretation": [
            "Baseline context only; not proof that current activity is normal or abnormal.",
            "Screening flag for unusually large deviation; requires review for artifacts and context.",
            "Unsupervised signal; not confirmation of an outbreak or public health event.",
            "Prioritization aid for analyst review; not an automated decision rule.",
            "Chooses the lowest-error candidate for the available history, not a universal best model.",
            "Directional forecast only; sensitive to series length, volatility, and recent changes.",
            "Directional forecast only; may understate sharp changes or sudden outbreaks.",
            "Benchmark and fallback forecasts that support robustness and interpretability.",
            "Evidence retrieval step; quality depends on available documentation and indexed context.",
            "Draft explanation only; answer must remain grounded and human-reviewed.",
            "Context summary only; less fluent but more directly traceable.",
            "Validation support only; synthetic patterns do not prove real-world performance.",
        ],
    })
    st.dataframe(model_inventory, use_container_width=True, hide_index=True)

    st.markdown("### LLM Prompting and Grounding Design")
    st.markdown(
        """
        The RAG assistant prompt is intentionally constrained. It instructs the LLM to use only retrieved project context
        and the live surveillance snapshot. It asks for clear headings, short paragraphs, concrete bullets, and explicit
        caveats. It also instructs the model not to provide medical advice, diagnosis, treatment guidance, official
        directives, or autonomous response instructions.

        The prompt structure asks for:

        1. **Short answer**
        2. **What the platform data or documentation supports**
        3. **What an analyst or reviewer should check next**
        4. **Caveats and responsible-use boundary**

        This design makes the assistant useful for explanation and briefing while keeping the output aligned with
        analyst-reviewed public health surveillance practice.
        """
    )

    st.markdown("## 6. Why Retrieval-Augmented Generation Is Included")
    st.markdown(
        """
        Retrieval-augmented generation is included because surveillance platforms often contain more context than a
        reviewer can absorb from charts alone. A reviewer may ask: why was a region flagged, how is risk calculated,
        what data quality issues matter, what limitations apply, or how should a briefing summarize current signals?

        The assistant does not generate answers from memory alone. It retrieves project documentation, model summaries,
        data dictionaries, use case descriptions, and a live surveillance snapshot before drafting a response. This
        grounding design is important for three reasons:

        - **Traceability:** answers can be connected to local project evidence and current dashboard context.
        - **Usability:** non-technical reviewers can ask questions in plain language instead of interpreting every table manually.
        - **Governance:** responses are constrained to analyst review, data quality, methodology, limitations, and planning context.

        The RAG assistant is split into two workflows. The Public Health Query Assistant explains data, charts, methods,
        and limitations. The Policy Briefing Assistant summarizes review priorities and governance considerations in a
        leadership-ready format. Both assistants are designed to avoid medical advice, diagnosis, official directives,
        and autonomous response recommendations.
        """
    )

    st.markdown("## 7. Applied Surveillance Use Cases")
    use_case_evidence = pd.DataFrame({
        "Use Case": [
            "Opioid overdose spike detection",
            "Respiratory illness surveillance",
            "Mental health and suicide-related surveillance",
            "Firearm injury and violence surveillance",
            "Heat illness and environmental health",
            "Gastrointestinal illness monitoring",
        ],
        "How the Prototype Supports Review": [
            "Synthetic overdose-related syndrome trends can be scored for unusual increases by geography and time.",
            "Public aggregate respiratory indicators support trend review, anomaly flags, and short-term forecasting.",
            "Synthetic mental health and suicide-related categories test whether the workflow can surface regional changes.",
            "Synthetic injury categories demonstrate how nonfatal injury signals could be prioritized for review.",
            "Synthetic heat-related illness patterns test seasonal and event-like increases across regions.",
            "Synthetic gastrointestinal illness patterns test outbreak-style increases and persistence logic.",
        ],
        "Analyst-Reviewed Output": [
            "Signal table, risk level, deviation, geospatial rank, and briefing summary.",
            "Observed trend, anomaly markers, forecast direction, and state-level ranking.",
            "Priority table, geographic concentration, and responsible-use caveats.",
            "Regional signal triage and data quality checks before interpretation.",
            "Map concentration, rolling baseline deviation, and short-term monitoring context.",
            "Anomaly persistence and deviation from recent baseline.",
        ],
    })
    st.dataframe(use_case_evidence, use_container_width=True, hide_index=True)

    st.markdown("## 8. Evaluation Approach")
    st.markdown(
        """
        The evaluation approach focuses on whether the prototype produces interpretable, reviewable outputs rather than
        claiming operational validation. The current repository supports:

        - **Data coverage checks:** records analyzed, date range, states/regions, syndrome count, missingness, duplicates.
        - **Model output checks:** anomaly flags, risk levels, baseline deviation, persistence, and geographic ranking.
        - **Forecast checks:** holdout MAE, RMSE, sMAPE, WMAPE, selected model, and fallback behavior.
        - **Interface checks:** separate tabs for signal review, geography, forecasting, data quality, methods, RAG, and whitepaper evidence.
        - **Unit tests:** tests for synthetic data generation, data cleaning, forecasting behavior, risk scoring, and geospatial features.

        A production-grade evaluation would require jurisdiction-specific validation, threshold calibration, analyst
        feedback loops, drift monitoring, prospective testing, and documented governance review.
        """
    )

    st.markdown("## 9. Responsible AI and Governance")
    st.markdown(
        """
        Responsible surveillance analytics requires more than model accuracy. The platform is designed around human
        review, explainable outputs, and explicit boundaries.

        Governance principles demonstrated in the prototype include:

        - **Human review:** model outputs prioritize analyst attention but do not automate decisions.
        - **Explainability:** baseline, observed value, deviation, anomaly score, risk score, and risk driver are visible.
        - **Threshold calibration:** score thresholds are configurable and should be validated before operational use.
        - **Model versioning:** model choices and parameters should be tracked as the system evolves.
        - **Drift monitoring:** changes in reporting behavior or syndrome distribution should be monitored over time.
        - **Auditability:** retrieved context, source files, and generated outputs should be reviewable.
        - **Privacy boundary:** the prototype uses public aggregate and synthetic data only.
        - **No autonomous response:** outputs do not provide clinical diagnosis, medical advice, or official response instructions.
        """
    )

    st.markdown("## 10. Limitations")
    st.markdown(
        """
        This application is a research prototype. Important limitations include:

        - Public aggregate data can be delayed, transformed, limited in syndrome breadth, and unsuitable for fine-grained local inference.
        - Synthetic data improves testing breadth but does not prove real-world performance.
        - Anomaly detection can flag reporting artifacts, noise, or expected seasonal changes.
        - Forecasting performance varies by syndrome, geography, history length, and volatility.
        - Risk scores are prioritization aids and require calibration before operational use.
        - RAG responses depend on the quality of retrieved documentation and should be reviewed by humans.
        - The platform is not connected to live operational feeds or official response workflows.
        """
    )

    st.markdown("## 11. Repository Evidence Map")
    repository_evidence = pd.DataFrame({
        "File": [
            "src/data_ingestion.py",
            "src/data_cleaning.py",
            "src/feature_engineering.py",
            "src/anomaly_detection.py",
            "src/risk_scoring.py",
            "src/forecasting.py",
            "src/geospatial.py",
            "src/rag_assistant.py",
            "src/synthetic_data.py",
            "src/hl7_generator.py",
            "app.py",
        ],
        "Evidence Provided": [
            "Demonstrates public/cached/synthetic data loading and reproducible ingestion boundaries.",
            "Demonstrates transformation of raw surveillance-style records into standardized analytic structure.",
            "Demonstrates creation of temporal features used by anomaly detection and risk scoring.",
            "Demonstrates transparent statistical detection and unsupervised anomaly detection.",
            "Demonstrates interpretable risk prioritization logic and level assignment.",
            "Demonstrates short-term forecasting with backtest-based model selection and fallback handling.",
            "Demonstrates state-level geographic aggregation and syndrome risk-driver identification.",
            "Demonstrates grounded retrieval and response generation for analyst-facing questions.",
            "Demonstrates synthetic syndrome expansion for controlled validation and broader use-case testing.",
            "Demonstrates interoperability-oriented synthetic message generation concepts.",
            "Demonstrates the analyst-facing Streamlit application and workflow design.",
        ],
    })
    st.dataframe(repository_evidence, use_container_width=True, hide_index=True)

    st.markdown("## 12. Conclusion")
    st.markdown(
        """
        The Public Syndromic Surveillance Platform demonstrates a practical architecture for modern surveillance-style
        analytics: public and synthetic data boundaries, transparent feature engineering, layered anomaly detection,
        interpretable risk scoring, geospatial prioritization, short-term forecasting, data quality review, and grounded
        natural-language assistance.

        The central contribution is not a claim that any single model can identify public health events on its own. The
        contribution is an explainable workflow that helps reviewers move from raw surveillance-style data to structured
        signal intelligence while preserving human judgment, methodological transparency, and responsible-use boundaries.
        """
    )


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
