from pathlib import Path

import pandas as pd

from src.geospatial import latest_geospatial_risk
from src.hl7_generator import generate_hl7_message
from src.nlp_classifier import classify_chief_complaint
from src.rag_assistant import answer_from_documents, answer_rag_question


def test_generate_hl7_message_contains_core_segments():
    # A generated synthetic HL7 message should contain core ADT and syndrome OBX segments.
    message = generate_hl7_message("SYN000001", "Georgia", "COVID-like illness", pd.Timestamp("2026-01-01"))

    assert "MSH|^~\\&" in message
    assert "PID|1||SYN000001" in message
    assert "OBX|2|TX|75325-1^Syndromic surveillance category^LN||COVID-like illness" in message


def test_chief_complaint_classifier_detects_rsv():
    # RSV-related keywords should map to the RSV-like illness category.
    result = classify_chief_complaint("infant cough with wheezing and difficulty breathing")

    assert result["predicted_syndrome"] == "RSV-like illness"
    assert result["confidence"] > 0


def test_rag_assistant_returns_local_source(tmp_path: Path):
    # Retrieval-only helper should cite the local document chunk that matches the query.
    doc = tmp_path / "model_summary.md"
    doc.write_text("Risk scoring combines z-score severity and anomaly persistence.", encoding="utf-8")

    answer, sources = answer_from_documents("What does risk scoring combine?", [doc])

    assert "Risk scoring" in answer
    assert sources.iloc[0]["source"] == "model_summary.md"


def test_llm_rag_falls_back_without_api_key(tmp_path: Path, monkeypatch):
    # When OPENAI_API_KEY is absent, RAG should return retrieved context instead of failing.
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    doc = tmp_path / "model_summary.md"
    doc.write_text("Risk scoring combines z-score severity and anomaly persistence.", encoding="utf-8")
    df = pd.DataFrame({
        "date": pd.to_datetime(["2026-01-01"]),
        "state": ["Georgia"],
        "syndrome": ["COVID-like illness"],
        "visit_percentage": [2.5],
        "risk_score": [72.0],
        "risk_level": ["High"],
        "any_anomaly": [True],
    })

    answer, sources, metadata = answer_rag_question(
        "Which states have high risk and how does risk scoring work?",
        [doc],
        surveillance_df=df,
        use_llm=True,
    )

    assert metadata["mode"] == "retrieval_only"
    assert "OPENAI_API_KEY is not set" in metadata["llm_error"]
    assert "live_surveillance_snapshot" in set(sources["source"])
    assert "retrieved local context" in answer


def test_latest_geospatial_risk_adds_coordinates():
    # Latest geospatial aggregation should attach state centroid coordinates and max risk.
    df = pd.DataFrame({
        "date": pd.to_datetime(["2026-01-01", "2026-01-02"]),
        "state": ["Georgia", "Georgia"],
        "syndrome": ["COVID-like illness", "Influenza-like illness"],
        "risk_score": [20.0, 80.0],
        "risk_level": ["Low", "High"],
        "any_anomaly": [False, True],
    })

    geo = latest_geospatial_risk(df)

    assert geo.iloc[0]["state"] == "Georgia"
    assert geo.iloc[0]["max_risk_score"] == 80.0
    assert pd.notna(geo.iloc[0]["lat"])
    assert pd.notna(geo.iloc[0]["lon"])
