from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def _chunk_text(text: str, chunk_size: int = 900, overlap: int = 120) -> list[str]:
    #     # 1. Walk through long documentation text in overlapping windows.
    # 2. Preserve overlap so important context is not lost between chunks.
    chunks = []
    start = 0
    while start < len(text):
        chunk = text[start : start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def load_document_chunks(paths: list[Path]) -> pd.DataFrame:
    # Read each local documentation file and convert it into searchable text chunks.
    rows = []
    for path in paths:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        for idx, chunk in enumerate(_chunk_text(text), start=1):
            rows.append({"source": path.name, "chunk_id": idx, "text": chunk})
    return pd.DataFrame(rows)


def build_surveillance_snapshot(df: pd.DataFrame) -> str:
    #     # 1. Summarize the currently loaded surveillance dataset.
    # 2. Include latest risk rows, anomaly counts, and data quality facts.
    # 3. Return text that can be indexed alongside static documentation.
    if df.empty:
        return "No surveillance records are currently loaded."

    latest_date = pd.to_datetime(df["date"]).max()
    latest = df[df["date"] == latest_date]
    high_risk = latest[latest["risk_level"].isin(["High", "Critical"])]
    anomalies = latest[latest["any_anomaly"]]
    top_risk = (
        latest.sort_values("risk_score", ascending=False)
        [["state", "syndrome", "visit_percentage", "risk_score", "risk_level", "any_anomaly"]]
        .head(10)
    )

    quality = {
        "records": len(df),
        "states_regions": df["state"].nunique(),
        "syndromes": df["syndrome"].nunique(),
        "start_date": str(pd.to_datetime(df["date"]).min().date()),
        "end_date": str(latest_date.date()),
        "missing_visit_percentage": int(df["visit_percentage"].isna().sum()),
        "duplicate_state_syndrome_date": int(df.duplicated(subset=["date", "state", "syndrome"]).sum()),
    }

    return "\n".join([
        "Live surveillance snapshot:",
        f"Latest date: {latest_date.date()}",
        f"Total records: {len(df)}",
        f"States/regions: {df['state'].nunique()}",
        f"Syndromes: {', '.join(sorted(df['syndrome'].dropna().unique()))}",
        f"Latest anomaly alerts: {len(anomalies)}",
        f"Latest high/critical risks: {len(high_risk)}",
        f"Data quality: {quality}",
        "Top latest risk rows:",
        top_risk.to_string(index=False),
    ])


def _append_extra_context(chunks: pd.DataFrame, extra_contexts: list[dict] | None) -> pd.DataFrame:
    # Merge generated runtime context, such as live surveillance snapshot, into doc chunks.
    if not extra_contexts:
        return chunks
    extra = pd.DataFrame(extra_contexts)
    if chunks.empty:
        return extra
    return pd.concat([chunks, extra], ignore_index=True)


def search_documents(
    query: str,
    paths: list[Path],
    top_k: int = 3,
    extra_contexts: list[dict] | None = None,
) -> pd.DataFrame:
    #     # 1. Load static documentation and append optional live context.
    # 2. Fit a TF-IDF vectorizer over chunks plus the user query.
    # 3. Rank chunks by cosine similarity to the query.
    # 4. Return the top retrieved sources for answer grounding.
    chunks = load_document_chunks(paths)
    chunks = _append_extra_context(chunks, extra_contexts)
    if chunks.empty or not query.strip():
        return pd.DataFrame(columns=["source", "chunk_id", "score", "text"])

    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(chunks["text"].tolist() + [query])
    scores = cosine_similarity(matrix[-1], matrix[:-1]).flatten()
    results = chunks.copy()
    results["score"] = scores
    return results.sort_values("score", ascending=False).head(top_k)[["source", "chunk_id", "score", "text"]]


def answer_from_documents(query: str, paths: list[Path], top_k: int = 3) -> tuple[str, pd.DataFrame]:
    # Provide a retrieval-only answer from local docs without calling an LLM.
    results = search_documents(query, paths, top_k=top_k)
    if results.empty or results["score"].max() == 0:
        return "No strong local documentation match was found for that question.", results

    answer_parts = []
    for _, row in results.iterrows():
        answer_parts.append(f"From {row['source']} section {row['chunk_id']}: {row['text']}")
    return "\n\n".join(answer_parts), results


def _retrieval_only_answer(query: str, results: pd.DataFrame) -> str:
    # Format retrieved context into a readable fallback answer when LLM is unavailable.
    if results.empty or results["score"].max() == 0:
        return (
            "I could not find a strong match in the local surveillance data or project documentation.\n\n"
            "Try asking about current signals, highest-risk regions, syndrome categories, data quality, forecasting, "
            "risk scoring, or how the platform should be used for analyst review."
        )

    lines = [
        "Here is a plain-language answer based on retrieved local context.",
        "",
        f"Question: {query}",
        "",
        "What the retrieved context says:",
    ]
    for _, row in results.head(3).iterrows():
        snippet = " ".join(str(row["text"]).split())[:650]
        lines.append(f"- {snippet}")
    lines.extend([
        "",
        "How to interpret this: use the answer as context for analyst review. It is not a diagnosis, a confirmed event, official public health guidance, or an operational response instruction.",
    ])
    return "\n".join(lines)


def _generate_llm_answer(query: str, results: pd.DataFrame, model: str, response_mode: str = "query") -> str:
    #     # 1. Require an OpenAI API key and client library.
    # 2. Build a prompt from retrieved local context only.
    # 3. Ask the LLM to answer with operational caveats and no medical directives.
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("The `openai` package is not installed. Run `pip install -r requirements.txt`.") from exc

    context = "\n\n".join(
        f"[{idx + 1}] Source: {row['source']} section {row['chunk_id']}\n{row['text']}"
        for idx, (_, row) in enumerate(results.iterrows())
    )
    mode_instruction = (
        "For the Public Health Query Assistant, focus on explaining surveillance data, charts, methods, syndrome categories, and data quality in plain language."
        if response_mode == "query"
        else "For the Policy Briefing Assistant, focus on leadership-ready situational awareness, review priorities, planning considerations, governance needs, and responsible-use boundaries."
    )
    instructions = (
        "You are an applied public health informatics assistant for a syndromic surveillance analytics prototype. "
        "Use only the retrieved context and live surveillance snapshot. "
        "Write for a mixed audience of analysts, program leaders, and non-technical reviewers. "
        "Use clear headings, short paragraphs, and concrete bullets. Avoid jargon unless you explain it. "
        f"{mode_instruction} "
        "Frame outputs as support for analyst review and planning. Do not claim that a signal confirms an outbreak, diagnosis, disease event, or official response need. "
        "When the context is insufficient, say what is missing and what data would be needed. "
        "Do not provide medical advice, diagnosis, treatment guidance, official public health directives, or autonomous response instructions."
    )
    prompt = (
        f"Question:\n{query}\n\n"
        f"Retrieved context:\n{context}\n\n"
        "Answer using this structure:\n"
        "1. Short answer\n"
        "2. What the platform data or documentation supports\n"
        "3. What an analyst or reviewer should check next\n"
        "4. Caveats and responsible-use boundary"
    )

    client = OpenAI(api_key=api_key)
    response = client.responses.create(
        model=model,
        instructions=instructions,
        input=prompt,
    )
    return response.output_text


def answer_rag_question(
    query: str,
    paths: list[Path],
    surveillance_df: pd.DataFrame | None = None,
    guidance_context: str | None = None,
    top_k: int = 5,
    use_llm: bool = True,
    model: str | None = None,
    response_mode: str = "query",
) -> tuple[str, pd.DataFrame, dict]:
    #     # 1. Add live surveillance snapshot context when dashboard data is available.
    # 2. Retrieve the most relevant local context for the user query.
    # 3. Use LLM generation only when enabled and retrieval found evidence.
    # 4. Fall back to retrieval-only output and return metadata about the mode used.
    extra_contexts = []
    if surveillance_df is not None:
        extra_contexts.append({
            "source": "live_surveillance_snapshot",
            "chunk_id": 1,
            "text": build_surveillance_snapshot(surveillance_df),
        })
    if guidance_context:
        extra_contexts.append({
            "source": "cdc_hhs_guidance_summary",
            "chunk_id": 1,
            "text": guidance_context,
        })

    results = search_documents(query, paths, top_k=top_k, extra_contexts=extra_contexts)
    selected_model = model or os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    metadata = {
        "mode": "retrieval_only",
        "model": None,
        "llm_error": None,
    }

    if use_llm and not results.empty and results["score"].max() > 0:
        try:
            answer = _generate_llm_answer(query, results, selected_model, response_mode=response_mode)
            metadata["mode"] = "llm_rag"
            metadata["model"] = selected_model
            return answer, results, metadata
        except Exception as exc:
            metadata["llm_error"] = str(exc)

    return _retrieval_only_answer(query, results), results, metadata
