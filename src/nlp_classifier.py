from __future__ import annotations


SYNDROME_KEYWORDS = {
    "COVID-like illness": {
        "covid",
        "coronavirus",
        "loss of taste",
        "loss of smell",
        "shortness of breath",
        "sob",
        "cough",
        "fever",
    },
    "Influenza-like illness": {
        "flu",
        "influenza",
        "chills",
        "body aches",
        "muscle aches",
        "headache",
        "fatigue",
        "fever",
    },
    "RSV-like illness": {
        "rsv",
        "wheezing",
        "bronchiolitis",
        "infant cough",
        "difficulty breathing",
        "nasal congestion",
        "cough",
    },
}


def classify_chief_complaint(text: str) -> dict:
    #     # 1. Normalize the chief complaint text.
    # 2. Count matched keyword phrases for each syndrome category.
    # 3. Select the syndrome with the most matches and compute a simple confidence score.
    # 4. Return scores and matched terms so the classification remains explainable.
    normalized = text.lower().strip()
    scores = {}
    matched_terms = {}

    for syndrome, keywords in SYNDROME_KEYWORDS.items():
        matches = sorted(term for term in keywords if term in normalized)
        scores[syndrome] = len(matches)
        matched_terms[syndrome] = matches

    best_syndrome = max(scores, key=scores.get)
    total_score = sum(scores.values())
    confidence = scores[best_syndrome] / total_score if total_score else 0.0

    if scores[best_syndrome] == 0:
        best_syndrome = "Unclassified"

    return {
        "chief_complaint": text,
        "predicted_syndrome": best_syndrome,
        "confidence": round(confidence, 2),
        "scores": scores,
        "matched_terms": matched_terms,
    }
