from __future__ import annotations

from datetime import datetime

import pandas as pd


CHIEF_COMPLAINTS = {
    "COVID-like illness": [
        "fever cough shortness of breath",
        "sore throat fever body aches",
        "loss of taste cough congestion",
    ],
    "Influenza-like illness": [
        "fever chills muscle aches",
        "headache cough fatigue",
        "sudden fever sore throat",
    ],
    "RSV-like illness": [
        "wheezing cough difficulty breathing",
        "infant cough nasal congestion",
        "bronchiolitis symptoms and wheezing",
    ],
}


def generate_hl7_message(
    patient_id: str,
    state: str,
    syndrome: str,
    visit_date: datetime | pd.Timestamp | None = None,
    chief_complaint: str | None = None,
) -> str:
    #     # 1. Choose a visit timestamp and chief complaint.
    # 2. Build a synthetic visit identifier from date and patient id.
    # 3. Assemble minimal ADT-style HL7 segments for demonstration only.
    visit_date = pd.Timestamp(visit_date or datetime.utcnow())
    complaint = chief_complaint or CHIEF_COMPLAINTS.get(syndrome, ["fever cough"])[0]
    timestamp = visit_date.strftime("%Y%m%d%H%M%S")
    visit_id = f"V{visit_date.strftime('%Y%m%d')}{patient_id[-4:]}"

    segments = [
        f"MSH|^~\\&|SYNTH_ED|{state}|PUBLIC_HEALTH|STATE_SURV|{timestamp}||ADT^A04|{visit_id}|P|2.5.1",
        f"PID|1||{patient_id}^^^SYNTH||DOE^TEST^^^^^L||19800101|U|||^^^^{state}",
        f"PV1|1|E|ED^{state}^^SYNTH||||||||||||||||{visit_id}",
        f"OBX|1|TX|8661-1^Chief complaint^LN||{complaint}",
        f"OBX|2|TX|75325-1^Syndromic surveillance category^LN||{syndrome}",
    ]
    return "\r".join(segments)


def generate_hl7_batch(df: pd.DataFrame, state: str, syndrome: str, count: int = 5) -> pd.DataFrame:
    #     # 1. Select recent surveillance rows for the requested state/syndrome.
    # 2. Cycle through syndrome-specific synthetic chief complaints.
    # 3. Return a dataframe of synthetic patient ids, metadata, and HL7 messages.
    subset = df[(df["state"] == state) & (df["syndrome"] == syndrome)].sort_values("date", ascending=False)
    if subset.empty:
        return pd.DataFrame(columns=["patient_id", "date", "state", "syndrome", "chief_complaint", "hl7_message"])

    complaints = CHIEF_COMPLAINTS.get(syndrome, ["fever cough"])
    rows = []
    for idx, (_, row) in enumerate(subset.head(count).iterrows(), start=1):
        patient_id = f"SYN{idx:06d}"
        complaint = complaints[(idx - 1) % len(complaints)]
        rows.append({
            "patient_id": patient_id,
            "date": row["date"],
            "state": state,
            "syndrome": syndrome,
            "chief_complaint": complaint,
            "hl7_message": generate_hl7_message(patient_id, state, syndrome, row["date"], complaint),
        })
    return pd.DataFrame(rows)
