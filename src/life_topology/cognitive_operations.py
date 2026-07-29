from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import chi2_contingency
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_mutual_info_score,
    confusion_matrix,
    mutual_info_score,
    normalized_mutual_info_score,
)
from sklearn.preprocessing import normalize

from .config import OUTPUT_ROOT, PARQUET_ROOT, SETTINGS
from .forecasting import _load_daily
from .topology import _jsonable


START = "2025-03-29"
END = "2026-07-29"
OUTPUT_DIR = OUTPUT_ROOT / "cognitive_operations"
OPERATIONS = ["Observe", "Experience", "Evaluate", "Model", "Execute"]

# These anchors define cognitive operations only. They contain no state names,
# state IDs, cluster profiles, dates, or daily-state feature terminology.
ANCHORS = {
    "Observe": [
        "Noticing and describing what is present without interpreting it.",
        "Reporting facts, details, events, messages, surroundings, or behavior.",
        "Gathering information and recording what happened.",
        "Looking closely, investigating, searching, reading, or inspecting evidence.",
        "The primary mental act is observation and information intake.",
    ],
    "Experience": [
        "Living through and expressing a feeling, sensation, relationship, or event.",
        "Describing an immediate emotional, bodily, musical, social, or personal experience.",
        "Being immersed in what is happening rather than analyzing or changing it.",
        "Sharing ordinary life, conversation, humor, connection, distress, or pleasure.",
        "The primary mental act is undergoing and expressing experience.",
    ],
    "Evaluate": [
        "Judging quality, fit, safety, meaning, value, correctness, or preference.",
        "Comparing alternatives and deciding what is good, bad, useful, wrong, or wanted.",
        "Critiquing a system, person, product, claim, job, place, or interpretation.",
        "Assessing evidence against a standard and reaching a verdict.",
        "The primary mental act is evaluation and discrimination.",
    ],
    "Model": [
        "Building an explanation, abstraction, theory, map, taxonomy, or causal structure.",
        "Connecting multiple observations into a coherent system or representation.",
        "Asking why something works and identifying dependencies, patterns, mechanisms, or equations.",
        "Reflecting across time to construct a model that explains many events.",
        "The primary mental act is synthesis and structural modeling.",
    ],
    "Execute": [
        "Taking action to build, change, ship, apply, configure, fix, organize, or complete something.",
        "Giving implementation instructions and moving a concrete project or plan forward.",
        "Writing, testing, deploying, editing, applying, scheduling, purchasing, or traveling.",
        "Turning an intention into an operational step or finished artifact.",
        "The primary mental act is execution and intervention.",
    ],
}


def _daily_documents(days: list[str]) -> tuple[list[str], list[int]]:
    episodes = pd.read_parquet(PARQUET_ROOT / "episodes.parquet")
    episodes = episodes[
        episodes["behavioral_day"].isin(days)
        & (episodes["origin_class"] != "ai_response")
        & episodes["text"].notna()
    ].copy()

    # Preserve broad daily evidence while keeping BGE within its context window.
    # Take content from across the day rather than only its beginning.
    documents: list[str] = []
    evidence_counts: list[int] = []
    for day in days:
        rows = episodes[episodes["behavioral_day"] == day].sort_values("started_at")
        chunks = []
        for text in rows["text"].astype(str):
            text = " ".join(text.split())
            if not text:
                continue
            chunks.append(text[:1800])
        evidence_counts.append(len(chunks))
        if not chunks:
            documents.append("[No user-authored textual evidence recorded.]")
            continue
        joined = "\n".join(chunks)
        if len(joined) > 10000:
            positions = np.linspace(0, len(joined) - 1800, 5, dtype=int)
            joined = "\n".join(joined[p : p + 1800] for p in positions)
        documents.append(joined)
    return documents, evidence_counts


def _classify(
    model_name: str, documents: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model = SentenceTransformer(model_name)
    model.max_seq_length = 1024 if "bge-m3" in model_name.lower() else 512
    anchor_texts = [text for op in OPERATIONS for text in ANCHORS[op]]
    anchor_vectors = normalize(
        model.encode(anchor_texts, batch_size=32, show_progress_bar=True)
    )
    day_vectors = normalize(
        model.encode(
            documents,
            batch_size=2 if "bge-m3" in model_name.lower() else 16,
            show_progress_bar=True,
        )
    )
    anchor_vectors = anchor_vectors.reshape(len(OPERATIONS), -1, anchor_vectors.shape[1])
    scores = np.column_stack(
        [day_vectors @ anchor_vectors[i].T for i in range(len(OPERATIONS))]
    ).reshape(len(documents), len(OPERATIONS), -1).mean(axis=2)
    labels = scores.argmax(axis=1)
    sorted_scores = np.sort(scores, axis=1)
    margins = sorted_scores[:, -1] - sorted_scores[:, -2]
    return labels, scores, margins


def _metrics(states: np.ndarray, operations: np.ndarray, rng: np.random.Generator) -> dict:
    matrix = confusion_matrix(states, operations, labels=range(6))
    # confusion_matrix assumes a common label set, so build the rectangular
    # six-by-five table directly.
    matrix = np.zeros((6, 5), dtype=int)
    for state, operation in zip(states, operations):
        matrix[int(state), int(operation)] += 1

    row_rate = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    global_rate = matrix.sum(axis=0) / matrix.sum()
    enrichment = row_rate / np.maximum(global_rate, 1e-12)
    chi2, p_chi, dof, expected = chi2_contingency(matrix)
    observed_mi = mutual_info_score(states, operations)
    null_mi = np.empty(10000)
    null_max_enrichment = np.empty(10000)
    for i in range(len(null_mi)):
        shuffled = rng.permutation(operations)
        null_mi[i] = mutual_info_score(states, shuffled)
        shuffled_matrix = np.zeros((6, 5), dtype=int)
        for state, operation in zip(states, shuffled):
            shuffled_matrix[int(state), int(operation)] += 1
        shuffled_rate = shuffled_matrix / np.maximum(
            shuffled_matrix.sum(axis=1, keepdims=True), 1
        )
        null_max_enrichment[i] = np.max(
            shuffled_rate / np.maximum(global_rate, 1e-12)
        )

    row_ind, col_ind = linear_sum_assignment(-matrix)
    matched_accuracy = matrix[row_ind, col_ind].sum() / matrix.sum()
    dominant = []
    for state in range(6):
        op = int(matrix[state].argmax())
        dominant.append(
            {
                "state": state,
                "operation": OPERATIONS[op],
                "count": int(matrix[state, op]),
                "within_state_rate": float(row_rate[state, op]),
                "enrichment": float(enrichment[state, op]),
            }
        )
    return {
        "matrix": matrix,
        "row_rates": row_rate,
        "global_operation_rates": global_rate,
        "enrichment": enrichment,
        "chi_square": float(chi2),
        "chi_square_p": float(p_chi),
        "degrees_of_freedom": int(dof),
        "expected": expected,
        "mutual_information_nats": float(observed_mi),
        "normalized_mutual_information": float(
            normalized_mutual_info_score(states, operations)
        ),
        "adjusted_mutual_information": float(
            adjusted_mutual_info_score(states, operations)
        ),
        "permutation_p_mi": float((1 + (null_mi >= observed_mi).sum()) / 10001),
        "mi_null_mean": float(null_mi.mean()),
        "mi_null_99": float(np.quantile(null_mi, 0.99)),
        "max_enrichment_null_99": float(np.quantile(null_max_enrichment, 0.99)),
        "best_one_to_one_matched_accuracy": float(matched_accuracy),
        "dominant_operation_by_state": dominant,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    base, state_vectors = _load_daily()
    base = base[
        (base["date"] >= pd.Timestamp(START))
        & (base["date"] <= pd.Timestamp(END))
    ].reset_index(drop=True)
    state_vectors = state_vectors[: len(base)]

    states = KMeans(
        n_clusters=6, random_state=SETTINGS.random_seed, n_init=80
    ).fit_predict(state_vectors)
    days = base["behavioral_day"].astype(str).tolist()
    documents, evidence_counts = _daily_documents(days)

    models = {
        "bge_m3": "BAAI/bge-m3",
        "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    }
    classifications = {}
    result = {
        "metadata": {
            "start": START,
            "end": END,
            "days": len(days),
            "state_learning": "KMeans k=6 on existing blind daily semantic+structural vectors",
            "operation_learning": "zero-shot semantic anchors from daily text only",
            "state_information_exposed_to_operation_classifier": False,
            "operations": OPERATIONS,
            "models": models,
        },
        "models": {},
    }
    rng = np.random.default_rng(SETTINGS.random_seed)
    rows = pd.DataFrame(
        {
            "behavioral_day": days,
            "state": states,
            "evidence_count": evidence_counts,
        }
    )

    for key, model_name in models.items():
        labels, scores, margins = _classify(model_name, documents)
        classifications[key] = labels
        rows[f"{key}_operation"] = [OPERATIONS[i] for i in labels]
        rows[f"{key}_margin"] = margins
        for i, operation in enumerate(OPERATIONS):
            rows[f"{key}_{operation.lower()}_score"] = scores[:, i]
        result["models"][key] = _metrics(states, labels, rng)

    agreement = classifications["bge_m3"] == classifications["minilm"]
    result["cross_model"] = {
        "agreement_rate": float(agreement.mean()),
        "agreement_count": int(agreement.sum()),
        "agreement_only": _metrics(
            states[agreement], classifications["bge_m3"][agreement], rng
        ),
    }
    rows.to_parquet(OUTPUT_DIR / "blind_daily_labels.parquet", index=False)
    (OUTPUT_DIR / "blind_cognitive_operations.json").write_text(
        json.dumps(_jsonable(result), indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(_jsonable(result), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
