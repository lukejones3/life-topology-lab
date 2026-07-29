from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.preprocessing import RobustScaler, StandardScaler

from .config import OUTPUT_ROOT, PARQUET_ROOT, SETTINGS
from .topology import _jsonable, _state_data


START = "2025-03-29"
END = "2026-07-29"
OUTPUT = OUTPUT_ROOT / "forecast" / "forecast_16m.json"
REPORT = OUTPUT_ROOT / "forecast" / "FORECAST.md"
EPS = 1e-8

# Every column below exists on or before the day being forecast. In particular,
# residual impact, later references, forward recurrence, and future alignment
# are excluded because they contain hindsight.
COUNT_COLUMNS = [
    "original_reflection_count", "question_count", "technical_count",
    "social_count", "distinct_people", "distinct_threads", "active_minutes",
    "late_night_count", "chars_original", "reflection_references",
    "retrospective_count", "prospective_count",
]
SAFE_DERIVED_COLUMNS = [
    "activity_z", "novelty", "lived_shift", "interpretive_shift",
    "prospective_shift", "cross_domain_spread", "activation_intensity",
    "lived_novelty_30", "interpretive_novelty_30", "prospective_novelty_30",
]
PROFILE_COLUMNS = [
    "technical_count", "prospective_count", "original_reflection_count",
    "retrospective_count", "social_count", "distinct_people",
    "activation_intensity", "cross_domain_spread",
]


def _softmax(values: np.ndarray) -> np.ndarray:
    values = values - np.max(values)
    out = np.exp(values)
    return out / out.sum()


def _load_daily() -> tuple[pd.DataFrame, np.ndarray]:
    base, _, _ = _state_data(start_date=START, include_sensors=False)
    base = base[base["date"] <= pd.Timestamp(END)].sort_values("date").reset_index(drop=True)

    vectors = pd.read_parquet(PARQUET_ROOT / "daily_vector_states.parquet")
    lived = vectors[vectors["view_type"] == "lived"].set_index("behavioral_day")
    dimension = int(lived["dimensions"].max())
    semantic = np.full((len(base), dimension), np.nan, dtype=np.float32)
    for i, day in enumerate(base["behavioral_day"]):
        if day not in lived.index:
            continue
        value = lived.loc[day, "vector"]
        if isinstance(value, pd.Series):
            value = value.iloc[0]
        if value is not None:
            semantic[i] = np.asarray(value, dtype=np.float32)
    observed = np.isfinite(semantic).all(axis=1)
    semantic_reduced = np.zeros((len(base), 16), dtype=float)
    semantic_reduced[observed] = PCA(
        n_components=16, random_state=SETTINGS.random_seed
    ).fit_transform(semantic[observed])

    structural_raw = base[COUNT_COLUMNS].fillna(0).astype(float).to_numpy()
    structural_counts = RobustScaler().fit_transform(np.log1p(np.maximum(structural_raw, 0)))
    derived = base[SAFE_DERIVED_COLUMNS].replace([np.inf, -np.inf], np.nan).fillna(0)
    derived_scaled = RobustScaler().fit_transform(derived)
    state = StandardScaler().fit_transform(
        np.column_stack([semantic_reduced, structural_counts, derived_scaled])
    )
    return base, state


def _weekly(base: pd.DataFrame, state: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    frame = base.copy()
    # Fixed seven-day bins anchored on the forecast window start; the final bin
    # is retained and explicitly marked partial.
    frame["week_index"] = ((frame["date"] - frame["date"].min()).dt.days // 7).astype(int)
    records: list[dict] = []
    vectors: list[np.ndarray] = []
    for week_index, group in frame.groupby("week_index", sort=True):
        idx = group.index.to_numpy()
        records.append({
            "week_index": int(week_index),
            "start": group["date"].min().date().isoformat(),
            "end": group["date"].max().date().isoformat(),
            "days": int(len(group)),
            **{column: float(group[column].fillna(0).mean()) for column in PROFILE_COLUMNS},
        })
        vectors.append(state[idx].mean(axis=0))
    return pd.DataFrame(records), StandardScaler().fit_transform(np.vstack(vectors))


def _state_names(weekly: pd.DataFrame, labels: np.ndarray) -> tuple[list[dict], dict[int, str]]:
    profile = StandardScaler().fit_transform(weekly[PROFILE_COLUMNS].fillna(0))
    profiles = []
    names: dict[int, str] = {}
    used: Counter[str] = Counter()
    vocabulary = {
        "technical_count": "Technical construction",
        "prospective_count": "Prospective mobilization",
        "original_reflection_count": "Integrative reflection",
        "retrospective_count": "Autobiographical retrieval",
        "social_count": "Social engagement",
        "distinct_people": "Relational breadth",
        "activation_intensity": "High activation",
        "cross_domain_spread": "Cross-domain integration",
    }
    for state_id in range(6):
        idx = labels == state_id
        mean = profile[idx].mean(axis=0)
        order = np.argsort(mean)[::-1]
        primary = vocabulary[PROFILE_COLUMNS[order[0]]]
        used[primary] += 1
        secondary = vocabulary[PROFILE_COLUMNS[order[1]]].lower()
        name = primary if used[primary] == 1 else f"{primary} + {secondary}"
        names[state_id] = name
        profiles.append({
            "id": state_id,
            "name": name,
            "weeks": int(idx.sum()),
            "profile": {key: float(value) for key, value in zip(PROFILE_COLUMNS, mean)},
            "topSignals": [
                {"signal": PROFILE_COLUMNS[i], "z": float(mean[i])}
                for i in order[:3]
            ],
        })
    return profiles, names


def _transition_probability(labels: np.ndarray, classes: np.ndarray, alpha: float = 0.75) -> np.ndarray:
    matrix = np.full((len(classes), len(classes)), alpha, dtype=float)
    lookup = {value: i for i, value in enumerate(classes)}
    for a, b in zip(labels[:-1], labels[1:]):
        matrix[lookup[a], lookup[b]] += 1
    return matrix / matrix.sum(axis=1, keepdims=True)


def _duration_probability(labels: np.ndarray, classes: np.ndarray, current_run: int) -> np.ndarray:
    current = labels[-1]
    runs: list[tuple[int, int, int | None]] = []
    start = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            runs.append((int(labels[start]), i - start, int(labels[i]) if i < len(labels) else None))
            start = i
    completed = [r for r in runs if r[0] == current and r[2] is not None]
    stay = (sum(length >= current_run + 1 for _, length, _ in completed) + 1) / (len(completed) + 2)
    destination = np.ones(len(classes), dtype=float) * 0.5
    lookup = {value: i for i, value in enumerate(classes)}
    for _, _, nxt in completed:
        destination[lookup[nxt]] += 1
    destination[lookup[current]] = 0
    if destination.sum() == 0:
        destination += 1
        destination[lookup[current]] = 0
    destination /= destination.sum()
    result = destination * (1 - stay)
    result[lookup[current]] = stay
    return result


def _analog_probability(vectors: np.ndarray, labels: np.ndarray, classes: np.ndarray, history_end: int) -> np.ndarray:
    if history_end < 5:
        return np.ones(len(classes)) / len(classes)
    query = vectors[history_end]
    candidates = np.arange(1, history_end)
    distances = np.linalg.norm(vectors[candidates] - query, axis=1)
    bandwidth = max(float(np.quantile(distances, 0.3)), 0.25)
    weights = np.exp(-(distances ** 2) / (2 * bandwidth ** 2))
    probability = np.ones(len(classes), dtype=float) * 0.35
    lookup = {value: i for i, value in enumerate(classes)}
    for candidate, weight in zip(candidates, weights):
        probability[lookup[labels[candidate + 1]]] += weight
    return probability / probability.sum()


def _feature_matrix(vectors: np.ndarray, labels: np.ndarray, classes: np.ndarray) -> np.ndarray:
    reduced = PCA(
        n_components=min(8, vectors.shape[1], len(vectors) - 1),
        random_state=SETTINGS.random_seed,
    ).fit_transform(vectors)
    rows = []
    for i in range(len(vectors)):
        lag1 = reduced[i]
        lag2 = reduced[max(0, i - 1)]
        lag4 = reduced[max(0, i - 3): i + 1].mean(axis=0)
        slope = lag1 - lag2
        run = 1
        while i - run >= 0 and labels[i - run] == labels[i]:
            run += 1
        one_hot = np.array([float(labels[i] == c) for c in classes])
        rows.append(np.concatenate([lag1, lag4, slope, one_hot, [run]]))
    return StandardScaler().fit_transform(np.vstack(rows))


def _predict_components(
    labels: np.ndarray,
    vectors: np.ndarray,
    features: np.ndarray,
    at: int,
    classes: np.ndarray,
) -> dict[str, np.ndarray]:
    train_labels = labels[: at + 1]
    run = 1
    while at - run >= 0 and labels[at - run] == labels[at]:
        run += 1
    markov = _transition_probability(train_labels, classes)[np.where(classes == labels[at])[0][0]]
    duration = _duration_probability(train_labels, classes, run)
    analog = _analog_probability(vectors, labels, classes, at)
    logistic = np.ones(len(classes)) / len(classes)
    x_train, y_train = features[:at], labels[1: at + 1]
    if len(np.unique(y_train)) >= 2:
        model = LogisticRegression(
            C=0.35, max_iter=3000, class_weight="balanced",
            random_state=SETTINGS.random_seed,
        ).fit(x_train, y_train)
        raw = model.predict_proba(features[[at]])[0]
        logistic = np.ones(len(classes)) * EPS
        for cls, value in zip(model.classes_, raw):
            logistic[np.where(classes == cls)[0][0]] = value
        logistic /= logistic.sum()
    return {"markov": markov, "duration": duration, "analog": analog, "logistic": logistic}


def _walk_forward(labels: np.ndarray, vectors: np.ndarray) -> tuple[dict, np.ndarray]:
    classes = np.arange(6)
    features = _feature_matrix(vectors, labels, classes)
    start = 28
    rows = []
    component_losses: dict[str, list[float]] = {name: [] for name in ("markov", "duration", "analog", "logistic")}
    for at in range(start, len(labels) - 1):
        components = _predict_components(labels, vectors, features, at, classes)
        actual = int(labels[at + 1])
        for name, probability in components.items():
            component_losses[name].append(-math.log(max(probability[actual], EPS)))
        rows.append({"at": at, "actual": actual, "components": components})
    mean_loss = {name: float(np.mean(values)) for name, values in component_losses.items()}
    weights_raw = {name: math.exp(-loss) for name, loss in mean_loss.items()}
    total = sum(weights_raw.values())
    weights = {name: value / total for name, value in weights_raw.items()}

    predictions = []
    actuals = []
    for row in rows:
        probability = sum(weights[name] * row["components"][name] for name in weights)
        probability /= probability.sum()
        predictions.append(probability)
        actuals.append(row["actual"])
    pred = np.vstack(predictions)
    actual = np.asarray(actuals)
    one_hot = np.eye(6)[actual]
    top1 = float(np.mean(np.argmax(pred, axis=1) == actual))
    top2 = float(np.mean([a in np.argsort(p)[-2:] for a, p in zip(actual, pred)]))
    brier = float(np.mean(np.sum((pred - one_hot) ** 2, axis=1)))
    ensemble_logloss = float(log_loss(actual, pred, labels=classes))
    baseline_rows = []
    for row in rows:
        counts = np.bincount(labels[: row["at"] + 1], minlength=6).astype(float) + 1
        baseline_rows.append(counts / counts.sum())
    baseline_pred = np.vstack(baseline_rows)
    baseline_logloss = float(log_loss(actual, baseline_pred, labels=classes))
    current = _predict_components(labels, vectors, features, len(labels) - 1, classes)
    current_probability = sum(weights[name] * current[name] for name in weights)
    current_probability /= current_probability.sum()
    return {
        "folds": len(actual),
        "warmupWeeks": start,
        "weights": weights,
        "componentLogLoss": mean_loss,
        "ensemble": {
            "logLoss": ensemble_logloss,
            "brier": brier,
            "top1Accuracy": top1,
            "top2Accuracy": top2,
        },
        "frequencyBaseline": {"logLoss": baseline_logloss},
        "logLossImprovement": baseline_logloss - ensemble_logloss,
        "beatFrequencyBaseline": ensemble_logloss < baseline_logloss,
    }, current_probability


def _simulate(
    labels: np.ndarray,
    next_probability: np.ndarray,
    leaf_to_macro: dict[int, int],
    simulations: int = 50_000,
    horizon: int = 52,
) -> dict:
    rng = np.random.default_rng(SETTINGS.random_seed)
    classes = np.arange(6)
    transition = _transition_probability(labels, classes, alpha=1.0)
    paths = np.empty((simulations, horizon + 1), dtype=np.int8)
    paths[:, 0] = labels[-1]
    paths[:, 1] = rng.choice(classes, size=simulations, p=next_probability)
    for step in range(2, horizon + 1):
        for state in classes:
            idx = np.where(paths[:, step - 1] == state)[0]
            if len(idx):
                paths[idx, step] = rng.choice(classes, size=len(idx), p=transition[state])
    occupancy = {
        str(h): np.bincount(paths[:, h], minlength=6).astype(float) / simulations
        for h in (1, 4, 12, 26, 52)
    }
    first_change = np.full(simulations, horizon + 1)
    for i in range(simulations):
        changed = np.where(paths[i, 1:] != paths[i, 0])[0]
        if len(changed):
            first_change[i] = changed[0] + 1
    leaf_exit_risk = {
        str(h): float(np.mean(first_change <= h)) for h in (2, 4, 8, 12, 26, 52)
    }
    macro_paths = np.vectorize(leaf_to_macro.get)(paths)
    macro_change = np.full(simulations, horizon + 1)
    for i in range(simulations):
        # A macro-regime crossing must persist for three consecutive weeks;
        # isolated weekly flips are state noise, not a major transition.
        for step in range(1, horizon - 1):
            if np.all(macro_paths[i, step: step + 3] != macro_paths[i, 0]):
                macro_change[i] = step
                break
    regime_transition_risk = {
        str(h): float(np.mean(macro_change <= h)) for h in (2, 4, 8, 12, 26, 52)
    }
    # Scenario families are defined by the dominant state over months 2–12,
    # with first-month state breaking ties. This is deterministic and auditable.
    family_counts: Counter[tuple[int, int]] = Counter()
    for path in paths:
        dominant = int(np.argmax(np.bincount(path[5:], minlength=6)))
        family_counts[(int(path[1]), dominant)] += 1
    families = [
        {"entryState": a, "dominantState": b, "probability": count / simulations}
        for (a, b), count in family_counts.most_common(8)
    ]
    return {
        "simulations": simulations,
        "horizonWeeks": horizon,
        "occupancy": occupancy,
        "leafStateExitRisk": leaf_exit_risk,
        "regimeTransitionRisk": regime_transition_risk,
        "scenarioFamilies": families,
    }


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    base, daily_state = _load_daily()
    weekly, weekly_vectors = _weekly(base, daily_state)
    labels = KMeans(
        n_clusters=6, random_state=SETTINGS.random_seed, n_init=80
    ).fit_predict(weekly_vectors)
    macro = KMeans(
        n_clusters=2, random_state=SETTINGS.random_seed, n_init=80
    ).fit_predict(weekly_vectors)
    profiles, names = _state_names(weekly, labels)
    validation, next_probability = _walk_forward(labels, weekly_vectors)
    leaf_to_macro = {
        state_id: int(np.bincount(macro[labels == state_id], minlength=2).argmax())
        for state_id in range(6)
    }
    simulation = _simulate(labels, next_probability, leaf_to_macro)

    weekly_rows = []
    for row, state_id, macro_id in zip(weekly.to_dict("records"), labels, macro):
        weekly_rows.append({
            **row, "state": int(state_id), "stateName": names[int(state_id)],
            "macroState": int(macro_id),
        })
    current = int(labels[-1])
    payload = {
        "metadata": {
            "forecastOrigin": END,
            "trainingStart": START,
            "days": len(base),
            "weeks": len(weekly),
            "completeWeeks": int((weekly["days"] == 7).sum()),
            "partialFinalWeekDays": int(weekly.iloc[-1]["days"]),
            "forecastSafe": True,
            "forbiddenHindsightFeatures": [
                "residual_impact", "later_reference_count",
                "forward_alignment", "forward_recurrence",
            ],
        },
        "states": profiles,
        "weeklyHistory": weekly_rows,
        "current": {
            "state": current,
            "stateName": names[current],
            "macroState": int(macro[-1]),
            "nextWeek": [
                {"state": int(i), "name": names[int(i)], "probability": float(value)}
                for i, value in sorted(enumerate(next_probability), key=lambda item: item[1], reverse=True)
            ],
        },
        "validation": validation,
        "simulation": simulation,
        "method": {
            "ensemble": "Walk-forward weighted blend of Laplace-smoothed Markov, explicit-duration, analog-kernel, and regularized multinomial-logistic forecasts.",
            "weights": "Each component weight is proportional to exp(-walk-forward log loss).",
            "simulation": "50,000 weekly Monte Carlo paths from the calibrated next-week posterior and smoothed observed transition matrix.",
            "scope": "Internal-state reachability only; no unobserved external-world event generator.",
        },
    }
    OUTPUT.write_text(
        json.dumps(_jsonable(payload), indent=2, allow_nan=False), encoding="utf-8"
    )
    report_lines = [
        "# Prospective forecast — 2026-07-29 origin",
        "",
        f"- Training window: {START} through {END}",
        f"- Weekly observations: {len(weekly)}",
        f"- Walk-forward folds: {validation['folds']}",
        f"- Ensemble log loss: {validation['ensemble']['logLoss']:.4f}",
        f"- Frequency-baseline log loss: {validation['frequencyBaseline']['logLoss']:.4f}",
        f"- Top-1 accuracy: {validation['ensemble']['top1Accuracy']:.1%}",
        f"- Top-2 accuracy: {validation['ensemble']['top2Accuracy']:.1%}",
        f"- Current state: S{current} — {names[current]}",
        "",
        "## Next week",
        "",
    ]
    report_lines += [
        f"- S{i} {names[i]}: {probability:.1%}"
        for i, probability in sorted(enumerate(next_probability), key=lambda item: item[1], reverse=True)
    ]
    report_lines += ["", "## Transition risk", ""]
    report_lines += [
        f"- Within {horizon} weeks: {risk:.1%}"
        for horizon, risk in simulation["regimeTransitionRisk"].items()
    ]
    REPORT.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(OUTPUT), "weeks": len(weekly), "current": names[current],
        "validation": validation["ensemble"],
        "baseline": validation["frequencyBaseline"],
        "nextWeek": payload["current"]["nextWeek"],
    }, indent=2))


if __name__ == "__main__":
    main()
