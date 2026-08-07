from __future__ import annotations

import argparse
import json
import math
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

import lancedb
import networkx as nx
import numpy as np
import pandas as pd
from persim import wasserstein
from ripser import ripser
from scipy.cluster.hierarchy import cophenet, linkage
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.signal import lombscargle
from scipy.stats import entropy, linregress, spearmanr
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import adjusted_mutual_info_score, mutual_info_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler, StandardScaler

from .config import DB_PATH, OUTPUT_ROOT, PARQUET_ROOT, SETTINGS, VECTOR_PATH
from .embed import _table_name
from .models import load_sentence_transformer


RESULT_ROOT = OUTPUT_ROOT / "topology"
EPS = 1e-12


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, np.integer):
        return value.item()
    if isinstance(value, (pd.Timestamp,)):
        return str(value.date())
    return value


def _write_json(name: str, value) -> None:
    (RESULT_ROOT / name).write_text(
        json.dumps(_jsonable(value), indent=2, allow_nan=False), encoding="utf-8"
    )


def _normalized_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, EPS)


def _finite_rows(x: np.ndarray) -> np.ndarray:
    return np.all(np.isfinite(x), axis=1)


def _state_data(
    start_date: str | None = None,
    include_sensors: bool = True,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], np.ndarray]:
    features = pd.read_parquet(PARQUET_ROOT / "daily_features.parquet")
    impact = pd.read_parquet(PARQUET_ROOT / "daily_impact_semantic.parquet")
    sensors = pd.read_parquet(PARQUET_ROOT / "daily_sensors.parquet")
    attractors = pd.read_parquet(PARQUET_ROOT / "attractor_states.parquet")
    base = features.merge(impact, on="behavioral_day", suffixes=("", "_impact"))
    base = base.merge(sensors, on="behavioral_day", how="left")
    base = base.merge(attractors, on="behavioral_day", how="left")
    base["date"] = pd.to_datetime(base["behavioral_day"])
    base = base.sort_values("date").reset_index(drop=True)
    if start_date is not None:
        base = base[base["date"] >= pd.Timestamp(start_date)].reset_index(drop=True)

    vectors = pd.read_parquet(PARQUET_ROOT / "daily_vector_states.parquet")
    views: dict[str, np.ndarray] = {}
    for view in ("lived", "interpretive", "prospective"):
        rows = vectors[vectors["view_type"] == view].set_index("behavioral_day")
        dim = int(rows["dimensions"].max())
        matrix = np.full((len(base), dim), np.nan, dtype=np.float32)
        for i, day in enumerate(base["behavioral_day"]):
            if day not in rows.index:
                continue
            value = rows.loc[day, "vector"]
            if isinstance(value, pd.Series):
                value = value.iloc[0]
            if value is not None:
                matrix[i] = np.asarray(value, dtype=np.float32)
        views[view] = matrix

    core_structural_columns = [
        "original_reflection_count", "question_count", "technical_count",
        "social_count", "distinct_people", "distinct_threads", "active_minutes",
        "late_night_count", "chars_original", "reflection_references",
        "retrospective_count", "prospective_count",
        "activation_intensity", "lived_novelty_30", "interpretive_novelty_30",
        "prospective_novelty_30", "residual_impact",
    ]
    sensor_columns = [
        "media_count", "music_plays", "music_minutes", "music_repeat_ratio", "youtube_events",
        "browser_events", "web_searches", "activation_intensity",
    ]
    # activation_intensity is a core derived signal, not an external sensor.
    sensor_columns.remove("activation_intensity")
    structural_columns = core_structural_columns + (sensor_columns if include_sensors else [])
    structural = base[structural_columns].fillna(0).astype(float)
    structural = RobustScaler().fit_transform(np.log1p(np.maximum(structural, 0)))

    lived = views["lived"].copy()
    observed = _finite_rows(lived)
    pca = PCA(n_components=min(24, observed.sum() - 1), random_state=SETTINGS.random_seed)
    semantic = np.zeros((len(base), pca.n_components), dtype=float)
    semantic[observed] = pca.fit_transform(lived[observed])
    state = StandardScaler().fit_transform(np.column_stack([semantic, structural]))
    return base, views, state


def _delayed_mutual_information(y: np.ndarray, max_lag: int = 90) -> dict:
    y = np.asarray(y, float)
    y = np.nan_to_num(y, nan=np.nanmedian(y))
    bins = np.quantile(y, np.linspace(0, 1, 13))
    bins = np.unique(bins)
    digitized = np.digitize(y, bins[1:-1])
    values = []
    for lag in range(1, min(max_lag, len(y) // 4) + 1):
        values.append(mutual_info_score(digitized[:-lag], digitized[lag:]))
    arr = np.asarray(values)
    local = [
        i + 1 for i in range(1, len(arr) - 1)
        if arr[i] < arr[i - 1] and arr[i] <= arr[i + 1]
    ]
    tau = local[0] if local else int(np.argmin(arr[: min(30, len(arr))]) + 1)
    return {"tau": tau, "lags": list(range(1, len(arr) + 1)), "mutual_information": arr}


def _delay_embed(y: np.ndarray, tau: int, dimension: int) -> np.ndarray:
    end = (dimension - 1) * tau
    return np.column_stack([y[end - j * tau: len(y) - j * tau] for j in range(dimension)])


def _false_nearest_dimension(y: np.ndarray, tau: int, max_dim: int = 10) -> dict:
    fractions = []
    for dim in range(1, max_dim + 1):
        current = _delay_embed(y, tau, dim)
        if dim == max_dim:
            fractions.append(np.nan)
            break
        nxt = _delay_embed(y, tau, dim + 1)
        current = current[tau:]
        if len(current) < 25:
            fractions.append(np.nan)
            continue
        nn = NearestNeighbors(n_neighbors=2).fit(current)
        dist, ind = nn.kneighbors(current)
        nearest = ind[:, 1]
        old = np.maximum(dist[:, 1], EPS)
        extra = np.abs(nxt[:, -1] - nxt[nearest, -1])
        ratio = extra / old
        scale_ratio = np.sqrt(old**2 + extra**2) / max(np.std(y), EPS)
        fractions.append(float(np.mean((ratio > 10) | (scale_ratio > 2))))
    below = next((i + 1 for i, f in enumerate(fractions) if np.isfinite(f) and f < 0.05), None)
    finite = [(i + 1, f) for i, f in enumerate(fractions) if np.isfinite(f)]
    chosen = below if below is not None else min(finite, key=lambda pair: pair[1])[0]
    return {"dimension": chosen, "false_neighbor_fraction": fractions}


def _persistence_summary(points: np.ndarray, max_points: int = 900) -> dict:
    points = StandardScaler().fit_transform(points)
    if len(points) > max_points:
        idx = np.linspace(0, len(points) - 1, max_points, dtype=int)
        points = points[idx]
    diagrams = ripser(points, maxdim=1)["dgms"]
    output = {}
    for dim, diagram in enumerate(diagrams):
        finite = diagram[np.isfinite(diagram[:, 1])]
        persistence = finite[:, 1] - finite[:, 0] if len(finite) else np.array([])
        output[f"h{dim}"] = {
            "count": int(len(finite)),
            "max_persistence": float(persistence.max()) if len(persistence) else 0.0,
            "total_persistence": float(persistence.sum()) if len(persistence) else 0.0,
            "diagram": finite,
        }
    return output


def delay_embedding_tests(
    base: pd.DataFrame,
    views: dict[str, np.ndarray],
    include_external_sensors: bool = True,
) -> dict:
    observables = {
        "activation": base["activation_intensity"].fillna(0).to_numpy(float),
        "reflection": np.log1p(base["original_reflection_count"].to_numpy(float)),
        "technical": np.log1p(base["technical_count"].to_numpy(float)),
        "social": np.log1p(base["social_count"].to_numpy(float)),
        "semantic_novelty": base["lived_novelty_30"].fillna(0).to_numpy(float),
        "prospective": np.log1p(base["prospective_count"].to_numpy(float)),
    }
    if include_external_sensors:
        observables["music"] = np.log1p(base["music_plays"].fillna(0).to_numpy(float))
    results = {}
    diagrams = {}
    embeddings = {}
    for name, raw in observables.items():
        y = StandardScaler().fit_transform(raw.reshape(-1, 1)).ravel()
        mi = _delayed_mutual_information(y)
        fnn = _false_nearest_dimension(y, mi["tau"])
        emb = _delay_embed(y, mi["tau"], fnn["dimension"])
        ph = _persistence_summary(emb)
        results[name] = {"mutual_information": mi, "fnn": fnn, "persistence": ph}
        embeddings[name] = emb
        diagrams[name] = ph["h1"]["diagram"]

    names = list(observables)
    distance = np.zeros((len(names), len(names)))
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            distance[i, j] = wasserstein(diagrams[a], diagrams[b], matching=False)
    results["cross_observable_h1_wasserstein"] = {"names": names, "matrix": distance}

    # Neighbor agreement tests whether independently reconstructed observables preserve
    # the same local temporal neighborhoods, after aligning their valid tails.
    agreement = {}
    rng = np.random.default_rng(SETTINGS.random_seed)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            n = min(len(embeddings[a]), len(embeddings[b]))
            ea, eb = embeddings[a][-n:], embeddings[b][-n:]
            ka = NearestNeighbors(n_neighbors=11).fit(ea).kneighbors(return_distance=False)[:, 1:]
            kb = NearestNeighbors(n_neighbors=11).fit(eb).kneighbors(return_distance=False)[:, 1:]
            scores = [len(set(x).intersection(y)) / 10 for x, y in zip(ka, kb)]
            observed = float(np.mean(scores))
            null = []
            for _ in range(100):
                shift = int(rng.integers(max(30, n // 20), max(31, n - n // 20)))
                shifted = (kb + shift) % n
                null.append(float(np.mean([
                    len(set(x).intersection(y)) / 10 for x, y in zip(ka, shifted)
                ])))
            agreement[f"{a}__{b}"] = {
                "observed": observed,
                "circular_shift_null_mean": float(np.mean(null)),
                "fold_over_null": observed / max(float(np.mean(null)), EPS),
                "p_value": (1 + sum(x >= observed for x in null)) / (len(null) + 1),
            }
    results["cross_observable_neighbor_overlap"] = agreement
    return results


def symbolic_grammar_tests(base: pd.DataFrame) -> dict:
    result = {}
    for level, col in (("leaf", "leaf_attractor_id"), ("macro", "macro_attractor_id")):
        seq = base[col].fillna(-1).astype(int).to_numpy()
        states = sorted(set(seq))
        transitions = Counter(zip(seq[:-1], seq[1:]))
        direct_forbidden = []
        gateways = []
        for a in states:
            for c in states:
                if a == c or transitions[(a, c)]:
                    continue
                two_step = [b for b in states if transitions[(a, b)] and transitions[(b, c)]]
                if two_step:
                    direct_forbidden.append({"from": a, "to": c, "observed_gateways": two_step})
        for b in states:
            count = sum(b in x["observed_gateways"] for x in direct_forbidden)
            gateways.append((b, count))

        word_stats = {}
        for k in range(1, 9):
            words = Counter(tuple(seq[i:i + k]) for i in range(len(seq) - k + 1))
            probs = np.asarray(list(words.values()), float)
            probs /= probs.sum()
            word_stats[k] = {
                "distinct": len(words),
                "topological_entropy_estimate": math.log(max(len(words), 1)) / k,
                "block_entropy": float(entropy(probs)),
            }
        era_stats = {}
        for era, mask in {
            "pre_2024": base["date"] < "2024-01-01",
            "2024_2025": (base["date"] >= "2024-01-01") & (base["date"] < "2026-01-01"),
            "2026": base["date"] >= "2026-01-01",
        }.items():
            sub = seq[np.asarray(mask)]
            if len(sub) < 20:
                continue
            counts = Counter(zip(sub[:-1], sub[1:]))
            probs = np.asarray(list(counts.values()), float)
            probs /= probs.sum()
            era_stats[era] = {
                "transition_entropy": float(entropy(probs)),
                "transition_types": len(counts),
                "normalized_entropy": float(entropy(probs) / math.log(max(len(counts), 2))),
            }
        result[level] = {
            "states": states,
            "transition_counts": {f"{a}->{b}": n for (a, b), n in transitions.items()},
            "forbidden_direct_with_observed_detours": direct_forbidden,
            "gateway_rank": sorted(gateways, key=lambda x: x[1], reverse=True),
            "word_statistics": word_stats,
            "era_statistics": era_stats,
        }
    return result


def _window_descriptor(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pca = PCA(n_components=min(6, points.shape[1], len(points) - 1), random_state=SETTINGS.random_seed)
    z = pca.fit_transform(points)
    ph = _persistence_summary(z, max_points=240)
    h1 = ph["h1"]["diagram"]
    half = len(z) // 2
    before, after = z[:half], z[half:]
    descriptor = np.array([
        ph["h0"]["max_persistence"], ph["h0"]["total_persistence"],
        ph["h1"]["max_persistence"], ph["h1"]["total_persistence"],
        np.linalg.norm(after.mean(0) - before.mean(0)),
        np.mean(np.linalg.norm(np.diff(z, axis=0), axis=1)),
        np.mean(np.var(before, axis=0)), np.mean(np.var(after, axis=0)),
        pca.explained_variance_ratio_[0], pca.explained_variance_ratio_[:3].sum(),
    ])
    return descriptor, h1


def hinge_echo_tests(base: pd.DataFrame, state: np.ndarray) -> dict:
    hinges = pd.read_parquet(PARQUET_ROOT / "hinge_candidates.parquet")
    day_index = {day: i for i, day in enumerate(base["behavioral_day"])}
    outputs = {}
    for radius in (30, 60, 90, 180):
        rows, diagrams, dates, shifts = [], [], [], []
        for day in hinges["behavioral_day"]:
            if day not in day_index:
                continue
            i = day_index[day]
            lo, hi = i - radius, i + radius + 1
            if lo < 0 or hi > len(state):
                continue
            descriptor, diagram = _window_descriptor(state[lo:hi])
            rows.append(descriptor)
            diagrams.append(diagram)
            dates.append(day)
            shifts.append(state[i + 1:hi].mean(0) - state[lo:i].mean(0))
        x = StandardScaler().fit_transform(np.vstack(rows))
        euclidean = squareform(pdist(x))
        topo = np.zeros_like(euclidean)
        for i in range(len(dates)):
            for j in range(i + 1, len(dates)):
                topo[i, j] = topo[j, i] = wasserstein(diagrams[i], diagrams[j], matching=False)
        shift_matrix = _normalized_rows(np.vstack(shifts))
        direction = np.clip((1 - shift_matrix @ shift_matrix.T) / 2, 0, 1)
        combined = 0.35 * (euclidean / max(np.median(euclidean[euclidean > 0]), EPS))
        if np.any(topo > 0):
            combined += 0.30 * (topo / max(np.median(topo[topo > 0]), EPS))
        combined += 0.35 * direction
        pairs = []
        for i in range(len(dates)):
            for j in range(i + 1, len(dates)):
                gap = abs((pd.Timestamp(dates[j]) - pd.Timestamp(dates[i])).days)
                if gap >= 180:
                    pairs.append({
                        "day_a": dates[i], "day_b": dates[j], "gap_days": gap,
                        "descriptor_distance": float(euclidean[i, j]),
                        "h1_wasserstein": float(topo[i, j]),
                        "shift_direction_cosine": float(shift_matrix[i] @ shift_matrix[j]),
                        "combined_distance": float(combined[i, j]),
                    })
        pairs.sort(key=lambda r: r["combined_distance"])
        cluster_count = min(8, max(2, len(dates) // 6))
        labels = AgglomerativeClustering(
            n_clusters=cluster_count, metric="precomputed", linkage="average"
        ).fit_predict(combined)
        outputs[str(radius)] = {
            "dates": dates,
            "descriptors": np.vstack(rows),
            "cluster_labels": labels,
            "strongest_cross_era_echoes": pairs[:500],
        }
    return outputs


def conley_morse_approximation(base: pd.DataFrame) -> dict:
    output = {}
    for level, col in (("leaf", "leaf_attractor_id"), ("macro", "macro_attractor_id")):
        seq = base[col].fillna(-1).astype(int).to_numpy()
        counts = Counter((int(a), int(b)) for a, b in zip(seq[:-1], seq[1:]) if a != -1 and b != -1)
        outgoing = Counter()
        for (a, _), count in counts.items():
            outgoing[a] += count
        graph = nx.DiGraph()
        for (a, b), count in counts.items():
            probability = count / outgoing[a]
            if count >= 3 and probability >= 0.02:
                graph.add_edge(a, b, weight=count, probability=probability)
        recurrent = list(nx.strongly_connected_components(graph))
        condensation = nx.condensation(graph, recurrent)
        mapping = condensation.graph["mapping"]
        output[level] = {
            "strongly_connected_morse_sets": [sorted(x) for x in recurrent],
            "morse_dag_edges": sorted((int(a), int(b)) for a, b in condensation.edges()),
            "state_to_morse_set": mapping,
            "source_morse_sets": [int(n) for n in condensation if condensation.in_degree(n) == 0],
            "sink_morse_sets": [int(n) for n in condensation if condensation.out_degree(n) == 0],
            "transition_edges": [
                {"from": int(a), "to": int(b), "count": int(data["weight"]),
                 "probability": float(data["probability"])}
                for a, b, data in graph.edges(data=True)
            ],
            "edge_threshold": {"minimum_count": 3, "minimum_conditional_probability": 0.02,
                               "unassigned_state_removed": True},
        }
    return output


def _effective_information(labels: np.ndarray) -> dict:
    labels = np.asarray(labels, int)
    states = sorted(set(labels))
    idx = {s: i for i, s in enumerate(states)}
    counts = np.zeros((len(states), len(states)))
    for a, b in zip(labels[:-1], labels[1:]):
        counts[idx[a], idx[b]] += 1
    rows = counts.sum(1, keepdims=True)
    transition = np.divide(counts, rows, out=np.zeros_like(counts), where=rows > 0)
    intervention = np.ones(len(states)) / len(states)
    output = intervention @ transition
    ei = 0.0
    for i in range(len(states)):
        for j in range(len(states)):
            p = intervention[i] * transition[i, j]
            if p > 0 and output[j] > 0:
                ei += p * math.log2(transition[i, j] / output[j])
    determinism = math.log2(max(len(states), 1)) - np.mean([
        entropy(row[row > 0], base=2) if np.any(row > 0) else 0 for row in transition
    ])
    return {
        "states": len(states), "effective_information_bits": ei,
        "normalized_effective_information": ei / max(math.log2(max(len(states), 2)), EPS),
        "determinism_bits": determinism, "transition_matrix": transition,
    }


def causal_emergence_tests(base: pd.DataFrame, state: np.ndarray) -> dict:
    rng = np.random.default_rng(SETTINGS.random_seed)
    output = {}
    def measured(labels):
        observed = _effective_information(labels)
        null = []
        for _ in range(200):
            shuffled = np.asarray(labels).copy()
            rng.shuffle(shuffled)
            null.append(_effective_information(shuffled)["normalized_effective_information"])
        observed["order_shuffle_null_mean"] = float(np.mean(null))
        observed["order_shuffle_p"] = (1 + sum(x >= observed["normalized_effective_information"] for x in null)) / (len(null) + 1)
        return observed
    output["leaf"] = measured(base["leaf_attractor_id"].fillna(-1).to_numpy(int))
    output["macro"] = measured(base["macro_attractor_id"].fillna(-1).to_numpy(int))
    for k in (3, 6, 12, 24, 48):
        labels = KMeans(n_clusters=k, random_state=SETTINGS.random_seed, n_init=20).fit_predict(state)
        output[f"daily_k{k}"] = measured(labels)
    # Non-overlapping blocks avoid manufacturing determinism through overlapping windows.
    for window in (7, 30, 90):
        usable = len(state) // window * window
        blocked = state[:usable].reshape(-1, window, state.shape[1]).mean(axis=1)
        clusters = min(6, len(blocked))
        labels = KMeans(n_clusters=clusters, random_state=SETTINGS.random_seed, n_init=20).fit_predict(blocked)
        output[f"rolling_{window}d_k{clusters}"] = measured(labels)
    return output


def bifurcation_tests(base: pd.DataFrame, state: np.ndarray) -> dict:
    rows = []
    window, step = 180, 30
    reduced = PCA(n_components=6, random_state=SETTINGS.random_seed).fit_transform(state)
    for end in range(window, len(base) + 1, step):
        points = reduced[end - window:end]
        ph = _persistence_summary(points, max_points=240)
        rows.append({
            "end_day": base.iloc[end - 1]["behavioral_day"],
            "h1_max": ph["h1"]["max_persistence"],
            "h1_total": ph["h1"]["total_persistence"],
            "h1_count": ph["h1"]["count"],
            "intrinsic_dimension_95": int(np.searchsorted(
                np.cumsum(PCA().fit(points).explained_variance_ratio_), 0.95
            ) + 1),
        })
    frame = pd.DataFrame(rows)
    for col in ("h1_max", "h1_total", "h1_count", "intrinsic_dimension_95"):
        frame[f"{col}_change"] = frame[col].diff()
    strongest = frame.reindex(frame["h1_total_change"].abs().sort_values(ascending=False).index).head(20)
    frame.to_csv(RESULT_ROOT / "rolling_bifurcations.csv", index=False)
    return {"window_days": window, "step_days": step, "strongest_topology_changes": strongest.to_dict("records")}


def divergence_tests(base: pd.DataFrame, state: np.ndarray) -> dict:
    dense = (base["date"] >= "2024-01-01") & (base["event_count"] > 0)
    dense_idx = np.where(np.asarray(dense))[0]
    # Keep only contiguous calendar positions whose full future horizon is observed.
    valid_start = set(dense_idx)
    x_all = PCA(n_components=12, random_state=SETTINGS.random_seed).fit_transform(state)
    x = x_all
    nn = NearestNeighbors(n_neighbors=80).fit(x)
    distances, indices = nn.kneighbors(x)
    pairs = []
    for i in dense_idx:
        if i >= len(x) - 61 or any(i + k not in valid_start for k in range(61)):
            continue
        for d, j in zip(distances[i, 1:], indices[i, 1:]):
            if (abs(i - j) >= 180 and j in valid_start and j < len(x) - 61
                    and d > 1e-6 and all(j + k in valid_start for k in range(61))):
                pairs.append((i, int(j), float(d)))
                break
    pairs = sorted(pairs, key=lambda z: z[2])[:500]
    horizon = 60
    curves = []
    for i, j, _ in pairs:
        curves.append([np.linalg.norm(x[i + k] - x[j + k]) for k in range(horizon)])
    mean = np.mean(curves, axis=0)
    fit_end = 21
    slope = linregress(np.arange(1, fit_end), np.log(np.maximum(mean[1:fit_end], EPS)))
    return {
        "pair_count": len(pairs), "mean_divergence_curve": mean,
        "finite_time_lambda_per_day": slope.slope, "fit_r_squared": slope.rvalue**2,
        "doubling_days": math.log(2) / slope.slope if slope.slope > 0 else None,
        "closest_pair_examples": [
            {"day_a": base.iloc[i]["behavioral_day"], "day_b": base.iloc[j]["behavioral_day"], "distance": d}
            for i, j, d in pairs[:30]
        ],
    }


def periodicity_tests(base: pd.DataFrame) -> dict:
    hinges = pd.read_parquet(PARQUET_ROOT / "hinge_candidates.parquet")
    idx = {d: i for i, d in enumerate(base["behavioral_day"])}
    signal = np.zeros(len(base))
    for d in hinges["behavioral_day"]:
        if d in idx:
            signal[idx[d]] = 1
    signal -= signal.mean()
    frequencies = np.fft.rfftfreq(len(signal), d=1.0)
    valid = (frequencies >= 1 / 730) & (frequencies <= 1 / 7)
    valid_freq = frequencies[valid]
    periods = 1 / valid_freq
    power = np.abs(np.fft.rfft(signal)[valid]) ** 2
    power /= max(power.sum(), EPS)
    observed = sorted(zip(power, periods), reverse=True)[:30]
    rng = np.random.default_rng(SETTINGS.random_seed)
    maxima = []
    for _ in range(500):
        # Preserve yearly event counts and weekday mix by permuting within calendar years.
        surrogate = np.zeros(len(signal))
        for year in sorted(base["date"].dt.year.unique()):
            positions = np.where(base["date"].dt.year.to_numpy() == year)[0]
            values = signal[positions].copy()
            rng.shuffle(values)
            surrogate[positions] = values
        surrogate_power = np.abs(np.fft.rfft(surrogate)[valid]) ** 2
        surrogate_power /= max(surrogate_power.sum(), EPS)
        maxima.append(float(surrogate_power.max()))
    max_power = float(power.max())
    return {
        "top_periods": [{"period_days": p, "power": q} for q, p in observed],
        "global_peak_surrogate_p": (1 + sum(x >= max_power for x in maxima)) / (len(maxima) + 1),
        "surrogate_max_power_quantiles": np.quantile(maxima, [0.5, 0.9, 0.95, 0.99]),
    }


def geometry_tests(base: pd.DataFrame, state: np.ndarray) -> dict:
    rng = np.random.default_rng(SETTINGS.random_seed)
    reduced = PCA(n_components=16, random_state=SETTINGS.random_seed).fit_transform(state)
    sample_idx = rng.choice(len(reduced), min(1400, len(reduced)), replace=False)
    sample = reduced[sample_idx]
    distances = squareform(pdist(sample))

    # Ultrametricity: fraction of triangles whose two largest sides are approximately equal.
    triangles = rng.integers(0, len(sample), size=(100_000, 3))
    triangles = triangles[np.all(np.diff(np.sort(triangles, axis=1), axis=1) != 0, axis=1)]
    sides = np.sort(np.column_stack([
        distances[triangles[:, 0], triangles[:, 1]],
        distances[triangles[:, 0], triangles[:, 2]],
        distances[triangles[:, 1], triangles[:, 2]],
    ]), axis=1)
    ultrametric = float(np.mean((sides[:, 2] - sides[:, 1]) / np.maximum(sides[:, 2], EPS) < 0.05))
    z = linkage(squareform(distances), method="average")
    cophenetic, _ = cophenet(z, squareform(distances))

    # Correlation dimension over the central scaling region.
    nonzero = distances[np.triu_indices_from(distances, 1)]
    epsilons = np.quantile(nonzero, np.linspace(0.02, 0.35, 24))
    correlation = np.array([np.mean(nonzero < e) for e in epsilons])
    fit = linregress(np.log(epsilons), np.log(np.maximum(correlation, EPS)))

    # Sampled Gromov delta-hyperbolicity through the four-point condition.
    quads = rng.integers(0, len(sample), size=(50_000, 4))
    deltas = []
    for a, b, c, d in quads:
        sums = sorted([
            distances[a, b] + distances[c, d],
            distances[a, c] + distances[b, d],
            distances[a, d] + distances[b, c],
        ])
        deltas.append((sums[2] - sums[1]) / 2)
    diameter = float(nonzero.max())
    observed = {
        "ultrametric_triangle_fraction_5pct": ultrametric,
        "hierarchical_cophenetic_correlation": float(cophenetic),
        "correlation_dimension": fit.slope,
        "correlation_dimension_r_squared": fit.rvalue**2,
        "correlation_dimension_scaling_epsilon": epsilons,
        "correlation_integral": correlation,
        "sampled_delta_hyperbolicity_median": float(np.median(deltas)),
        "sampled_delta_hyperbolicity_95pct": float(np.quantile(deltas, 0.95)),
        "delta_over_diameter_95pct": float(np.quantile(deltas, 0.95) / diameter),
    }
    # Marginal-preserving independent-column shuffles test whether the geometry
    # comes from joint structure rather than each coordinate's distribution.
    null_rows = []
    for _ in range(30):
        null_state = sample.copy()
        for col in range(null_state.shape[1]):
            rng.shuffle(null_state[:, col])
        null_dist = squareform(pdist(null_state))
        tri = rng.integers(0, len(null_state), size=(20_000, 3))
        tri = tri[np.all(np.diff(np.sort(tri, axis=1), axis=1) != 0, axis=1)]
        null_sides = np.sort(np.column_stack([
            null_dist[tri[:, 0], tri[:, 1]], null_dist[tri[:, 0], tri[:, 2]],
            null_dist[tri[:, 1], tri[:, 2]],
        ]), axis=1)
        null_ultra = float(np.mean(
            (null_sides[:, 2] - null_sides[:, 1]) / np.maximum(null_sides[:, 2], EPS) < 0.05
        ))
        null_nonzero = null_dist[np.triu_indices_from(null_dist, 1)]
        null_eps = np.quantile(null_nonzero, np.linspace(0.02, 0.35, 24))
        null_corr = np.array([np.mean(null_nonzero < e) for e in null_eps])
        null_fit = linregress(np.log(null_eps), np.log(np.maximum(null_corr, EPS)))
        null_rows.append((null_ultra, null_fit.slope))
    null_rows = np.asarray(null_rows)
    observed["independent_column_shuffle_null"] = {
        "ultrametric_mean": float(null_rows[:, 0].mean()),
        "ultrametric_p": float((1 + np.sum(null_rows[:, 0] >= ultrametric)) / 31),
        "correlation_dimension_mean": float(null_rows[:, 1].mean()),
        "correlation_dimension_p_lower": float((1 + np.sum(null_rows[:, 1] <= fit.slope)) / 31),
    }
    return observed


def graph_time_tests() -> dict:
    conn = sqlite3.connect(DB_PATH)
    links = pd.read_sql_query("SELECT * FROM semantic_links", conn)
    refs = pd.read_sql_query(
        "SELECT recorded_at, target_start, relation_type, confidence FROM temporal_references "
        "WHERE target_start IS NOT NULL", conn
    )
    conn.close()
    graph = nx.DiGraph()
    for row in links.itertuples():
        graph.add_edge(row.target_day, row.source_day, weight=row.confidence, kind=row.relation_type)
    if graph.number_of_nodes():
        undirected = graph.to_undirected()
        articulation = list(nx.articulation_points(undirected))
        centrality = nx.betweenness_centrality(undirected, k=min(500, len(undirected)), seed=SETTINGS.random_seed)
        generators = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:30]
        cycles = nx.cycle_basis(undirected)
    else:
        articulation, generators, cycles = [], [], []
    refs["recorded_at"] = pd.to_datetime(refs["recorded_at"], errors="coerce", utc=True)
    refs["target_start"] = pd.to_datetime(refs["target_start"], errors="coerce", utc=True)
    refs["lag_days"] = (refs["recorded_at"] - refs["target_start"]).dt.total_seconds() / 86400
    refs = refs[np.isfinite(refs["lag_days"]) & (refs["lag_days"] >= 0)]
    return {
        "interpretation_graph_nodes": graph.number_of_nodes(),
        "interpretation_graph_edges": graph.number_of_edges(),
        "cycle_rank": graph.to_undirected().number_of_edges() - graph.number_of_nodes() + nx.number_connected_components(graph.to_undirected()) if graph.number_of_nodes() else 0,
        "cycle_count_enumerated": len(cycles),
        "cycle_length_distribution": Counter(map(len, cycles)),
        "articulation_days": articulation,
        "top_generator_days_by_betweenness": generators,
        "retrospective_lag_quantiles_days": refs["lag_days"].quantile([0.25, 0.5, 0.75, 0.9, 0.99]).to_dict(),
        "longest_retrospective_edges": refs.nlargest(30, "lag_days")[
            ["recorded_at", "target_start", "lag_days", "relation_type", "confidence"]
        ].astype({"recorded_at": str, "target_start": str}).to_dict("records"),
    }


def sheaf_obstruction_approximation(base: pd.DataFrame, views: dict[str, np.ndarray]) -> dict:
    # Local sections are independently reduced context views. Pairwise Procrustes maps
    # are restriction maps; triangle-map disagreement measures failure to glue.
    local = {}
    for name, matrix in views.items():
        valid = _finite_rows(matrix)
        z = np.zeros((len(matrix), 8))
        z[valid] = PCA(n_components=8, random_state=SETTINGS.random_seed).fit_transform(matrix[valid])
        local[name] = StandardScaler().fit_transform(z)
    maps = {}
    residuals = {}
    names = list(local)
    for a in names:
        for b in names:
            xa, xb = local[a], local[b]
            u, _, vt = np.linalg.svd(xa.T @ xb, full_matrices=False)
            maps[(a, b)] = u @ vt
            residuals[f"{a}->{b}"] = float(np.linalg.norm(xa @ maps[(a, b)] - xb) / np.linalg.norm(xb))
    cycles = {}
    for a in names:
        for b in names:
            for c in names:
                if len({a, b, c}) < 3:
                    continue
                direct = maps[(a, c)]
                composed = maps[(a, b)] @ maps[(b, c)]
                cycles[f"{a}->{b}->{c}"] = float(np.linalg.norm(direct - composed, ord="fro"))
    rng = np.random.default_rng(SETTINGS.random_seed)
    null = []
    for _ in range(250):
        shuffled = local[names[2]][rng.permutation(len(base))]
        u, _, vt = np.linalg.svd(local[names[0]].T @ shuffled, full_matrices=False)
        null.append(float(np.linalg.norm(local[names[0]] @ (u @ vt) - shuffled) / np.linalg.norm(shuffled)))
    observed = residuals[f"{names[0]}->{names[2]}"]
    return {
        "local_sections": names, "pairwise_gluing_residual": residuals,
        "triangle_cocycle_inconsistency": cycles,
        "lived_to_prospective_null_p": (1 + sum(x <= observed for x in null)) / (len(null) + 1),
        "interpretation": "Procrustes/cocycle obstruction approximation; not a computed sheaf cohomology group.",
    }


def controllability_and_invariants(base: pd.DataFrame, state: np.ndarray) -> dict:
    x = PCA(n_components=12, random_state=SETTINGS.random_seed).fit_transform(state)
    x0, x1 = x[:-1], x[1:]
    a = np.linalg.lstsq(x0, x1, rcond=None)[0].T
    eigvals, left = np.linalg.eig(a.T)
    order = np.argsort(np.abs(eigvals - 1))
    invariants = []
    for idx in order[:8]:
        coordinate = np.real(x @ left[:, idx])
        change = np.diff(coordinate)
        invariants.append({
            "eigenvalue_real": float(np.real(eigvals[idx])),
            "eigenvalue_imag": float(np.imag(eigvals[idx])),
            "mean_absolute_daily_change": float(np.mean(np.abs(change))),
            "signal_std": float(np.std(coordinate)),
            "stability_ratio": float(np.mean(np.abs(change)) / max(np.std(coordinate), EPS)),
            "minimum_days": base.iloc[np.argsort(coordinate)[:10]]["behavioral_day"].tolist(),
            "maximum_days": base.iloc[np.argsort(coordinate)[-10:]]["behavioral_day"].tolist(),
        })
    # Stabilize A for finite Gramian calculation.
    radius = max(abs(np.linalg.eigvals(a)))
    stable = a / max(radius * 1.02, 1.02)
    energies = []
    for driver in range(stable.shape[0]):
        b = np.zeros((stable.shape[0], 1))
        b[driver] = 1
        gram = np.zeros_like(stable)
        power = np.eye(stable.shape[0])
        for _ in range(250):
            gram += power @ b @ b.T @ power.T
            power = stable @ power
        energies.append({
            "driver_coordinate": driver,
            "gramian_trace": float(np.trace(gram)),
            "minimum_eigenvalue": float(np.linalg.eigvalsh(gram).min()),
        })
    return {
        "koopman_like_near_unit_eigenfunctions": invariants,
        "linear_transition_spectral_radius": float(radius),
        "driver_coordinate_controllability": sorted(energies, key=lambda r: r["gramian_trace"], reverse=True),
    }


def renormalization_tests(base: pd.DataFrame, state: np.ndarray) -> dict:
    output = {}
    for window in (1, 7, 30, 90, 180):
        if window == 1:
            x = state
        else:
            usable = len(state) // window * window
            x = state[:usable].reshape(-1, window, state.shape[1]).mean(axis=1)
        reduced = PCA(
            n_components=min(12, x.shape[1], len(x)),
            random_state=SETTINGS.random_seed,
        ).fit_transform(x)
        variance = PCA().fit(reduced).explained_variance_ratio_
        intrinsic = int(np.searchsorted(np.cumsum(variance), 0.95) + 1)
        velocity = np.linalg.norm(np.diff(reduced, axis=0), axis=1)
        clusters = min(6, len(reduced))
        labels = KMeans(n_clusters=clusters, random_state=SETTINGS.random_seed, n_init=20).fit_predict(reduced)
        ei = _effective_information(labels)
        output[str(window)] = {
            "observations": len(x), "intrinsic_dimension_95": intrinsic,
            "mean_velocity": float(velocity.mean()),
            "velocity_cv": float(velocity.std() / max(velocity.mean(), EPS)),
            "effective_information": ei["effective_information_bits"],
            "normalized_effective_information": ei["normalized_effective_information"],
            "clusters": clusters,
        }
    return output


def robustness_tests(base: pd.DataFrame, state: np.ndarray) -> dict:
    rng = np.random.default_rng(SETTINGS.random_seed)
    observed_leaf = base["leaf_attractor_id"].fillna(-1).astype(int).to_numpy()
    compact = PCA(n_components=min(20, state.shape[1]), random_state=SETTINGS.random_seed).fit_transform(state)
    results = {}
    for keep in (0.5, 0.7, 0.9):
        scores = []
        for _ in range(12):
            dims = rng.choice(compact.shape[1], max(2, int(compact.shape[1] * keep)), replace=False)
            labels = KMeans(n_clusters=len(set(observed_leaf)), random_state=int(rng.integers(1e9)), n_init=5).fit_predict(compact[:, dims])
            scores.append(adjusted_mutual_info_score(observed_leaf, labels))
        results[f"dimension_keep_{keep}"] = {
            "ami_median": float(np.median(scores)), "ami_05": float(np.quantile(scores, .05)),
            "ami_95": float(np.quantile(scores, .95)),
        }
    return results


def winding_tests(base: pd.DataFrame, views: dict[str, np.ndarray]) -> dict:
    """Measure chronological winding around semantic anchors in a fixed PCA plane."""
    lived = views["lived"]
    observed = _finite_rows(lived)
    dates = base.loc[observed, "behavioral_day"].to_numpy()
    x = _normalized_rows(lived[observed].astype(float))
    plane = PCA(n_components=2, random_state=SETTINGS.random_seed).fit(x)
    projected = plane.transform(x)
    model = load_sentence_transformer(SETTINGS.embedding_model)
    concepts = [
        "relocation and choosing where to build a life",
        "authorship and making something of my own",
        "attachment and romantic relationship",
        "systems, dependencies, and control planes",
        "home, family, and belonging",
        "being accurately seen and understood",
        "music creation and embodied expression",
    ]
    rng = np.random.default_rng(SETTINGS.random_seed)
    output = {}
    for concept, anchor in zip(
        concepts, model.encode(concepts, normalize_embeddings=True, show_progress_bar=False)
    ):
        center = plane.transform(np.asarray(anchor, float).reshape(1, -1))[0]
        angles = np.unwrap(np.arctan2(projected[:, 1] - center[1], projected[:, 0] - center[0]))
        observed_winding = float((angles[-1] - angles[0]) / (2 * np.pi))
        # The null is a chronological permutation of points, with circular differences.
        null = []
        for _ in range(500):
            order = rng.permutation(len(projected))
            a = np.unwrap(np.arctan2(
                projected[order, 1] - center[1], projected[order, 0] - center[0]
            ))
            null.append(float((a[-1] - a[0]) / (2 * np.pi)))
        null_arr = np.asarray(null)
        p = float((1 + np.sum(np.abs(null_arr) >= abs(observed_winding))) / (len(null_arr) + 1))
        output[concept] = {
            "winding_number": observed_winding,
            "null_mean": float(null_arr.mean()),
            "null_std": float(null_arr.std()),
            "two_sided_p": p,
            "start": str(dates[0]),
            "end": str(dates[-1]),
        }
    return {
        "pca_explained_variance": plane.explained_variance_ratio_,
        "concepts": output,
        "note": "A winding is plane-dependent; significance tests chronology against point-order permutations.",
    }


def cross_embedding_robustness_tests() -> dict:
    """Compare local day geometry under MiniLM and BGE without aligning dimensions."""
    db = lancedb.connect(VECTOR_PATH)
    models = [SETTINGS.embedding_model, "BAAI/bge-m3"]
    day_vectors = {}
    for model_name in models:
        table = db.open_table(_table_name(model_name))
        df = table.to_arrow().select([
            "item_type", "behavioral_day", "view_type", "vector",
        ]).to_pandas()
        df = df[
            (df["item_type"] == "day_facet") &
            (df["view_type"].isin(["lived", "lived_current"]))
        ].copy()
        rows = {}
        for day, group in df.groupby("behavioral_day"):
            matrix = _normalized_rows(np.vstack(group["vector"]).astype(float))
            rows[day] = _normalized_rows(matrix.mean(axis=0, keepdims=True))[0]
        day_vectors[model_name] = rows
    common = sorted(set(day_vectors[models[0]]) & set(day_vectors[models[1]]))
    rng = np.random.default_rng(SETTINGS.random_seed)
    if len(common) > 1_200:
        common = sorted(rng.choice(common, 1_200, replace=False).tolist())
    matrices = {
        name: np.vstack([day_vectors[name][day] for day in common])
        for name in models
    }
    distances = {
        name: squareform(pdist(matrix, metric="cosine"))
        for name, matrix in matrices.items()
    }
    upper = np.triu_indices(len(common), 1)
    rho = float(spearmanr(distances[models[0]][upper], distances[models[1]][upper]).statistic)
    k = min(10, len(common) - 1)
    neighbors = {}
    for name in models:
        order = np.argsort(distances[name], axis=1)[:, 1:k + 1]
        neighbors[name] = order
    overlap = float(np.mean([
        len(set(neighbors[models[0]][i]) & set(neighbors[models[1]][i])) / k
        for i in range(len(common))
    ]))
    null_rho, null_overlap = [], []
    for _ in range(200):
        permutation = rng.permutation(len(common))
        permuted = distances[models[1]][permutation][:, permutation]
        null_rho.append(float(spearmanr(distances[models[0]][upper], permuted[upper]).statistic))
        order = np.argsort(permuted, axis=1)[:, 1:k + 1]
        null_overlap.append(float(np.mean([
            len(set(neighbors[models[0]][i]) & set(order[i])) / k
            for i in range(len(common))
        ])))
    return {
        "models": models,
        "common_days": len(common),
        "pairwise_distance_spearman": rho,
        "distance_null_mean": float(np.mean(null_rho)),
        "distance_p": float((1 + np.sum(np.asarray(null_rho) >= rho)) / 201),
        "ten_neighbor_overlap": overlap,
        "neighbor_null_mean": float(np.mean(null_overlap)),
        "neighbor_p": float((1 + np.sum(np.asarray(null_overlap) >= overlap)) / 201),
    }


def build_report(results: dict) -> str:
    delay = results["delay_embedding"]
    grammar = results["symbolic_grammar"]
    echo = results["hinge_echoes"]["90"]["strongest_cross_era_echoes"][:12]
    causal = results["causal_emergence"]
    geometry = results["geometry"]
    divergence = results["divergence"]
    periodicity = results["periodicity"]
    conley = results["conley_morse"]
    lines = [
        "# Longitudinal Topology and Dynamics Test Report",
        "",
        f"Generated from the daily-state corpus on {pd.Timestamp.now(tz='America/Chicago')}.",
        "",
        "## Dataset",
        "",
        f"- Behavioral days: {results['metadata']['days']:,}",
        f"- Date range: {results['metadata']['start']} through {results['metadata']['end']}",
        "- Primary state: lived semantic PCA plus robust structural and sensor features.",
        "- Independent observables: activation, reflection, technical, social, music, novelty, prospective language.",
        "",
        "## Delay embeddings",
        "",
    ]
    for name in ("activation", "reflection", "technical", "social", "music", "semantic_novelty", "prospective"):
        row = delay[name]
        lines.append(
            f"- **{name}:** tau={row['mutual_information']['tau']} days; "
            f"FNN dimension={row['fnn']['dimension']}; "
            f"max H1 persistence={row['persistence']['h1']['max_persistence']:.4f}."
        )
    lines += [
        "",
        "## Symbolic dynamics",
        "",
        f"- Leaf states: {len(grammar['leaf']['states'])}; strongest gateways: {grammar['leaf']['gateway_rank'][:5]}.",
        f"- Macro states: {len(grammar['macro']['states'])}; strongest gateways: {grammar['macro']['gateway_rank'][:5]}.",
        "",
        "## Conley–Morse transition approximation",
        "",
        f"- Leaf recurrent SCC/Morse sets: {conley['leaf']['strongly_connected_morse_sets']}.",
        f"- Macro recurrent SCC/Morse sets: {conley['macro']['strongly_connected_morse_sets']}.",
        "",
        "## Cross-era topological echoes (90-day radius)",
        "",
    ]
    for row in echo:
        lines.append(
            f"- {row['day_a']} ↔ {row['day_b']} ({row['gap_days']} days): "
            f"combined distance {row['combined_distance']:.4f}, "
            f"H1 Wasserstein {row['h1_wasserstein']:.4f}."
        )
    lines += [
        "",
        "## Causal emergence",
        "",
    ]
    for name, row in causal.items():
        lines.append(
            f"- **{name}:** EI={row['effective_information_bits']:.4f} bits; "
            f"normalized EI={row['normalized_effective_information']:.4f}; states={row['states']}."
        )
    lines += [
        "",
        "## Geometry",
        "",
        f"- Correlation dimension: {geometry['correlation_dimension']:.3f} "
        f"(scaling-fit R²={geometry['correlation_dimension_r_squared']:.3f}).",
        f"- Approximate ultrametric triangle fraction: {geometry['ultrametric_triangle_fraction_5pct']:.3f}.",
        f"- Hierarchical cophenetic correlation: {geometry['hierarchical_cophenetic_correlation']:.3f}.",
        f"- Normalized 95th-percentile delta-hyperbolicity: {geometry['delta_over_diameter_95pct']:.3f}.",
        "",
        "## Divergence",
        "",
        f"- Finite-time divergence estimate: λ={divergence['finite_time_lambda_per_day']:.5f}/day "
        f"(R²={divergence['fit_r_squared']:.3f}).",
        f"- Implied doubling time: {divergence['doubling_days']}.",
        "",
        "## Periodicity",
        "",
        f"- Strongest period: {periodicity['top_periods'][0]['period_days']:.2f} days.",
        f"- Global peak p-value against year-preserving shuffles: {periodicity['global_peak_surrogate_p']:.4f}.",
        "",
        "## Interpretation protocol",
        "",
        "This report records mathematical results before autobiographical interpretation. "
        "The JSON artifacts retain complete matrices, diagrams, null distributions, dates, "
        "and ranked candidates for evidence retrieval.",
    ]
    return "\n".join(lines) + "\n"


def run_all() -> dict:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    base, views, state = _state_data()
    results = {
        "metadata": {
            "days": len(base), "start": base.iloc[0]["behavioral_day"],
            "end": base.iloc[-1]["behavioral_day"], "random_seed": SETTINGS.random_seed,
        }
    }
    tests = [
        ("delay_embedding", lambda: delay_embedding_tests(base, views)),
        ("symbolic_grammar", lambda: symbolic_grammar_tests(base)),
        ("hinge_echoes", lambda: hinge_echo_tests(base, state)),
        ("conley_morse", lambda: conley_morse_approximation(base)),
        ("causal_emergence", lambda: causal_emergence_tests(base, state)),
        ("bifurcation", lambda: bifurcation_tests(base, state)),
        ("divergence", lambda: divergence_tests(base, state)),
        ("periodicity", lambda: periodicity_tests(base)),
        ("geometry", lambda: geometry_tests(base, state)),
        ("autobiographical_time", graph_time_tests),
        ("sheaf_obstruction", lambda: sheaf_obstruction_approximation(base, views)),
        ("controllability_invariants", lambda: controllability_and_invariants(base, state)),
        ("renormalization", lambda: renormalization_tests(base, state)),
        ("robustness", lambda: robustness_tests(base, state)),
        ("winding", lambda: winding_tests(base, views)),
        ("cross_embedding_robustness", cross_embedding_robustness_tests),
    ]
    for name, function in tests:
        print(f"[topology] running {name}", flush=True)
        results[name] = function()
        _write_json(f"{name}.json", results[name])
    report = build_report(results)
    (RESULT_ROOT / "REPORT.md").write_text(report, encoding="utf-8")
    _write_json("summary.json", results)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", help="Run the complete topology suite")
    args = parser.parse_args()
    if not args.run:
        parser.error("pass --run")
    results = run_all()
    print(json.dumps(results["metadata"], indent=2))


if __name__ == "__main__":
    main()
