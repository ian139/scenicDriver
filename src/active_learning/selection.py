"""Deterministic active-learning selection and geographic split primitives.

The module deliberately keeps all candidate scoring in tabular columns.  A
candidate may contain only image identity and geography, or may additionally
contain model/heuristic scores, uncertainty, cluster IDs, and serialized
embedding/features.  Missing optional signals remain missing; in particular,
uncertainty is never inferred from a prediction score.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .common import atomic_write_json, jsonable, sha256_bytes, sha256_file

SCHEMA_VERSION = 1

IMAGE_COLUMNS = ("image_path", "satellite_path", "image_id", "tile_id", "image_identity")
MODEL_COLUMNS = (
    "model_score",
    "model_prediction",
    "predicted_score",
    "scenic_score_model",
    "scenic_score",
    "prediction",
)
HEURISTIC_COLUMNS = (
    "heuristic_score",
    "scenic_score_heuristic",
    "heuristic_prediction",
    "heuristic",
)
UNCERTAINTY_COLUMNS = (
    "uncertainty",
    "model_uncertainty",
    "predictive_uncertainty",
    "prediction_uncertainty",
    "entropy",
    "score_std",
)
CLUSTER_COLUMNS = ("embedding_cluster", "cluster_id", "cluster", "embedding_cluster_id")
ANNOTATED_COLUMNS = (
    "already_annotated",
    "prior_annotated",
    "annotation_complete",
    "labeled",
    "label_complete",
)

DEFAULT_WEIGHTS: dict[str, float] = {
    "disagreement": 0.22,
    "uncertainty": 0.18,
    "diversity": 0.18,
    "geographic": 0.18,
    "underrepresented": 0.12,
    "score_band": 0.08,
    "random_control": 0.04,
}
REASON_NAMES = {
    "disagreement": "disagreement",
    "uncertainty": "uncertainty",
    "diversity": "diversity",
    "geographic": "geographic_coverage",
    "underrepresented": "underrepresented_stratum",
    "score_band": "score_band",
    "random_control": "random_control",
}


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if bool(pd.isna(value)):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "nat"} else text


def _to_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None:
        return False
    try:
        if bool(pd.isna(value)):
            return False
    except (TypeError, ValueError):
        pass
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "t", "on", "completed"}


def _first_column(frame: pd.DataFrame, names: Sequence[str]) -> pd.Series:
    for name in names:
        if name in frame.columns:
            return frame[name]
    return pd.Series([None] * len(frame), index=frame.index, dtype="object")


def _finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _normalise(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    finite = numeric.replace([np.inf, -np.inf], np.nan)
    result = pd.Series(0.0, index=values.index, dtype="float64")
    valid = finite.notna()
    if not valid.any():
        return result
    low = float(finite.loc[valid].min())
    high = float(finite.loc[valid].max())
    if high <= low:
        result.loc[valid] = 0.5
    else:
        result.loc[valid] = ((finite.loc[valid] - low) / (high - low)).clip(0.0, 1.0)
    return result


def _normalise_nonnegative(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    numeric = numeric.where(np.isfinite(numeric))
    result = pd.Series(0.0, index=values.index, dtype="float64")
    valid = numeric.notna()
    if not valid.any():
        return result
    low = float(numeric.loc[valid].min())
    shifted = (numeric.loc[valid] - low).clip(lower=0.0)
    high = float(shifted.max())
    result.loc[valid] = 0.5 if high == 0.0 else (shifted / high).clip(0.0, 1.0)
    return result


def _stable_int(seed: int, key: str) -> int:
    payload = f"{int(seed)}\0{key}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)


def _stable_key(value: str, seed: int = 0) -> str:
    return f"{_stable_int(seed, value):020d}:{value}"


def _parse_triplet(value: Any) -> tuple[int | None, int | None, int | None]:
    text = _clean_text(value)
    if not text:
        return None, None, None
    patterns = (
        r"(?:^|[/_\-])z?(\d+)[/_\-](\d+)[_\-](\d+)(?:\D|$)",
        r"(?:^|[/_\-])z?(\d+)[/](?:[^/]+/)?(\d+)[_\-](\d+)(?:\D|$)",
        r"(?:^|[/_\-])z?(\d+)[/_\-](\d+)[/_\-](\d+)(?:\D|$)",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return tuple(int(part) for part in match.groups())  # type: ignore[return-value]
    return None, None, None


def _tile_parts(row: pd.Series) -> tuple[int | None, int | None, int | None]:
    values: list[int | None] = []
    for name in ("z", "zoom"):
        if name in row.index:
            values.append(int(float(row[name])) if _finite_number(row[name]) is not None else None)
            break
    else:
        values.append(None)
    for names in (("x", "tile_x"), ("y", "tile_y")):
        for name in names:
            if name in row.index and _finite_number(row[name]) is not None:
                values.append(int(float(row[name])))
                break
        else:
            values.append(None)
    if all(value is not None for value in values):
        return values[0], values[1], values[2]  # type: ignore[return-value]
    for name in ("tile_id", "image_id", "image_identity", "image_path", "satellite_path"):
        if name in row.index:
            parsed = _parse_triplet(row[name])
            if all(value is not None for value in parsed):
                return parsed
    return None, None, None


def _parse_vector(value: Any) -> list[float] | None:
    if isinstance(value, np.ndarray):
        values = value.tolist()
    elif isinstance(value, (list, tuple)):
        values = list(value)
    else:
        text = _clean_text(value)
        if not text:
            return None
        try:
            values = json.loads(text)
        except (TypeError, ValueError, json.JSONDecodeError):
            try:
                values = ast.literal_eval(text)
            except (ValueError, SyntaxError):
                return None
    if not isinstance(values, (list, tuple)):
        return None
    result: list[float] = []
    for item in values:
        number = _finite_number(item)
        if number is None:
            return None
        result.append(number)
    return result or None


def _feature_vector(row: pd.Series) -> list[float] | None:
    for name in ("embedding", "embedding_vector", "features", "feature_vector"):
        if name in row.index:
            vector = _parse_vector(row[name])
            if vector is not None:
                return vector
    feature_names = [
        name
        for name in row.index
        if str(name).lower().startswith(("embedding_", "embed_", "feature_", "feat_"))
        and str(name).lower() not in {"embedding_cluster", "embedding_cluster_id"}
    ]
    values: list[float] = []
    for name in sorted(feature_names):
        number = _finite_number(row[name])
        if number is None:
            return None
        values.append(number)
    return values or None


def _haversine_km(lat_a: float, lon_a: float, lat_b: float, lon_b: float) -> float:
    radians = math.pi / 180.0
    d_lat = (lat_b - lat_a) * radians
    d_lon = (lon_b - lon_a) * radians
    a = math.sin(d_lat / 2) ** 2 + math.cos(lat_a * radians) * math.cos(lat_b * radians) * math.sin(d_lon / 2) ** 2
    return 6371.0088 * 2 * math.asin(min(1.0, math.sqrt(max(0.0, a))))


def _row_lat_lon(row: pd.Series) -> tuple[float | None, float | None]:
    lat = _finite_number(row.get("lat"))
    lon = _finite_number(row.get("lon"))
    return lat, lon


def _tile_key(z: int | None, x: int | None, y: int | None, identity: str) -> str:
    if z is not None and x is not None and y is not None:
        return f"z{z}/{x}/{y}"
    return f"image:{identity}"


def _canonicalise(frame: pd.DataFrame, *, seed: int = 0, deduplicate: bool = True) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("candidate input must be a pandas DataFrame")
    if frame.empty:
        raise ValueError("candidate input has no rows")
    out = frame.copy()
    identity_values: list[str] = []
    image_values: list[str] = []
    satellite_values: list[str] = []
    terrain_values: list[str] = []
    regions: list[str] = []
    classes: list[str] = []
    zs: list[int | None] = []
    xs: list[int | None] = []
    ys: list[int | None] = []
    tile_keys: list[str] = []
    vectors: list[list[float] | None] = []
    for _, row in out.iterrows():
        identity = ""
        for name in ("image_identity", "image_id", "tile_id", "image_path", "satellite_path"):
            if name in row.index:
                identity = _clean_text(row[name])
                if identity:
                    break
        if not identity:
            raise ValueError("candidate rows require image_path, satellite_path, image_id, or tile_id")
        image = _clean_text(row.get("image_path")) or _clean_text(row.get("satellite_path")) or identity
        satellite = _clean_text(row.get("satellite_path")) or image
        terrain = _clean_text(row.get("terrain_path"))
        region = _clean_text(row.get("region")) or "unknown"
        class_id = _clean_text(row.get("class_id")) or "unknown"
        z, x, y = _tile_parts(row)
        identity_values.append(identity)
        image_values.append(image)
        satellite_values.append(satellite)
        terrain_values.append(terrain)
        regions.append(region)
        classes.append(class_id)
        zs.append(z)
        xs.append(x)
        ys.append(y)
        tile_keys.append(_tile_key(z, x, y, identity))
        vectors.append(_feature_vector(row))
    out["image_path"] = image_values
    out["satellite_path"] = satellite_values
    out["terrain_path"] = terrain_values
    out["region"] = regions
    out["class_id"] = classes
    out["z"] = pd.array(zs, dtype="Int64")
    out["x"] = pd.array(xs, dtype="Int64")
    out["y"] = pd.array(ys, dtype="Int64")
    out["_image_identity"] = identity_values
    out["_tile_key"] = tile_keys
    out["_feature_vector"] = vectors
    out["_stable_key"] = [_stable_key(value, seed) for value in tile_keys]
    out = out.sort_values(["_stable_key", "image_path"], kind="mergesort").reset_index(drop=True)
    # A tile is an identity even if two candidate rows point at different style
    # files.  Keeping one row prevents style duplicates from dominating a batch.
    if deduplicate:
        out = out.drop_duplicates(subset=["_tile_key"], keep="first").reset_index(drop=True)
    return out


def _prior_identity_set(prior_annotations: Any) -> set[str]:
    if prior_annotations is None:
        return set()
    if isinstance(prior_annotations, (str, Path)):
        path = Path(prior_annotations)
        if not path.exists():
            raise FileNotFoundError(f"prior annotations not found: {path}")
        prior = load_candidate_table(path)
    elif isinstance(prior_annotations, pd.DataFrame):
        prior = prior_annotations.copy()
    else:
        raise TypeError("prior_annotations must be a path or DataFrame")
    if prior.empty:
        return set()
    if "skip" in prior.columns:
        prior = prior.loc[~prior["skip"].map(_to_bool)]
    if "scenic_human" in prior.columns:
        prior = prior.loc[pd.to_numeric(prior["scenic_human"], errors="coerce").notna()]
    result: set[str] = set()
    for _, row in prior.iterrows():
        for name in ("image_path", "satellite_path", "image_identity", "image_id", "tile_id"):
            if name in row.index:
                value = _clean_text(row[name])
                if value:
                    result.add(value)
                    break
    return result


def _column_numeric(frame: pd.DataFrame, names: Sequence[str]) -> pd.Series:
    return pd.to_numeric(_first_column(frame, names), errors="coerce")


def _prepare_for_selection(
    candidates: pd.DataFrame,
    *,
    seed: int,
    prior_annotations: Any = None,
) -> pd.DataFrame:
    out = _canonicalise(candidates, seed=seed)
    model = _column_numeric(out, MODEL_COLUMNS)
    heuristic = _column_numeric(out, HEURISTIC_COLUMNS)
    uncertainty = _column_numeric(out, UNCERTAINTY_COLUMNS)
    out["_model_raw"] = model
    out["_heuristic_raw"] = heuristic
    out["_uncertainty_raw"] = uncertainty
    model_norm = _normalise(model)
    heuristic_norm = _normalise(heuristic)
    out["score_disagreement"] = (model_norm - heuristic_norm).abs().where(model.notna() & heuristic.notna(), 0.0)
    out["score_uncertainty"] = _normalise_nonnegative(uncertainty).where(uncertainty.notna(), 0.0)
    out["uncertainty_available"] = bool(uncertainty.notna().any())
    out["model_score_available"] = model.notna()
    out["heuristic_score_available"] = heuristic.notna()
    if model.notna().any():
        model_rank = model.rank(method="first", pct=True)
        out["score_band"] = pd.cut(
            model_rank,
            bins=[-np.inf, 1 / 3, 2 / 3, np.inf],
            labels=["low", "middle", "high"],
            include_lowest=True,
        ).astype("object")
        out.loc[model.isna(), "score_band"] = "unknown"
    else:
        out["score_band"] = "unknown"
    band_counts = out.loc[out["score_band"].ne("unknown"), "score_band"].value_counts().to_dict()
    n_scored = sum(int(value) for value in band_counts.values())
    band_targets = {
        band: max(1.0, n_scored * fraction)
        for band, fraction in (("low", 0.25), ("middle", 0.50), ("high", 0.25))
    }
    out["score_band_score"] = out["score_band"].map(
        lambda value: max(0.0, (band_targets.get(str(value), 0.0) - band_counts.get(str(value), 0)) / max(1.0, band_targets.get(str(value), 1.0)))
    ).astype(float)
    cluster = _first_column(out, CLUSTER_COLUMNS).map(_clean_text)
    out["_cluster"] = cluster
    cluster_counts = cluster.where(cluster.ne(""), "unknown").value_counts().to_dict()
    max_cluster = max(cluster_counts.values(), default=1)
    out["score_diversity"] = cluster.where(cluster.ne(""), "unknown").map(
        lambda value: 1.0 - (cluster_counts.get(value or "unknown", 1) - 1) / max(1, max_cluster)
    ).clip(0.0, 1.0).astype(float)
    strata = out["region"].astype(str) + "\0" + out["class_id"].astype(str)
    strata_counts = strata.value_counts().to_dict()
    max_stratum = max(strata_counts.values(), default=1)
    out["score_underrepresented"] = strata.map(
        lambda value: 1.0 - (strata_counts.get(value, 1) - 1) / max(1, max_stratum)
    ).clip(0.0, 1.0).astype(float)
    out["_stratum"] = strata
    out["_random_key"] = [
        _stable_int(seed, f"random-control\0{tile_key}") / float(2**64 - 1)
        for tile_key in out["_tile_key"]
    ]
    prior_ids = _prior_identity_set(prior_annotations)
    annotated_markers = pd.Series(False, index=out.index)
    for name in ANNOTATED_COLUMNS:
        if name in out.columns:
            annotated_markers |= out[name].map(_to_bool)
    annotated_markers |= out["_image_identity"].isin(prior_ids)
    annotated_markers |= out["image_path"].isin(prior_ids)
    annotated_markers |= out["_tile_key"].isin(prior_ids)
    out["_prior_annotated"] = annotated_markers
    out["_prior_identity"] = out["_image_identity"].isin(prior_ids) | out["image_path"].isin(prior_ids) | out["_tile_key"].isin(prior_ids)
    return out


def _vector_distance(a: Sequence[float] | None, b: Sequence[float] | None) -> float | None:
    if a is None or b is None or len(a) != len(b) or not a:
        return None
    return float(math.sqrt(sum((left - right) ** 2 for left, right in zip(a, b))))


def _blocked(row: pd.Series, selected: Iterable[pd.Series], *, adjacency_radius: int, min_separation_km: float) -> bool:
    z, x, y = row.get("z"), row.get("x"), row.get("y")
    z = int(z) if _finite_number(z) is not None else None
    x = int(x) if _finite_number(x) is not None else None
    y = int(y) if _finite_number(y) is not None else None
    lat, lon = _row_lat_lon(row)
    for previous in selected:
        if row.get("_tile_key") == previous.get("_tile_key"):
            return True
        pz, px, py = previous.get("z"), previous.get("x"), previous.get("y")
        pz = int(pz) if _finite_number(pz) is not None else None
        px = int(px) if _finite_number(px) is not None else None
        py = int(py) if _finite_number(py) is not None else None
        if None not in (z, x, y, pz, px, py) and z == pz:
            if max(abs(x - px), abs(y - py)) <= max(0, int(adjacency_radius)):
                return True
        if min_separation_km > 0.0 and lat is not None and lon is not None:
            p_lat, p_lon = _row_lat_lon(previous)
            if p_lat is not None and p_lon is not None and _haversine_km(lat, lon, p_lat, p_lon) < min_separation_km:
                return True
    return False


def _dynamic_components(row: pd.Series, selected: list[pd.Series], selected_regions: set[str], selected_clusters: set[str], target_bands: Mapping[str, float]) -> dict[str, float]:
    components = {
        "disagreement": float(row.get("score_disagreement", 0.0) or 0.0),
        "uncertainty": float(row.get("score_uncertainty", 0.0) or 0.0),
        "underrepresented": float(row.get("score_underrepresented", 0.0) or 0.0),
        "score_band": float(row.get("score_band_score", 0.0) or 0.0),
        "diversity": float(row.get("score_diversity", 0.0) or 0.0),
        "geographic": 0.0,
        "random_control": 0.0,
    }
    cluster = _clean_text(row.get("_cluster")) or "unknown"
    if cluster not in selected_clusters:
        components["diversity"] = max(components["diversity"], 1.0)
    elif selected:
        components["diversity"] = min(components["diversity"], 0.25)
    vector = row.get("_feature_vector")
    if selected and vector is not None:
        distances = [_vector_distance(vector, previous.get("_feature_vector")) for previous in selected]
        distances = [distance for distance in distances if distance is not None]
        if distances:
            # A scale of one is intentionally conservative: feature vectors are
            # often already normalized, while raw distances remain auditable.
            components["diversity"] = max(components["diversity"], min(1.0, min(distances)))
    region = _clean_text(row.get("region")) or "unknown"
    geographic = 1.0 if region not in selected_regions else 0.0
    lat, lon = _row_lat_lon(row)
    if selected and lat is not None and lon is not None:
        distances: list[float] = []
        for previous in selected:
            p_lat, p_lon = _row_lat_lon(previous)
            if p_lat is not None and p_lon is not None:
                distances.append(_haversine_km(lat, lon, p_lat, p_lon))
        if distances:
            geographic = 0.6 * geographic + 0.4 * min(1.0, min(distances) / 25.0)
    components["geographic"] = geographic
    band = _clean_text(row.get("score_band"))
    if band in target_bands:
        components["score_band"] = max(
            components["score_band"],
            max(0.0, target_bands[band] - sum(1 for item in selected if _clean_text(item.get("score_band")) == band)) / max(1.0, target_bands[band]),
        )
    return {name: float(max(0.0, min(1.0, value))) for name, value in components.items()}


def _weighted_score(components: Mapping[str, float], weights: Mapping[str, float]) -> float:
    return float(sum(float(weights.get(name, 0.0)) * float(value) for name, value in components.items()))


def _reason(components: Mapping[str, float]) -> str:
    order = ("disagreement", "uncertainty", "diversity", "geographic", "underrepresented", "score_band", "random_control")
    best = max(order, key=lambda name: (float(components.get(name, 0.0)), -order.index(name)))
    return REASON_NAMES[best]


def _normalise_weights(weights: Mapping[str, float] | None) -> dict[str, float]:
    result = dict(DEFAULT_WEIGHTS)
    if weights:
        aliases = {"geographic_coverage": "geographic", "underrepresented_strata": "underrepresented", "band": "score_band", "random": "random_control"}
        for key, value in weights.items():
            canonical = aliases.get(str(key), str(key))
            if canonical not in result:
                raise ValueError(f"unknown selection weight: {key}")
            number = _finite_number(value)
            if number is None or number < 0:
                raise ValueError(f"selection weight must be finite and non-negative: {key}")
            result[canonical] = float(number)
    if sum(result.values()) <= 0:
        raise ValueError("at least one selection weight must be positive")
    return result


@dataclass(frozen=True)
class SelectionConfig:
    batch_size: int = 100
    seed: int = 0
    run_name: str = "active_learning"
    adjacency_radius: int = 1
    min_separation_km: float = 0.0
    qa_overlap_count: int = 0
    qa_overlap_fraction: float = 0.0
    random_control_count: int = 0
    random_control_fraction: float = 0.0
    weights: Mapping[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))


@dataclass
class SelectionArtifacts:
    selected: pd.DataFrame
    normalized_candidates: pd.DataFrame
    diagnostics: dict[str, Any]
    batch_manifest: dict[str, Any]
    geographic_splits: pd.DataFrame
    leakage_audit: dict[str, Any]


def select_batch(
    candidates: pd.DataFrame,
    *,
    config: SelectionConfig | None = None,
    prior_annotations: Any = None,
) -> SelectionArtifacts:
    """Select one deterministic batch and produce split/audit data in memory."""

    config = config or SelectionConfig()
    if int(config.batch_size) < 0:
        raise ValueError("batch_size must be non-negative")
    if int(config.adjacency_radius) < 0:
        raise ValueError("adjacency_radius must be non-negative")
    if float(config.min_separation_km) < 0:
        raise ValueError("min_separation_km must be non-negative")
    weights = _normalise_weights(config.weights)
    prepared = _prepare_for_selection(candidates, seed=int(config.seed), prior_annotations=prior_annotations)
    eligible = prepared.loc[~prepared["_prior_annotated"]].copy()
    overlap_pool = prepared.loc[prepared["_prior_annotated"]].copy()
    scored_count = int(prepared["model_score_available"].sum())
    uncertainty_count = int(prepared["_uncertainty_raw"].notna().sum())
    batch_size = min(int(config.batch_size), len(prepared))
    requested_overlap = int(config.qa_overlap_count)
    if requested_overlap <= 0 and float(config.qa_overlap_fraction) > 0:
        requested_overlap = int(round(batch_size * float(config.qa_overlap_fraction)))
    requested_overlap = max(0, min(batch_size, requested_overlap))
    requested_controls = int(config.random_control_count)
    if requested_controls <= 0 and float(config.random_control_fraction) > 0:
        requested_controls = int(round(batch_size * float(config.random_control_fraction)))
    requested_controls = max(0, min(batch_size - requested_overlap, requested_controls))
    regular_capacity = max(0, batch_size - requested_overlap)
    selected_indices: list[int] = []
    records: dict[int, dict[str, Any]] = {}
    selected_rows: list[pd.Series] = []
    selected_regions: set[str] = set()
    selected_clusters: set[str] = set()
    band_targets = {"low": max(1.0, regular_capacity * 0.25), "middle": max(1.0, regular_capacity * 0.50), "high": max(1.0, regular_capacity * 0.25)}

    def add(index: int, components: Mapping[str, float], *, reason: str, control: bool = False, overlap: bool = False) -> None:
        row = prepared.loc[index]
        selected_indices.append(index)
        selected_rows.append(row)
        selected_regions.add(_clean_text(row.get("region")) or "unknown")
        selected_clusters.add(_clean_text(row.get("_cluster")) or "unknown")
        payload = dict(components)
        payload["random_control"] = float(row.get("_random_key", 0.0)) if control else 0.0
        records[index] = {
            "components": payload,
            "selection_score": _weighted_score(payload, weights),
            "selection_reason": reason,
            "is_random_control": bool(control),
            "is_qa_overlap": bool(overlap),
        }

    # Controls are deterministic random draws, but remain subject to spatial
    # exclusion so the control cannot itself create a duplicate-heavy batch.
    if requested_controls:
        control_order = sorted(eligible.index.tolist(), key=lambda index: (float(prepared.loc[index, "_random_key"]), prepared.loc[index, "_stable_key"]))
        for index in control_order:
            if len([item for item in selected_indices if not records[item]["is_qa_overlap"]]) >= min(regular_capacity, requested_controls):
                break
            row = prepared.loc[index]
            if _blocked(row, selected_rows, adjacency_radius=int(config.adjacency_radius), min_separation_km=float(config.min_separation_km)):
                continue
            components = _dynamic_components(row, selected_rows, selected_regions, selected_clusters, band_targets)
            add(index, components, reason="random_control", control=True)

    while len([item for item in selected_indices if not records[item]["is_qa_overlap"]]) < regular_capacity:
        available = [index for index in eligible.index.tolist() if index not in records]
        ranked: list[tuple[float, str, int, dict[str, float]]] = []
        for index in available:
            row = prepared.loc[index]
            if _blocked(row, selected_rows, adjacency_radius=int(config.adjacency_radius), min_separation_km=float(config.min_separation_km)):
                continue
            components = _dynamic_components(row, selected_rows, selected_regions, selected_clusters, band_targets)
            ranked.append((_weighted_score(components, weights), str(row["_stable_key"]), index, components))
        if not ranked:
            break
        ranked.sort(key=lambda item: (-item[0], item[1]))
        _, _, index, components = ranked[0]
        add(index, components, reason=_reason(components))

    # Repeat/overlap rows are intentionally exempt from the completed-label
    # filter, but are still kept spatially separate from one another and from
    # newly selected rows.
    if requested_overlap:
        overlap_order = sorted(overlap_pool.index.tolist(), key=lambda index: (prepared.loc[index, "_stable_key"], prepared.loc[index, "_image_identity"]))
        for index in overlap_order:
            if len(selected_indices) >= batch_size:
                break
            row = prepared.loc[index]
            if _blocked(row, selected_rows, adjacency_radius=int(config.adjacency_radius), min_separation_km=float(config.min_separation_km)):
                continue
            components = _dynamic_components(row, selected_rows, selected_regions, selected_clusters, band_targets)
            add(index, components, reason="qa_overlap", overlap=True)

    # If no explicit prior file was supplied, a candidate may carry a prior
    # marker.  It is useful as an overlap source only when requested.
    if requested_overlap and len(selected_indices) < batch_size:
        remaining = [index for index in prepared.index.tolist() if index not in records and bool(prepared.loc[index, "_prior_annotated"])]
        for index in sorted(remaining, key=lambda item: str(prepared.loc[item, "_stable_key"])):
            if len(selected_indices) >= batch_size:
                break
            row = prepared.loc[index]
            if _blocked(row, selected_rows, adjacency_radius=int(config.adjacency_radius), min_separation_km=float(config.min_separation_km)):
                continue
            components = _dynamic_components(row, selected_rows, selected_regions, selected_clusters, band_targets)
            add(index, components, reason="qa_overlap", overlap=True)

    output = prepared.loc[selected_indices].copy() if selected_indices else prepared.iloc[0:0].copy()
    output = output.reset_index(drop=True)
    output_columns_to_drop = [column for column in output.columns if column.startswith("_")]
    if output_columns_to_drop:
        output = output.drop(columns=output_columns_to_drop)
    for column in ("selection_reason", "selection_score", "selection_rank", "rank", "batch_id", "run_id", "is_random_control", "is_qa_overlap", "uncertainty_observed", "disagreement_score", "uncertainty_score", "diversity_score", "geographic_score", "underrepresented_score", "score_band_component", "random_control_score"):
        if column in output.columns:
            output = output.drop(columns=[column])
    batch_id_seed = "|".join(str(value) for value in prepared["_tile_key"].tolist())
    batch_id = f"batch-{sha256_bytes(f'{config.run_name}|{config.seed}|{batch_id_seed}'.encode('utf-8'))[:16]}"
    output["selection_reason"] = [records[index]["selection_reason"] for index in selected_indices]
    output["selection_score"] = [round(float(records[index]["selection_score"]), 12) for index in selected_indices]
    output["selection_rank"] = list(range(1, len(output) + 1))
    output["rank"] = output["selection_rank"]
    output["batch_id"] = batch_id
    output["run_id"] = str(config.run_name)
    output["is_random_control"] = [records[index]["is_random_control"] for index in selected_indices]
    output["is_qa_overlap"] = [records[index]["is_qa_overlap"] for index in selected_indices]
    output["uncertainty_observed"] = [bool(prepared.loc[index, "_uncertainty_raw"] == prepared.loc[index, "_uncertainty_raw"]) for index in selected_indices]
    component_columns = {
        "disagreement": "disagreement_score",
        "uncertainty": "uncertainty_score",
        "diversity": "diversity_score",
        "geographic": "geographic_score",
        "underrepresented": "underrepresented_score",
        "score_band": "score_band_component",
        "random_control": "random_control_score",
    }
    for component, column in component_columns.items():
        output[column] = [round(float(records[index]["components"].get(component, 0.0)), 12) for index in selected_indices]
        output[f"component_{component}"] = output[column]
    # Keep the output easy for the web annotator to consume while retaining the
    # original candidate columns as auxiliary metadata.
    preferred = [
        "image_path", "satellite_path", "terrain_path", "image_id", "tile_id",
        "region", "class_id", "z", "x", "y", "lat", "lon", "score_band",
        "selection_reason", "selection_score", "selection_rank", "rank", "batch_id", "run_id",
        "disagreement_score", "uncertainty_score", "diversity_score", "geographic_score",
        "underrepresented_score", "score_band_component", "random_control_score",
        "component_disagreement", "component_uncertainty", "component_diversity",
        "component_geographic", "component_underrepresented", "component_score_band",
        "component_random_control", "uncertainty_observed", "is_random_control", "is_qa_overlap",
    ]
    ordered = [column for column in preferred if column in output.columns]
    ordered.extend(column for column in output.columns if column not in ordered)
    output = output[ordered]
    output.attrs["batch_id"] = batch_id
    output.attrs["run_id"] = str(config.run_name)

    split_source = prepared.drop(columns=[column for column in prepared.columns if column.startswith("_")], errors="ignore")
    splits = build_geographic_splits(split_source, seed=int(config.seed), adjacency_radius=int(config.adjacency_radius))
    audit = audit_geographic_leakage(splits, adjacency_radius=int(config.adjacency_radius))
    reason_counts = output["selection_reason"].value_counts().sort_index().to_dict() if not output.empty else {}
    diagnostics: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_name": str(config.run_name),
        "batch_id": batch_id,
        "seed": int(config.seed),
        "candidate_rows": int(len(prepared)),
        "eligible_rows": int(len(eligible)),
        "selected_rows": int(len(output)),
        "scored_model_rows": scored_count,
        "uncertainty_rows_observed": uncertainty_count,
        "uncertainty_available": bool(uncertainty_count > 0),
        "prior_annotated_rows": int(prepared["_prior_annotated"].sum()),
        "requested_overlap_rows": requested_overlap,
        "selected_overlap_rows": int(output["is_qa_overlap"].sum()) if not output.empty else 0,
        "requested_random_controls": requested_controls,
        "selected_random_controls": int(output["is_random_control"].sum()) if not output.empty else 0,
        "adjacency_radius": int(config.adjacency_radius),
        "minimum_separation_km": float(config.min_separation_km),
        "weights": {key: float(value) for key, value in weights.items()},
        "stable_ordering": "selection_rank ascending; score ties use deterministic SHA-256 identity key",
        "component_availability": {
            "disagreement": bool(prepared["model_score_available"].any() and prepared["heuristic_score_available"].any()),
            "uncertainty": bool(uncertainty_count > 0),
            "embedding_features": bool(prepared["_feature_vector"].notna().any()),
            "embedding_clusters": bool(prepared["_cluster"].ne("").any()),
            "geography": bool(prepared["region"].ne("unknown").any() or prepared[["lat", "lon"]].notna().all(axis=1).any()),
        },
        "reason_distribution": {str(key): int(value) for key, value in reason_counts.items()},
        "score_band_distribution": {str(key): int(value) for key, value in (output["score_band"].value_counts().sort_index().to_dict().items() if not output.empty and "score_band" in output.columns else [])},
        "region_distribution": {str(key): int(value) for key, value in (output["region"].value_counts().sort_index().to_dict().items() if not output.empty else [])},
        "cluster_distribution": {str(key): int(value) for key, value in (output.get("embedding_cluster", output.get("cluster_id", pd.Series(dtype=object))).value_counts().sort_index().to_dict().items() if not output.empty else [])},
        "geographic_coverage": {
            "unique_regions": sorted(str(value) for value in output["region"].dropna().unique()) if not output.empty and "region" in output.columns else [],
            "lat_min": _finite_number(output["lat"].min()) if not output.empty and "lat" in output.columns and output["lat"].notna().any() else None,
            "lat_max": _finite_number(output["lat"].max()) if not output.empty and "lat" in output.columns and output["lat"].notna().any() else None,
            "lon_min": _finite_number(output["lon"].min()) if not output.empty and "lon" in output.columns and output["lon"].notna().any() else None,
            "lon_max": _finite_number(output["lon"].max()) if not output.empty and "lon" in output.columns and output["lon"].notna().any() else None,
        },
        "geographic_split_counts": {str(key): int(value) for key, value in splits["split"].value_counts().sort_index().to_dict().items()},
        "leakage_audit_valid": bool(audit["valid"]),
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_id": str(config.run_name),
        "run_name": str(config.run_name),
        "batch_id": batch_id,
        "mode": "active_learning_selection",
        "seed": int(config.seed),
        "row_count": int(len(output)),
        "ordering": "selection_rank ascending; canonical SHA-256 identity tie-break",
        "input_columns": [str(column) for column in candidates.columns],
        "output_columns": [str(column) for column in output.columns],
        "selection_config": {
            "batch_size": int(config.batch_size),
            "adjacency_radius": int(config.adjacency_radius),
            "minimum_separation_km": float(config.min_separation_km),
            "qa_overlap_count": int(config.qa_overlap_count),
            "qa_overlap_fraction": float(config.qa_overlap_fraction),
            "random_control_count": int(config.random_control_count),
            "random_control_fraction": float(config.random_control_fraction),
            "weights": {key: float(value) for key, value in weights.items()},
        },
        "uncertainty_definition": "normalized provided uncertainty column only; absent values remain unavailable",
    }
    return SelectionArtifacts(
        selected=output,
        normalized_candidates=split_source,
        diagnostics=jsonable(diagnostics),
        batch_manifest=jsonable(manifest),
        geographic_splits=splits,
        leakage_audit=jsonable(audit),
    )


def select_candidates(
    candidates: pd.DataFrame,
    *,
    batch_size: int,
    seed: int = 0,
    run_name: str = "active_learning",
    prior_annotations: Any = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Convenience API returning only the selected rows.

    Diagnostics and the manifest are available through ``DataFrame.attrs`` and
    through :func:`select_batch` when callers need the complete artifact set.
    """

    config = SelectionConfig(batch_size=batch_size, seed=seed, run_name=run_name, **kwargs)
    artifacts = select_batch(candidates, config=config, prior_annotations=prior_annotations)
    artifacts.selected.attrs["diagnostics"] = artifacts.diagnostics
    artifacts.selected.attrs["batch_manifest"] = artifacts.batch_manifest
    artifacts.selected.attrs["geographic_splits"] = artifacts.geographic_splits
    artifacts.selected.attrs["leakage_audit"] = artifacts.leakage_audit
    return artifacts.selected


select_active_learning = select_candidates


def _tile_block(row: pd.Series, block_size: int, block_degrees: float) -> str:
    region = _clean_text(row.get("region")) or "unknown"
    z = _finite_number(row.get("z"))
    x = _finite_number(row.get("x"))
    y = _finite_number(row.get("y"))
    if z is not None and x is not None and y is not None:
        return f"{region}|z{int(z)}|b{int(math.floor(x / max(1, block_size)))}|{int(math.floor(y / max(1, block_size)))}"
    lat, lon = _row_lat_lon(row)
    if lat is not None and lon is not None:
        return f"{region}|g{int(math.floor(lat / block_degrees))}|{int(math.floor(lon / block_degrees))}"
    return f"{region}|identity|{_clean_text(row.get('_tile_key')) or _clean_text(row.get('image_path'))}"


def _union_find(size: int) -> tuple[list[int], Any, Any]:
    parent = list(range(size))
    rank = [0] * size

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: int, right: int) -> None:
        root_left, root_right = find(left), find(right)
        if root_left == root_right:
            return
        if rank[root_left] < rank[root_right]:
            root_left, root_right = root_right, root_left
        parent[root_right] = root_left
        if rank[root_left] == rank[root_right]:
            rank[root_left] += 1

    return parent, find, union


def _adjacency_groups(frame: pd.DataFrame, *, adjacency_radius: int, distance_km: float = 0.0) -> dict[int, list[int]]:
    parent, find, union = _union_find(len(frame))
    tile_lookup: dict[tuple[int, int, int], int] = {}
    identity_lookup: dict[str, int] = {}
    for index, row in frame.reset_index(drop=True).iterrows():
        key = _clean_text(row.get("_tile_key")) or _clean_text(row.get("image_path"))
        if key and key in identity_lookup:
            union(index, identity_lookup[key])
        elif key:
            identity_lookup[key] = index
        z, x, y = row.get("z"), row.get("x"), row.get("y")
        if None not in (z, x, y) and all(_finite_number(value) is not None for value in (z, x, y)):
            zxy = (int(z), int(x), int(y))
            tile_lookup[zxy] = index
            radius = max(0, int(adjacency_radius))
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    other = tile_lookup.get((zxy[0], zxy[1] + dx, zxy[2] + dy))
                    if other is not None:
                        union(index, other)
    if distance_km > 0.0:
        geo_buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
        # A coarse bucket keeps the fallback quadratic check bounded while
        # still checking every point in neighboring buckets.
        degrees = max(0.01, distance_km / 111.0)
        reset = frame.reset_index(drop=True)
        for index, row in reset.iterrows():
            lat, lon = _row_lat_lon(row)
            if lat is None or lon is None:
                continue
            bucket = (int(math.floor(lat / degrees)), int(math.floor(lon / degrees)))
            for bx in range(bucket[0] - 1, bucket[0] + 2):
                for by in range(bucket[1] - 1, bucket[1] + 2):
                    for other in geo_buckets.get((bx, by), []):
                        p_lat, p_lon = _row_lat_lon(reset.loc[other])
                        if p_lat is not None and p_lon is not None and _haversine_km(lat, lon, p_lat, p_lon) < distance_km:
                            union(index, other)
            geo_buckets[bucket].append(index)
    groups: dict[int, list[int]] = defaultdict(list)
    for index in range(len(frame)):
        groups[find(index)].append(index)
    return dict(groups)


def build_geographic_splits(
    frame: pd.DataFrame,
    *,
    seed: int = 0,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    block_size: int = 8,
    block_degrees: float = 0.25,
    adjacency_radius: int = 1,
    adjacency_distance_km: float = 0.0,
) -> pd.DataFrame:
    """Assign deterministic geographic components to train/val/test.

    Duplicate or adjacent tile components are indivisible units.  This makes
    the generated assignment safe by construction; :func:`audit_geographic_leakage`
    remains an independent check for externally supplied split files.
    """

    if not 0 <= val_fraction < 1 or not 0 <= test_fraction < 1 or val_fraction + test_fraction >= 1:
        raise ValueError("val_fraction and test_fraction must be non-negative and sum to less than one")
    if block_size <= 0 or block_degrees <= 0:
        raise ValueError("block_size and block_degrees must be positive")
    prepared = _canonicalise(frame, seed=seed) if not {"_tile_key", "_stable_key"}.issubset(frame.columns) else frame.copy()
    prepared = prepared.reset_index(drop=True)
    if prepared.empty:
        output = prepared.copy()
        output["geographic_block"] = pd.Series(dtype="object")
        output["split"] = pd.Series(dtype="object")
        return output
    prepared["geographic_block"] = [
        _tile_block(row, block_size=block_size, block_degrees=block_degrees) for _, row in prepared.iterrows()
    ]
    groups = _adjacency_groups(prepared, adjacency_radius=adjacency_radius, distance_km=adjacency_distance_km)
    component_records: list[dict[str, Any]] = []
    for root, indices in groups.items():
        block = min(str(prepared.loc[index, "geographic_block"]) for index in indices)
        stable = min(str(prepared.loc[index, "_stable_key"]) for index in indices) if "_stable_key" in prepared.columns else min(str(prepared.loc[index, "image_path"]) for index in indices)
        component_records.append({"root": root, "indices": indices, "block": block, "stable": stable, "size": len(indices)})
    component_records.sort(key=lambda item: (item["block"], item["stable"]))
    n_rows = len(prepared)
    targets = {
        "train": max(1, int(round(n_rows * (1.0 - val_fraction - test_fraction)))) if n_rows else 0,
        "val": int(round(n_rows * val_fraction)),
        "test": int(round(n_rows * test_fraction)),
    }
    if len(component_records) >= 2 and val_fraction > 0:
        targets["val"] = max(1, targets["val"])
    if len(component_records) >= 3 and test_fraction > 0:
        targets["test"] = max(1, targets["test"])
    split_order = ("train", "val", "test")
    assigned = {name: 0 for name in split_order}
    component_split: dict[int, str] = {}
    for component in component_records:
        choices = sorted(
            split_order,
            key=lambda name: (
                assigned[name] / max(1, targets[name]),
                assigned[name],
                _stable_int(seed, f"split\0{name}\0{component['block']}"),
                split_order.index(name),
            ),
        )
        chosen = choices[0]
        component_split[int(component["root"])] = chosen
        assigned[chosen] += int(component["size"])
    prepared["split"] = [component_split[next(root for root, indices in groups.items() if index in indices)] for index in range(len(prepared))]
    prepared["split_seed"] = int(seed)
    prepared = prepared.sort_values(["split", "geographic_block", "_stable_key"], kind="mergesort").reset_index(drop=True)
    drop_private = [column for column in prepared.columns if column.startswith("_")]
    if drop_private:
        prepared = prepared.drop(columns=drop_private)
    preferred = ["image_path", "satellite_path", "terrain_path", "image_id", "tile_id", "region", "class_id", "z", "x", "y", "lat", "lon", "geographic_block", "split", "split_seed"]
    ordered = [column for column in preferred if column in prepared.columns]
    ordered.extend(column for column in prepared.columns if column not in ordered)
    return prepared[ordered]


def _split_values(value: Any) -> set[str]:
    if value is None:
        return set()
    text = _clean_text(value)
    return {text} if text else set()


def audit_geographic_leakage(
    frame: pd.DataFrame,
    *,
    adjacency_radius: int = 1,
    adjacency_distance_km: float = 0.0,
) -> dict[str, Any]:
    """Find duplicate or adjacent tiles assigned to different splits."""

    if "split" not in frame.columns:
        return {
            "schema_version": SCHEMA_VERSION,
            "valid": False,
            "checked_rows": int(len(frame)),
            "duplicate_cross_split": False,
            "adjacent_cross_split": False,
            "violations": [{"kind": "schema", "message": "split column is required"}],
            "split_counts": {},
        }
    prepared = _canonicalise(frame, seed=0, deduplicate=False) if not {"_tile_key"}.issubset(frame.columns) else frame.reset_index(drop=True).copy()
    # Canonicalisation may discard split if it was not carried as an original
    # column; restore it from the source frame by positional order.
    if "split" not in prepared.columns:
        prepared["split"] = frame.reset_index(drop=True)["split"].astype(str).tolist()
    violations: list[dict[str, Any]] = []
    duplicate_cross = False
    adjacent_cross = False
    identity_splits: dict[str, set[str]] = defaultdict(set)
    identity_rows: dict[str, list[int]] = defaultdict(list)
    for index, row in prepared.iterrows():
        key = _clean_text(row.get("_tile_key")) or _clean_text(row.get("image_path"))
        split = _clean_text(row.get("split"))
        if key:
            identity_splits[key].add(split)
            identity_rows[key].append(int(index))
    for key in sorted(identity_splits):
        splits = sorted(value for value in identity_splits[key] if value)
        if len(splits) > 1:
            duplicate_cross = True
            violations.append({"kind": "duplicate", "tile_key": key, "splits": splits, "rows": identity_rows[key]})

    tile_lookup: dict[tuple[int, int, int], int] = {}
    for index, row in prepared.iterrows():
        z, x, y = row.get("z"), row.get("x"), row.get("y")
        if None in (z, x, y) or not all(_finite_number(value) is not None for value in (z, x, y)):
            continue
        tile_lookup[(int(z), int(x), int(y))] = int(index)
    seen_pairs: set[tuple[int, int]] = set()
    radius = max(0, int(adjacency_radius))
    for (z, x, y), index in sorted(tile_lookup.items()):
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                other = tile_lookup.get((z, x + dx, y + dy))
                if other is None or other == index:
                    continue
                pair = tuple(sorted((index, other)))
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                split_a, split_b = _clean_text(prepared.loc[index, "split"]), _clean_text(prepared.loc[other, "split"])
                if split_a and split_b and split_a != split_b:
                    adjacent_cross = True
                    violations.append({"kind": "adjacent", "tile_keys": [f"z{z}/{x}/{y}", f"z{z}/{int(prepared.loc[other, 'x'])}/{int(prepared.loc[other, 'y'])}"], "splits": sorted((split_a, split_b)), "rows": [index, other]})
    if adjacency_distance_km > 0.0:
        geo_rows = [(int(index), *_row_lat_lon(row), _clean_text(row.get("split"))) for index, row in prepared.iterrows()]
        for left_position, left in enumerate(geo_rows):
            left_index, left_lat, left_lon, left_split = left
            if left_lat is None or left_lon is None or not left_split:
                continue
            for right in geo_rows[left_position + 1 :]:
                right_index, right_lat, right_lon, right_split = right
                if right_lat is None or right_lon is None or not right_split or left_split == right_split:
                    continue
                if _haversine_km(left_lat, left_lon, right_lat, right_lon) < adjacency_distance_km:
                    adjacent_cross = True
                    violations.append({"kind": "adjacent_geographic", "distance_km": _haversine_km(left_lat, left_lon, right_lat, right_lon), "splits": sorted((left_split, right_split)), "rows": [left_index, right_index]})
    violations.sort(key=lambda item: (str(item.get("kind")), json.dumps(item, sort_keys=True)))
    return {
        "schema_version": SCHEMA_VERSION,
        "valid": not violations,
        "checked_rows": int(len(prepared)),
        "duplicate_cross_split": duplicate_cross,
        "adjacent_cross_split": adjacent_cross,
        "violation_count": int(len(violations)),
        "violations": violations,
        "split_counts": {str(key): int(value) for key, value in frame["split"].astype(str).value_counts().sort_index().to_dict().items()},
    }


def load_candidate_table(path: str | Path) -> pd.DataFrame:
    """Read CSV or Parquet candidate data with an explicit extension check."""

    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"candidate input not found: {source}")
    suffix = source.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(source)
    if suffix in {".csv", ".tsv", ".txt"}:
        return pd.read_csv(source, sep="\t" if suffix == ".tsv" else ",")
    raise ValueError(f"candidate input must be CSV or Parquet: {source}")


def run_selection(
    candidate_input: str | Path,
    *,
    output_dir: str | Path = "data/processed/active_learning",
    run_name: str = "active_learning",
    config: SelectionConfig | None = None,
    prior_annotations: Any = None,
) -> SelectionArtifacts:
    """Run selection and persist the contracted batch artifacts."""

    source = Path(candidate_input)
    candidates = load_candidate_table(source)
    config = config or SelectionConfig(run_name=run_name)
    if config.run_name != run_name:
        config = SelectionConfig(
            batch_size=config.batch_size,
            seed=config.seed,
            run_name=run_name,
            adjacency_radius=config.adjacency_radius,
            min_separation_km=config.min_separation_km,
            qa_overlap_count=config.qa_overlap_count,
            qa_overlap_fraction=config.qa_overlap_fraction,
            random_control_count=config.random_control_count,
            random_control_fraction=config.random_control_fraction,
            weights=config.weights,
        )
    artifacts = select_batch(candidates, config=config, prior_annotations=prior_annotations)
    base = Path(output_dir) / run_name
    base.mkdir(parents=True, exist_ok=True)
    batch_path = base / "annotation_batch.csv"
    split_path = base / "geographic_splits.csv"
    diagnostics_path = base / "selection_diagnostics.json"
    manifest_path = base / "batch_manifest.json"
    leakage_path = base / "leakage_audit.json"
    artifacts.selected.to_csv(batch_path, index=False, lineterminator="\n")
    artifacts.geographic_splits.to_csv(split_path, index=False, lineterminator="\n")
    input_digest = sha256_file(source)
    artifacts.batch_manifest["candidate_input"] = {"path": str(source), "sha256": input_digest, "rows": int(len(candidates))}
    artifacts.batch_manifest["outputs"] = {
        "annotation_batch_csv": str(batch_path),
        "batch_manifest_json": str(manifest_path),
        "selection_diagnostics_json": str(diagnostics_path),
        "geographic_splits_csv": str(split_path),
        "leakage_audit_json": str(leakage_path),
    }
    artifacts.diagnostics["candidate_input"] = {"path": str(source), "sha256": input_digest}
    artifacts.diagnostics["outputs"] = artifacts.batch_manifest["outputs"]
    atomic_write_json(manifest_path, artifacts.batch_manifest)
    atomic_write_json(diagnostics_path, artifacts.diagnostics)
    atomic_write_json(leakage_path, artifacts.leakage_audit)
    return artifacts
