"""Precompute UMAP coordinates and DBSCAN clusters for the cluster map UI."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import umap

from .attr_index import ATTR_KEYS, AttributeIndexer

CLUSTER_ALLOWED_MODES = ("location", "subject", "image")


@dataclass(slots=True)
class ClusterJobConfig:
    """Configuration options for a single cluster precomputation run."""

    n_neighbors: int = 30
    min_dist: float = 0.15
    metric: str = "cosine"
    random_state: int = 42
    min_samples: int = 10
    eps: float | None = None
    eps_percentile: float = 90.0
    limit: int | None = None


@dataclass(slots=True)
class ClusterArtifacts:
    """Artifacts produced by a cluster precomputation run."""

    coords: np.ndarray
    image_ids: np.ndarray
    labels: np.ndarray
    hulls: list[dict[str, object]]
    meta: dict[str, object]


def precompute_cluster(indexer: AttributeIndexer, mode: str, config: ClusterJobConfig) -> ClusterArtifacts:
    """Run UMAP + DBSCAN clustering for a specific mode."""

    mode = mode.lower()
    if mode not in CLUSTER_ALLOWED_MODES:
        raise ValueError(f"Unsupported cluster mode '{mode}'. Expected one of {CLUSTER_ALLOWED_MODES}.")

    if mode in ATTR_KEYS:
        image_ids, embeddings = indexer.load_attr_embeddings(mode)
    else:
        image_ids, embeddings = indexer.load_image_embeddings()

    if image_ids.size == 0:
        raise ValueError(f"No embeddings stored for mode '{mode}'. 作成済みインデックスを確認してください。")

    order = np.argsort(image_ids)
    image_ids = image_ids[order]
    embeddings = embeddings[order]

    if config.limit is not None and config.limit > 0:
        image_ids = image_ids[: config.limit]
        embeddings = embeddings[: config.limit]

    reducer = umap.UMAP(
        n_neighbors=config.n_neighbors,
        min_dist=config.min_dist,
        metric=config.metric,
        random_state=config.random_state,
    )
    coords = reducer.fit_transform(embeddings.astype("float32"))

    eps_value = config.eps
    if eps_value is None:
        eps_value = _estimate_eps(coords, config.min_samples, percentile=config.eps_percentile)

    clustering = DBSCAN(eps=eps_value, min_samples=config.min_samples)
    clustering.fit(coords)
    labels = clustering.labels_.astype("int32")

    hulls = _build_hulls(coords, labels)
    meta = _build_meta(
        mode=mode,
        coords=coords,
        labels=labels,
        config=config,
        eps=eps_value,
    )

    return ClusterArtifacts(
        coords=coords.astype("float32"),
        image_ids=image_ids.astype("int64"),
        labels=labels,
        hulls=hulls,
        meta=meta,
    )


def save_artifacts(target_dir: Path, artifacts: ClusterArtifacts) -> None:
    """Persist cluster artifacts to disk."""

    target_dir.mkdir(parents=True, exist_ok=True)
    np.save(target_dir / "coords.npy", artifacts.coords)
    np.save(target_dir / "image_ids.npy", artifacts.image_ids)
    np.save(target_dir / "labels.npy", artifacts.labels)

    meta_path = target_dir / "meta.json"
    meta_path.write_text(json.dumps(artifacts.meta, ensure_ascii=False, indent=2), encoding="utf-8")

    hull_path = target_dir / "hulls.geojson"
    if artifacts.hulls:
        geojson = {
            "type": "FeatureCollection",
            "features": artifacts.hulls,
        }
        hull_path.write_text(json.dumps(geojson, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        hull_path.write_text(json.dumps({"type": "FeatureCollection", "features": []}, ensure_ascii=False), encoding="utf-8")


def _estimate_eps(coords: np.ndarray, min_samples: int, *, percentile: float = 90.0) -> float:
    if coords.shape[0] < max(min_samples, 5):
        return 0.5
    kneigh = NearestNeighbors(n_neighbors=min_samples, metric="euclidean")
    kneigh.fit(coords)
    distances, _ = kneigh.kneighbors(coords)
    kth = distances[:, -1]
    value = float(np.percentile(kth, percentile))
    return max(value, 1e-3)


def _build_meta(
    *,
    mode: str,
    coords: np.ndarray,
    labels: np.ndarray,
    config: ClusterJobConfig,
    eps: float,
) -> dict[str, object]:
    total = int(coords.shape[0])
    cluster_ids = np.unique(labels)
    cluster_count = int(np.sum(cluster_ids != -1))
    noise_count = int(np.sum(labels == -1))
    now = int(time.time())

    bounds = {
        "min_x": float(np.min(coords[:, 0])) if total else 0.0,
        "max_x": float(np.max(coords[:, 0])) if total else 0.0,
        "min_y": float(np.min(coords[:, 1])) if total else 0.0,
        "max_y": float(np.max(coords[:, 1])) if total else 0.0,
    }

    params = {
        "umap": {
            "n_neighbors": config.n_neighbors,
            "min_dist": config.min_dist,
            "metric": config.metric,
            "random_state": config.random_state,
        },
        "dbscan": {
            "min_samples": config.min_samples,
            "eps": float(eps),
            "percentile": config.eps_percentile,
            "auto": config.eps is None,
        },
    }
    if config.limit:
        params["limit"] = int(config.limit)

    return {
        "mode": mode,
        "total": total,
        "cluster_count": cluster_count,
        "noise_count": noise_count,
        "noise_ratio": float(noise_count / total) if total else 0.0,
        "params": params,
        "generated_at": now,
        "bounds": bounds,
    }


def _build_hulls(coords: np.ndarray, labels: np.ndarray) -> list[dict[str, object]]:
    features: list[dict[str, object]] = []
    unique_labels = np.unique(labels)
    for cluster_id in unique_labels:
        if cluster_id == -1:
            continue
        mask = labels == cluster_id
        cluster_points = coords[mask]
        if cluster_points.shape[0] == 0:
            continue
        polygon = _cluster_polygon(cluster_points)
        feature = {
            "type": "Feature",
            "properties": {
                "cluster_id": int(cluster_id),
                "count": int(cluster_points.shape[0]),
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [polygon],
            },
        }
        features.append(feature)
    return features


def _cluster_polygon(points: np.ndarray) -> list[list[float]]:
    if points.shape[0] <= 2:
        center = points.mean(axis=0)
        spread = max(float(np.std(points)), 0.02)
        return _square_polygon(center, spread * 3)

    unique_points = _unique_rows(points)
    if unique_points.shape[0] < 3:
        center = unique_points.mean(axis=0)
        return _square_polygon(center, 0.05)

    hull = _monotonic_chain(unique_points)
    hull.append(hull[0])
    return [[float(x), float(y)] for x, y in hull]


def _unique_rows(points: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return points
    order = np.lexsort((points[:, 1], points[:, 0]))
    sorted_points = points[order]
    dedup = np.unique(sorted_points, axis=0)
    return dedup


def _monotonic_chain(points: np.ndarray) -> list[tuple[float, float]]:
    pts = sorted((float(x), float(y)) for x, y in points)
    if len(pts) <= 1:
        return pts

    def cross(o: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper: list[tuple[float, float]] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    return lower[:-1] + upper[:-1]


def _square_polygon(center: np.ndarray, size: float) -> list[list[float]]:
    half = max(size, 0.02) * 0.5
    cx, cy = float(center[0]), float(center[1])
    return [
        [cx - half, cy - half],
        [cx + half, cy - half],
        [cx + half, cy + half],
        [cx - half, cy + half],
        [cx - half, cy - half],
    ]


__all__ = [
    "CLUSTER_ALLOWED_MODES",
    "ClusterArtifacts",
    "ClusterJobConfig",
    "precompute_cluster",
    "save_artifacts",
]
