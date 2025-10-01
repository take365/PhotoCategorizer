"""Helpers for serving precomputed UMAP + clustering datasets."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Mapping, Sequence

import numpy as np

from .attr_index import AttributeIndexer, ATTR_KEYS

CLUSTER_MODES: tuple[str, ...] = ("location", "subject", "image")


@dataclass(slots=True)
class ClusterPoint:
    """Single point entry within a cluster map."""

    index: int
    image_id: int
    x: float
    y: float
    cluster_id: int


@dataclass(slots=True)
class ClusterDataset:
    """Container for precomputed UMAP coordinates and cluster assignments."""

    mode: str
    coords: np.ndarray
    image_ids: np.ndarray
    labels: np.ndarray
    meta: dict[str, Any]
    hulls: list[dict[str, Any]]
    cluster_indices: dict[int, np.ndarray]
    cluster_counts: dict[int, int]

    @property
    def total(self) -> int:
        return int(self.coords.shape[0])

    @property
    def noise_count(self) -> int:
        return int(self.cluster_counts.get(-1, 0))

    @property
    def cluster_ids(self) -> Sequence[int]:
        return tuple(sorted(self.cluster_counts.keys()))

    def iter_slice(self, start: int, stop: int) -> Iterator[ClusterPoint]:
        start = max(0, start)
        stop = min(self.total, stop)
        if start >= stop:
            return iter(())
        for idx in range(start, stop):
            yield ClusterPoint(
                index=idx,
                image_id=int(self.image_ids[idx]),
                x=float(self.coords[idx, 0]),
                y=float(self.coords[idx, 1]),
                cluster_id=int(self.labels[idx]),
            )

    def get_indices_for_cluster(self, cluster_id: int) -> np.ndarray:
        return self.cluster_indices.get(int(cluster_id), np.array([], dtype=np.int64))


class ClusterDatasetManager:
    """Lazy loader and summariser for precomputed cluster datasets."""

    def __init__(self, indexer: AttributeIndexer, cluster_root: Path) -> None:
        self.indexer = indexer
        self.cluster_root = cluster_root
        self._cache: dict[str, ClusterDataset] = {}

    # ------------------------------------------------------------------
    # Dataset accessors

    def available_modes(self) -> Sequence[str]:
        return CLUSTER_MODES

    def get_dataset(self, mode: str) -> ClusterDataset:
        mode = mode.lower()
        if mode not in CLUSTER_MODES:
            raise ValueError(f"mode must be one of: {', '.join(CLUSTER_MODES)}")
        if mode not in self._cache:
            self._cache[mode] = self._load_dataset(mode)
        return self._cache[mode]

    def _load_dataset(self, mode: str) -> ClusterDataset:
        base = self.cluster_root / mode
        if not base.exists():
            raise FileNotFoundError(f"Cluster dataset not found: {base}")

        coords = self._load_array(base, "coords")
        image_ids = self._load_array(base, "image_ids").astype("int64")
        labels = self._load_array(base, "labels").astype("int32")

        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError(f"coords array must be shape (N, 2); got {coords.shape}")
        if image_ids.shape[0] != coords.shape[0] or labels.shape[0] != coords.shape[0]:
            raise ValueError("coords, image_ids, labels must have the same length")

        meta_path = base / "meta.json"
        if meta_path.exists():
            with meta_path.open("r", encoding="utf-8") as fh:
                meta = json.load(fh)
        else:
            meta = {}

        hulls_path = base / "hulls.geojson"
        hulls: list[dict[str, Any]] = []
        if hulls_path.exists():
            with hulls_path.open("r", encoding="utf-8") as fh:
                hull_data = json.load(fh)
            if isinstance(hull_data, dict):
                if hull_data.get("type") == "FeatureCollection":
                    features = hull_data.get("features", [])
                    hulls = [feat for feat in features if isinstance(feat, dict)]
                else:
                    hulls = [hull_data]
            elif isinstance(hull_data, list):
                hulls = [item for item in hull_data if isinstance(item, dict)]

        cluster_indices: dict[int, np.ndarray] = {}
        cluster_counts: dict[int, int] = {}
        labels_int = labels.astype("int64")
        unique_labels = np.unique(labels_int)
        for cluster_id in unique_labels.tolist():
            indices = np.flatnonzero(labels_int == cluster_id)
            cluster_indices[int(cluster_id)] = indices
            cluster_counts[int(cluster_id)] = int(indices.shape[0])

        return ClusterDataset(
            mode=mode,
            coords=coords.astype("float32"),
            image_ids=image_ids.astype("int64"),
            labels=labels.astype("int32"),
            meta=meta,
            hulls=hulls,
            cluster_indices=cluster_indices,
            cluster_counts=cluster_counts,
        )

    def _load_array(self, base: Path, stem: str) -> np.ndarray:
        npy_path = base / f"{stem}.npy"
        if npy_path.exists():
            return np.load(npy_path)
        npz_path = base / f"{stem}.npz"
        if npz_path.exists():
            with np.load(npz_path) as data:
                if stem in data:
                    return data[stem]
                keys = list(data.keys())
                if not keys:
                    raise ValueError(f"{npz_path} does not contain any arrays")
                return data[keys[0]]
        raise FileNotFoundError(f"Missing array file for '{stem}' in {base}")

    # ------------------------------------------------------------------
    # Summaries

    def dataset_meta(self, mode: str) -> dict[str, Any]:
        dataset = self.get_dataset(mode)
        total = dataset.total
        if total <= 0:
            return {
                "mode": dataset.mode,
                "total": 0,
                "cluster_count": 0,
                "noise_count": 0,
                "noise_ratio": 0.0,
                "clusters": [],
                "hulls": [],
                "params": dataset.meta.get("params", {}),
                "bounds": {"min_x": 0.0, "max_x": 0.0, "min_y": 0.0, "max_y": 0.0},
            }

        payload_clusters: list[dict[str, Any]] = []
        for cluster_id in dataset.cluster_ids:
            count = dataset.cluster_counts.get(cluster_id, 0)
            centroid = self._cluster_centroid(dataset, cluster_id)
            spread = self._cluster_spread(dataset, cluster_id)
            payload_clusters.append(
                {
                    "cluster_id": cluster_id,
                    "count": count,
                    "ratio": count / total,
                    "is_noise": cluster_id == -1,
                    "centroid": centroid,
                    "spread": spread,
                    "color": self._cluster_color(cluster_id),
                }
            )

        hulls_payload: list[dict[str, Any]] = []
        for item in dataset.hulls:
            cluster_id = self._extract_cluster_id(item)
            hulls_payload.append(
                {
                    "cluster_id": cluster_id,
                    "color": self._cluster_color(cluster_id),
                    "geometry": item.get("geometry", item),
                    "properties": {k: v for k, v in item.get("properties", {}).items() if k != "cluster_id"},
                }
            )

        bounds = {
            "min_x": float(np.min(dataset.coords[:, 0])),
            "max_x": float(np.max(dataset.coords[:, 0])),
            "min_y": float(np.min(dataset.coords[:, 1])),
            "max_y": float(np.max(dataset.coords[:, 1])),
        }

        return {
            "mode": dataset.mode,
            "total": total,
            "cluster_count": len([c for c in dataset.cluster_ids if c != -1]),
            "noise_count": dataset.noise_count,
            "noise_ratio": dataset.noise_count / total,
            "clusters": payload_clusters,
            "hulls": hulls_payload,
            "params": dataset.meta.get("params", {}),
            "bounds": bounds,
        }

    def _cluster_centroid(self, dataset: ClusterDataset, cluster_id: int) -> dict[str, float]:
        indices = dataset.get_indices_for_cluster(cluster_id)
        if indices.size == 0:
            return {"x": 0.0, "y": 0.0}
        coords = dataset.coords[indices]
        centroid = coords.mean(axis=0)
        return {"x": float(centroid[0]), "y": float(centroid[1])}

    def _cluster_spread(self, dataset: ClusterDataset, cluster_id: int) -> float:
        indices = dataset.get_indices_for_cluster(cluster_id)
        if indices.size == 0:
            return 0.0
        coords = dataset.coords[indices]
        centroid = coords.mean(axis=0)
        distances = np.linalg.norm(coords - centroid, axis=1)
        return float(distances.mean() if distances.size else 0.0)

    def _extract_cluster_id(self, feature: Mapping[str, Any]) -> int:
        properties = feature.get("properties") if isinstance(feature, Mapping) else None
        if isinstance(properties, Mapping) and "cluster_id" in properties:
            try:
                return int(properties["cluster_id"])
            except (TypeError, ValueError):
                return -1
        return -1

    def cluster_color(self, cluster_id: int) -> str:
        return self._cluster_color(cluster_id)

    @staticmethod
    def _cluster_color(cluster_id: int) -> str:
        if cluster_id < 0:
            return "#94a3b8"  # slate-400
        hue = (cluster_id * 53) % 360
        return f"hsl({hue}, 70%, 55%)"

    # ------------------------------------------------------------------
    # Chunks & detail

    def chunk_points(self, mode: str, chunk_index: int, chunk_size: int) -> dict[str, Any]:
        dataset = self.get_dataset(mode)
        total = dataset.total
        if total == 0:
            return {
                "mode": dataset.mode,
                "chunk": 0,
                "chunk_size": chunk_size,
                "total": 0,
                "total_chunks": 0,
                "points": [],
            }

        chunk_size = max(1, chunk_size)
        total_chunks = math.ceil(total / chunk_size)
        chunk_index = max(0, min(chunk_index, total_chunks - 1))
        start = chunk_index * chunk_size
        stop = min(start + chunk_size, total)
        points = list(dataset.iter_slice(start, stop))
        payload_points = [
            {
                "index": point.index,
                "image_id": point.image_id,
                "x": point.x,
                "y": point.y,
                "cluster_id": point.cluster_id,
            }
            for point in points
        ]
        return {
            "mode": dataset.mode,
            "chunk": chunk_index,
            "chunk_size": chunk_size,
            "total": total,
            "total_chunks": total_chunks,
            "points": payload_points,
        }

    def cluster_detail(
        self,
        mode: str,
        cluster_id: int,
        *,
        top_k: int = 5,
    ) -> dict[str, Any]:
        dataset = self.get_dataset(mode)
        indices = dataset.get_indices_for_cluster(cluster_id)
        if indices.size == 0:
            raise ValueError(f"Cluster {cluster_id} has no points")

        coords = dataset.coords[indices]
        centroid = coords.mean(axis=0)
        distances = np.linalg.norm(coords - centroid, axis=1)
        order = np.argsort(distances)
        representative_indices = indices[order[: min(top_k, indices.size)]]
        representative_ids = [int(dataset.image_ids[idx]) for idx in representative_indices]

        summary = self._summarise_attributes(indices, dataset)
        cluster_size = int(indices.size)
        total = dataset.total or 1
        density = float(1.0 / (1.0 + distances.mean())) if distances.size else 0.0

        return {
            "mode": dataset.mode,
            "cluster_id": int(cluster_id),
            "count": cluster_size,
            "ratio": cluster_size / total,
            "centroid": {"x": float(centroid[0]), "y": float(centroid[1])},
            "density": density,
            "representative_image_ids": representative_ids,
            "attribute_summary": summary,
            "is_noise": cluster_id == -1,
            "color": self._cluster_color(int(cluster_id)),
        }

    def _summarise_attributes(self, indices: np.ndarray, dataset: ClusterDataset) -> dict[str, Any]:
        image_ids = [int(dataset.image_ids[idx]) for idx in indices.tolist()]
        attr_counters: dict[str, dict[str, int]] = {key: {} for key in ATTR_KEYS}
        keyword_counter: dict[str, int] = {}

        for image_id in image_ids:
            attrs = self.indexer.get_attributes(image_id)
            for key in ATTR_KEYS:
                value = str(attrs.get(key, "")).strip()
                if not value:
                    continue
                counter = attr_counters[key]
                counter[value] = counter.get(value, 0) + 1
                for token in self._tokenise(value):
                    keyword_counter[token] = keyword_counter.get(token, 0) + 1

        summary: dict[str, Any] = {}
        for key, counter in attr_counters.items():
            ordered = sorted(counter.items(), key=lambda item: item[1], reverse=True)
            summary[key] = [value for value, _ in ordered[:3]]

        ordered_keywords = sorted(keyword_counter.items(), key=lambda item: item[1], reverse=True)
        summary["keywords"] = [word for word, _ in ordered_keywords[:10]]
        return summary

    @staticmethod
    @lru_cache(maxsize=1024)
    def _tokenise(text: str) -> tuple[str, ...]:
        if not text:
            return tuple()
        separators = ["\n", "\r", ",", " "]
        tokens = [text]
        for sep in separators:
            parts: list[str] = []
            for token in tokens:
                parts.extend(token.split(sep))
            tokens = parts
        cleaned = []
        for token in tokens:
            word = token.strip()
            if not word:
                continue
            cleaned.append(word)
        return tuple(cleaned)


__all__ = [
    "CLUSTER_MODES",
    "ClusterDataset",
    "ClusterDatasetManager",
    "ClusterPoint",
]
