"""Scatter plot projection utilities for location/subject embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

from .attr_index import ATTR_KEYS, AttributeIndexer


@dataclass(slots=True)
class AxisDefinition:
    """Axis configuration consisting of attribute key and term clusters."""

    key: str
    positives: Sequence[str]
    negatives: Sequence[str]


@dataclass(slots=True)
class ScatterPoint:
    """Projected point metadata for scatter visualisation."""

    image_id: int
    raw_x: float
    raw_y: float
    x: float
    y: float
    magnitude: float


@dataclass(slots=True)
class ScatterResult:
    """Bundle of scatter projection outputs."""

    points: list[ScatterPoint]
    stats: Mapping[str, float]


class ScatterProjector:
    """Compute scatter projections from attribute embeddings."""

    def __init__(self, indexer: AttributeIndexer) -> None:
        self.indexer = indexer

    # ------------------------------------------------------------------
    # Public API

    def project(
        self,
        axis_x: AxisDefinition,
        axis_y: AxisDefinition,
        *,
        limit: int = 200,
        scaling: str = "robust",
    ) -> ScatterResult:
        if axis_x.key not in ATTR_KEYS:
            raise ValueError(f"Unsupported axis key: {axis_x.key}")
        if axis_y.key not in ATTR_KEYS:
            raise ValueError(f"Unsupported axis key: {axis_y.key}")

        x_axis = self._build_axis_vector(axis_x)
        y_axis = self._build_axis_vector(axis_y)
        # グラム・シュミットで直交化
        y_axis = self._orthogonalise(y_axis, x_axis)

        ids_x, embeddings_x = self.indexer.load_attr_embeddings(axis_x.key)
        ids_y, embeddings_y = self.indexer.load_attr_embeddings(axis_y.key)

        map_x = {int(idx): embeddings_x[pos] for pos, idx in enumerate(ids_x)}
        map_y = {int(idx): embeddings_y[pos] for pos, idx in enumerate(ids_y)}
        common_ids = sorted(set(map_x.keys()) & set(map_y.keys()))
        if not common_ids:
            raise ValueError("No overlapping images between axis datasets")

        raw_coords: list[tuple[int, float, float]] = []
        for image_id in common_ids:
            vec_x = map_x[image_id]
            vec_y = map_y[image_id]
            raw_x = float(np.dot(vec_x, x_axis))
            raw_y = float(np.dot(vec_y, y_axis))
            raw_coords.append((image_id, raw_x, raw_y))

        xs = np.array([coord[1] for coord in raw_coords], dtype="float32")
        ys = np.array([coord[2] for coord in raw_coords], dtype="float32")
        if scaling == "robust":
            scaled_x, sx_stats = self._robust_scale(xs)
            scaled_y, sy_stats = self._robust_scale(ys)
        elif scaling == "none":
            scaled_x, sx_stats = xs, {"median": float(np.median(xs)), "iqr": 0.0}
            scaled_y, sy_stats = ys, {"median": float(np.median(ys)), "iqr": 0.0}
        else:
            raise ValueError(f"Unsupported scaling mode: {scaling}")

        points: list[ScatterPoint] = []
        for (image_id, raw_x, raw_y), sx, sy in zip(raw_coords, scaled_x, scaled_y, strict=False):
            magnitude = abs(sx) + abs(sy)
            points.append(
                ScatterPoint(
                    image_id=image_id,
                    raw_x=raw_x,
                    raw_y=raw_y,
                    x=float(sx),
                    y=float(sy),
                    magnitude=float(magnitude),
                )
            )

        points.sort(key=lambda item: item.magnitude, reverse=True)
        if limit > 0:
            points = points[: min(limit, len(points))]

        stats = {
            "count": float(len(raw_coords)),
            "x_median": sx_stats.get("median", 0.0),
            "x_iqr": sx_stats.get("iqr", 0.0),
            "y_median": sy_stats.get("median", 0.0),
            "y_iqr": sy_stats.get("iqr", 0.0),
        }
        return ScatterResult(points=points, stats=stats)

    # ------------------------------------------------------------------
    # Axis helpers

    def _build_axis_vector(self, axis: AxisDefinition) -> np.ndarray:
        pos = self._mean_embedding(axis.positives)
        neg = self._mean_embedding(axis.negatives)
        vector = pos - neg
        norm = np.linalg.norm(vector)
        if not np.isfinite(norm) or norm == 0.0:
            raise ValueError("Axis vector has zero norm; adjust terms")
        return vector / norm

    def _mean_embedding(self, terms: Sequence[str]) -> np.ndarray:
        vectors: list[np.ndarray] = []
        seen: set[str] = set()
        for term in terms:
            cleaned = term.strip()
            if not cleaned:
                continue
            if cleaned.lower() in seen:
                continue
            seen.add(cleaned.lower())
            vec = self.indexer.encode_text(cleaned)
            vectors.append(vec.astype("float32"))
        if not vectors:
            raise ValueError("Axis requires at least one non-empty term")
        stacked = np.stack(vectors)
        mean_vec = stacked.mean(axis=0)
        norm = np.linalg.norm(mean_vec)
        if not np.isfinite(norm) or norm == 0.0:
            raise ValueError("Mean embedding has zero norm")
        return mean_vec / norm

    def _orthogonalise(self, vector: np.ndarray, basis: np.ndarray) -> np.ndarray:
        proj = np.dot(vector, basis) * basis
        adjusted = vector - proj
        norm = np.linalg.norm(adjusted)
        if not np.isfinite(norm) or norm == 0.0:
            return basis.copy()
        return adjusted / norm

    def _robust_scale(self, values: np.ndarray) -> tuple[np.ndarray, Mapping[str, float]]:
        median = float(np.median(values))
        q1 = float(np.percentile(values, 25))
        q3 = float(np.percentile(values, 75))
        iqr = q3 - q1
        scale = iqr if iqr > 1e-6 else float(np.std(values))
        if scale <= 1e-6:
            scale = 1.0
        scaled = (values - median) / scale
        stats = {"median": median, "iqr": iqr}
        return scaled, stats
