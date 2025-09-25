"""Helpers for combining predictions from multiple sources."""

from __future__ import annotations

from typing import Dict, Iterable, Mapping


def merge_predictions(predictions: Iterable[Mapping[str, float]]) -> Dict[str, float]:
    """Average probability distributions with shared label keys."""

    accum: Dict[str, float] = {}
    count = 0
    for distribution in predictions:
        for label, value in distribution.items():
            accum[label] = accum.get(label, 0.0) + float(value)
        count += 1
    if count == 0:
        raise ValueError("merge_predictions requires at least one distribution")
    for label in accum:
        accum[label] /= count
    return accum
