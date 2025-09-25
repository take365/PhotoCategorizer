"""Common helpers for PhotoCategorizer."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Sequence

import numpy as np
import torch
import yaml

ALLOWED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass(slots=True)
class ZeroShotConfig:
    """Configuration values governing zero-shot classification."""

    model: str = "siglip"
    checkpoint: str | None = None
    device: str = "auto"
    batch_size: int = 16
    score_threshold: float = 0.55
    margin_threshold: float = 0.10
    multilingual: bool = True
    alternate_model: str | None = None
    alternate_checkpoint: str | None = None
    alternate_weight: float = 0.5


@dataclass(slots=True)
class ColorJudgeConfig:
    """Configuration for color/mono judgement."""

    method: str = "hsv_s_mean"
    threshold: float = 0.08
    pixel_threshold: float = 0.12
    min_color_ratio: float = 0.02


@dataclass(slots=True)
class OutputConfig:
    """Outputfile related settings."""

    csv_path: Path = Path("outputs/pred.csv")
    json_path: Path | None = Path("outputs/pred.json")
    move_to_class_dirs: bool = False
    class_dir_base: Path = Path("outputs/classified")


@dataclass(slots=True)
class ReviewConfig:
    """Review label configuration."""

    review_label: str = "needs_review"


@dataclass(slots=True)
class Config:
    """Top-level configuration dataclass."""

    classes: Dict[str, List[str]] = field(default_factory=dict)
    zero_shot: ZeroShotConfig = field(default_factory=ZeroShotConfig)
    color_judge: ColorJudgeConfig = field(default_factory=ColorJudgeConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    review: ReviewConfig = field(default_factory=ReviewConfig)
    seed: int = 42


def load_config(path: Path) -> Config:
    """Load YAML configuration from *path* into a Config object."""

    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    classes = {str(k): list(v) for k, v in (raw.get("classes") or {}).items()}
    if not classes:
        raise ValueError("config.yaml must define at least one class under 'classes'")

    zero_shot_cfg = ZeroShotConfig(
        model=str(_get_nested(raw, ("zero_shot", "model"), default="siglip")),
        checkpoint=_get_optional_str(raw, ("zero_shot", "checkpoint")),
        device=str(_get_nested(raw, ("zero_shot", "device"), default="auto")),
        batch_size=int(_get_nested(raw, ("zero_shot", "batch_size"), default=16)),
        score_threshold=float(
            _get_nested(raw, ("zero_shot", "score_threshold"), default=0.55)
        ),
        margin_threshold=float(
            _get_nested(raw, ("zero_shot", "margin_threshold"), default=0.10)
        ),
        multilingual=bool(_get_nested(raw, ("zero_shot", "multilingual"), default=True)),
        alternate_model=_get_optional_str(raw, ("zero_shot", "alternate_model")),
        alternate_checkpoint=_get_optional_str(
            raw, ("zero_shot", "alternate_checkpoint")
        ),
        alternate_weight=float(
            _get_nested(raw, ("zero_shot", "alternate_weight"), default=0.5)
        ),
    )

    color_cfg = ColorJudgeConfig(
        method=str(_get_nested(raw, ("color_judge", "method"), default="hsv_s_mean")),
        threshold=float(_get_nested(raw, ("color_judge", "threshold"), default=0.08)),
        pixel_threshold=float(
            _get_nested(raw, ("color_judge", "pixel_threshold"), default=0.12)
        ),
        min_color_ratio=float(
            _get_nested(raw, ("color_judge", "min_color_ratio"), default=0.02)
        ),
    )

    output_cfg = OutputConfig(
        csv_path=Path(
            _get_nested(raw, ("output", "csv_path"), default="outputs/pred.csv")
        ),
        json_path=_get_optional_path(raw, ("output", "json_path")),
        move_to_class_dirs=bool(
            _get_nested(raw, ("output", "move_to_class_dirs"), default=False)
        ),
        class_dir_base=Path(
            _get_nested(raw, ("output", "class_dir_base"), default="outputs/classified")
        ),
    )

    review_cfg = ReviewConfig(
        review_label=str(
            _get_nested(raw, ("review", "review_label"), default="needs_review")
        )
    )

    return Config(
        classes=classes,
        zero_shot=zero_shot_cfg,
        color_judge=color_cfg,
        output=output_cfg,
        review=review_cfg,
        seed=int(raw.get("seed", 42)),
    )


def _get_nested(mapping: Mapping, path: Sequence[str], default=None):
    current = mapping
    for key in path[:-1]:
        if not isinstance(current, Mapping):
            return default
        current = current.get(key)  # type: ignore[index]
    if isinstance(current, Mapping) and path[-1] in current:
        return current[path[-1]]
    return default


def _get_optional_str(mapping: Mapping, path: Sequence[str]) -> str | None:
    value = _get_nested(mapping, path, default=None)
    if value is None:
        return None
    return str(value)


def _get_optional_path(mapping: Mapping, path: Sequence[str]) -> Path | None:
    value = _get_optional_str(mapping, path)
    if value is None or value.strip() == "":
        return None
    return Path(value)


def resolve_device(requested: str) -> torch.device:
    """Resolve the torch.device to use from a config string."""

    requested = requested.lower().strip()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():  # type: ignore[attr-defined]
            return torch.device("mps")
        return torch.device("cpu")
    if requested in {"cuda", "cpu", "mps"}:
        return torch.device(requested)
    if requested.startswith("cuda:"):
        return torch.device(requested)
    raise ValueError(f"Unsupported device specifier: {requested}")


def seed_everything(seed: int) -> None:
    """Seed random number generators for reproducibility."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def iter_batches(items: Sequence[Path], batch_size: int) -> Iterator[List[Path]]:
    """Yield successive batches of `items` with size up to *batch_size*."""

    total = len(items)
    for start in range(0, total, batch_size):
        yield list(items[start : start + batch_size])


def list_image_paths(root: Path) -> List[Path]:
    """Return sorted image paths under *root* (non-recursive)."""

    if not root.exists():
        raise FileNotFoundError(f"Input directory not found: {root}")
    paths = [p for p in sorted(root.iterdir()) if p.suffix.lower() in ALLOWED_IMAGE_SUFFIXES]
    if not paths:
        raise FileNotFoundError(
            f"No images with extensions {sorted(ALLOWED_IMAGE_SUFFIXES)} found in {root}"
        )
    return paths


def ensure_parent_dir(path: Path) -> None:
    """Ensure the parent directory of *path* exists."""

    path.parent.mkdir(parents=True, exist_ok=True)


def to_serialisable(value):  # type: ignore[no-untyped-def]
    """Convert tensors/numpy values to native Python types for JSON output."""

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value
