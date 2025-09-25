"""Color/monochrome judgement utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from PIL import Image

from .utils import ColorJudgeConfig


@dataclass(slots=True)
class ColorDecision:
    """Simple value object describing a color judgement result."""

    label: str
    score: float


def judge_color_mode(path: Path, config: ColorJudgeConfig) -> ColorDecision:
    """Return color/mono label for *path* based on *config*."""

    if config.method != "hsv_s_mean":
        raise ValueError(f"Unsupported color judgement method: {config.method}")

    with Image.open(path) as image:
        rgb = image.convert("RGB")
        arr = np.array(rgb)

    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    saturation = hsv[..., 1].astype(np.float32) / 255.0

    saturation_mean = float(saturation.mean())
    color_ratio = float(np.mean(saturation >= config.pixel_threshold))

    is_color = saturation_mean >= config.threshold or color_ratio >= config.min_color_ratio
    label = "color" if is_color else "mono"
    score = max(saturation_mean, color_ratio)
    return ColorDecision(label=label, score=score)
