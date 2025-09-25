"""Evaluation and reporting helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from .zero_shot import ZeroShotResult
from .gpt import GPTCostEstimate


@dataclass(slots=True)
class EvaluationMetrics:
    """Container for evaluation summary information."""

    labels: List[str]
    confusion: np.ndarray
    report: str
    accuracy: float
    auto_confirm_rate: float
    auto_confirm_accuracy: float


def compute_metrics(
    predictions: Sequence[ZeroShotResult],
    ground_truth: Dict[str, str],
    class_labels: Sequence[str],
) -> EvaluationMetrics:
    """Compute confusion matrix and summary statistics."""

    y_true: List[str] = []
    y_pred: List[str] = []
    review_flags: List[bool] = []

    for pred in predictions:
        key = pred.path.name
        if key not in ground_truth:
            raise KeyError(f"Missing ground-truth label for {key}")
        y_true.append(ground_truth[key])
        y_pred.append(pred.label)
        review_flags.append(pred.needs_review)

    labels = list(class_labels)
    confusion = confusion_matrix(y_true, y_pred, labels=labels)
    accuracy = (confusion.diagonal().sum() / confusion.sum()) if confusion.sum() else 0.0

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
        target_names=labels,
    )

    auto_confirm_mask = [not flag for flag in review_flags]
    if any(auto_confirm_mask):
        auto_indices = [i for i, flag in enumerate(auto_confirm_mask) if flag]
        correct_auto = sum(
            1 for idx in auto_indices if y_true[idx] == y_pred[idx]
        )
        auto_confirm_rate = len(auto_indices) / len(predictions)
        auto_confirm_accuracy = correct_auto / max(len(auto_indices), 1)
    else:
        auto_confirm_rate = 0.0
        auto_confirm_accuracy = 0.0

    return EvaluationMetrics(
        labels=labels,
        confusion=confusion,
        report=report,
        accuracy=accuracy,
        auto_confirm_rate=auto_confirm_rate,
        auto_confirm_accuracy=auto_confirm_accuracy,
    )


def render_markdown_report(
    metrics: EvaluationMetrics,
    output_path: Path,
    review_label: str,
    *,
    model_name: str,
    reference_notes: Optional[Sequence[str]] = None,
    gpt_cost: Optional[GPTCostEstimate] = None,
) -> None:
    """Write a Markdown evaluation report to *output_path*."""

    df_conf = pd.DataFrame(metrics.confusion, index=metrics.labels, columns=metrics.labels)
    with output_path.open("w", encoding="utf-8") as fh:
        fh.write("# PhotoCategorizer Evaluation\n\n")
        fh.write(f"- 評価モデル: {model_name}\n")
        fh.write(f"- Overall accuracy: {metrics.accuracy:.3f}\n")
        fh.write(
            f"- Auto-confirm rate (not flagged as {review_label}): {metrics.auto_confirm_rate:.3f}\n"
        )
        fh.write(
            f"- Accuracy within auto-confirmed subset: {metrics.auto_confirm_accuracy:.3f}\n\n"
        )
        fh.write("## Confusion Matrix\n\n")
        fh.write(df_conf.to_markdown())
        fh.write("\n\n## Classification Report\n\n")
        fh.write("````\n")
        fh.write(metrics.report)
        fh.write("\n````\n")

        if reference_notes:
            fh.write("\n## 参考情報\n")
            for note in reference_notes:
                fh.write(f"- {note}\n")

        if gpt_cost:
            fh.write("\n## GPT利用コスト見積\n")
            fh.write(f"- モデル: {gpt_cost.model}\n")
            fh.write(f"- 入力トークン: {gpt_cost.input_tokens:,}\n")
            if gpt_cost.cached_input_tokens:
                fh.write(f"- キャッシュ済み入力トークン: {gpt_cost.cached_input_tokens:,}\n")
            fh.write(f"- 出力トークン: {gpt_cost.output_tokens:,}\n")
            fh.write(f"- 合計トークン: {gpt_cost.total_tokens:,}\n")
            fh.write(
                f"- 合計コスト: ${gpt_cost.cost_usd:.5f} ≈ {gpt_cost.cost_jpy:.2f}円\n"
            )
            fh.write("（1ドル=150円換算）\n")
