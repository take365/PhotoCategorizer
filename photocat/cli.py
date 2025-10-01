"""Command-line interface for PhotoCategorizer."""

from __future__ import annotations

import copy
import json
import os
import shutil
from collections import Counter, defaultdict
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence

import pandas as pd
import typer
import uvicorn

from .color_judge import ColorDecision, judge_color_mode
from .attr_index import ATTR_KEYS, AttributeIndexer, IndexPaths, create_text_client, load_attr_records
from .cluster_precompute import (
    CLUSTER_ALLOWED_MODES,
    ClusterJobConfig,
    precompute_cluster,
    save_artifacts,
)
from .attr_poc import ModelSpec, generate_html_report, run_attribute_extraction
from .server import create_app
from .gpt import (
    GPTCostEstimate,
    GPTDecision,
    classify_with_gpt,
    estimate_gpt_cost,
)
from .pixabay import (
    PixabayClient,
    build_params,
    load_api_key,
    save_metadata,
)
from .utils import (
    Config,
    ensure_parent_dir,
    list_image_paths,
    load_config,
    seed_everything,
    to_serialisable,
)
from .zero_shot import ZeroShotResult, classify_zero_shot

app = typer.Typer(help="Zero-shot photo categorization toolkit")

MODEL_DISPLAY_NAMES = {
    "openclip": "OpenCLIP",
    "siglip": "SigLIP",
}

DECISION_SOURCES = {
    "openclip": "OpenCLIP",
    "gpt": "GPT",
}

OPENCLIP_REVIEW_THRESHOLD = 0.003  # 0.3% difference


def _apply_overrides(
    config: Config,
    model: Optional[str],
    checkpoint: Optional[str],
    batch_size: Optional[int],
    score_threshold: Optional[float],
    margin_threshold: Optional[float],
    csv_path: Optional[Path],
    json_path: Optional[Path],
    move_to_class_dirs: Optional[bool],
    class_dir_base: Optional[Path],
) -> None:
    if model:
        config.zero_shot.model = model
    if checkpoint:
        config.zero_shot.checkpoint = checkpoint
    if batch_size:
        config.zero_shot.batch_size = batch_size
    if score_threshold:
        config.zero_shot.score_threshold = score_threshold
    if margin_threshold:
        config.zero_shot.margin_threshold = margin_threshold
    if csv_path:
        config.output.csv_path = csv_path
    if json_path is not None:
        config.output.json_path = json_path
    if move_to_class_dirs is not None:
        config.output.move_to_class_dirs = move_to_class_dirs
    if class_dir_base:
        config.output.class_dir_base = class_dir_base


def _load_env_value(name: str, env_path: Optional[Path]) -> Optional[str]:
    value = os.getenv(name)
    if value:
        return value
    env_file = env_path or Path(".env")
    if not env_file.exists():
        return None
    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, raw_value = line.split("=", 1)
        if key.strip() == name:
            cleaned = raw_value.strip().strip('"').strip("'")
            if cleaned:
                return cleaned
    return None


def _load_env_int(name: str, env_path: Optional[Path]) -> Optional[int]:
    value = _load_env_value(name, env_path)
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _load_env_float(name: str, env_path: Optional[Path]) -> Optional[float]:
    value = _load_env_value(name, env_path)
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


@app.command()
def classify(
    input_dir: Path = typer.Argument(..., exists=True, file_okay=False, resolve_path=True),
    config_path: Path = typer.Option(Path("config.yaml"), "--config", exists=True, resolve_path=True),
    model: Optional[str] = typer.Option(None, help="Primary zero-shot backend to use (siglip/openclip)"),
    checkpoint: Optional[str] = typer.Option(None, help="Override checkpoint identifier"),
    batch_size: Optional[int] = typer.Option(None, min=1),
    score_threshold: Optional[float] = typer.Option(None, min=0.0, max=1.0),
    margin_threshold: Optional[float] = typer.Option(None, min=0.0, max=1.0),
    out_csv: Optional[Path] = typer.Option(None, "--out-csv", resolve_path=True),
    out_json: Optional[Path] = typer.Option(None, "--out-json", resolve_path=True),
    move_to_class_dirs: Optional[bool] = typer.Option(None, help="Move files into class folders"),
    class_dir_base: Optional[Path] = typer.Option(
        None, "--class-dir-base", help="Base directory for class folders", resolve_path=True
    ),
    dry_run: bool = typer.Option(False, help="Skip file moves and writes; show preview only"),
) -> None:
    """Classify images in *input_dir* using zero-shot models."""

    config = load_config(config_path)
    _apply_overrides(
        config,
        model,
        checkpoint,
        batch_size,
        score_threshold,
        margin_threshold,
        out_csv,
        out_json,
        move_to_class_dirs,
        class_dir_base,
    )
    seed_everything(config.seed)

    image_paths = list_image_paths(input_dir)
    typer.echo(f"Found {len(image_paths)} images under {input_dir}")

    zero_shot_results = classify_zero_shot(config, image_paths)
    color_results = {res.path: judge_color_mode(res.path, config.color_judge) for res in zero_shot_results}

    table_rows = [_row_dict(res, color_results[res.path], config) for res in zero_shot_results]
    df = pd.DataFrame(table_rows)

    typer.echo("Top categories (first 5):")
    for row in table_rows[:5]:
        typer.echo(
            f" - {Path(row['path']).name}: {row['label']} "
            f"(p={row['score']:.3f}, margin={row['margin']:.3f}, color={row['color_mode']})"
        )

    if dry_run:
        typer.echo("Dry-run enabled; skipping file writes and moves")
        return

    ensure_parent_dir(config.output.csv_path)
    df.to_csv(config.output.csv_path, index=False)
    typer.echo(f"Wrote CSV to {config.output.csv_path}")

    if config.output.json_path:
        ensure_parent_dir(config.output.json_path)
        with config.output.json_path.open("w", encoding="utf-8") as fh:
            json.dump([_serialise_row(row) for row in table_rows], fh, ensure_ascii=False, indent=2)
        typer.echo(f"Wrote JSON to {config.output.json_path}")

    if config.output.move_to_class_dirs:
        _move_files(zero_shot_results, color_results, config)


def _row_dict(result: ZeroShotResult, color: ColorDecision, config: Config) -> dict:
    return {
        "path": str(result.path),
        "label": result.label,
        "score": result.score,
        "runner_up": result.runner_up_label or "",
        "runner_up_score": result.runner_up_score,
        "margin": result.margin,
        config.review.review_label: result.needs_review,
        "color_mode": color.label,
        "color_score": color.score,
    }


def _serialise_row(row: dict) -> dict:
    return {key: to_serialisable(value) for key, value in row.items()}


def _move_files(
    zero_shot_results: Sequence[ZeroShotResult],
    color_results: dict[Path, ColorDecision],
    config: Config,
) -> None:
    base = config.output.class_dir_base
    for result in zero_shot_results:
        decision = color_results[result.path]
        label = "needs_review" if result.needs_review else result.label
        target_dir = base / label / decision.label
        target_dir.mkdir(parents=True, exist_ok=True)
        destination = target_dir / result.path.name
        if destination.exists():
            destination = _resolve_collision(destination)
        shutil.move(str(result.path), destination)
    typer.echo(f"Moved files into class directories under {base}")


def _resolve_collision(path: Path) -> Path:
    stem, suffix = path.stem, path.suffix
    counter = 1
    candidate = path
    while candidate.exists():
        candidate = path.with_name(f"{stem}_{counter}{suffix}")
        counter += 1
    return candidate


def _load_ground_truth(labels_csv: Path) -> dict[str, str]:
    df = pd.read_csv(labels_csv)
    expected_columns = {"filename", "label"}
    alternate_columns = {"path", "label"}
    if expected_columns.issubset(df.columns):
        keys = df["filename"].astype(str)
    elif alternate_columns.issubset(df.columns):
        keys = df["path"].astype(str)
    else:
        raise typer.BadParameter(
            "labels CSV must contain either columns 'filename,label' or 'path,label'",
            param_hint="--labels",
        )
    labels = df["label"].astype(str)
    return {Path(name).name: label for name, label in zip(keys, labels)}


def _prepare_model_config(config: Config, model_name: str) -> Config:
    cfg = copy.deepcopy(config)
    cfg.zero_shot.model = model_name
    cfg.zero_shot.alternate_model = None
    cfg.zero_shot.alternate_checkpoint = None
    cfg.zero_shot.alternate_weight = 0.0
    if model_name == "openclip" and not cfg.zero_shot.checkpoint:
        # Fallback to alternate checkpoint if provided.
        if config.zero_shot.alternate_checkpoint:
            cfg.zero_shot.checkpoint = config.zero_shot.alternate_checkpoint
    return cfg


def _map_results_by_path(results: Sequence[ZeroShotResult]) -> dict[Path, ZeroShotResult]:
    return {res.path: res for res in results}


def _gather_prediction_records(
    image_paths: Sequence[Path],
    predictions: dict[str, dict[Path, ZeroShotResult]],
    gpt_decisions: dict[Path, GPTDecision],
    ground_truth: dict[str, str],
    eval_dir: Path,
    primary_model: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in image_paths:
        record: dict[str, Any] = {
            "path": str(path),
            "original_path": str(path),
            "filename": path.name,
            "relative_path": os.path.relpath(path, eval_dir),
        }
        gt_label = ground_truth.get(path.name)
        if gt_label is not None:
            record["ground_truth"] = gt_label
        for model_name, result_map in predictions.items():
            result = result_map.get(path)
            if not result:
                continue
            prefix = model_name.lower()
            record[f"{prefix}_label"] = result.label
            record[f"{prefix}_score"] = result.score
            record[f"{prefix}_runner_up_label"] = result.runner_up_label
            record[f"{prefix}_runner_up_score"] = result.runner_up_score
            record[f"{prefix}_margin"] = result.margin
            record[f"{prefix}_needs_review"] = result.needs_review
            record[f"{prefix}_scores"] = result.scores
            if model_name == primary_model:
                record["needs_review"] = result.needs_review
        decision = gpt_decisions.get(path)
        if decision:
            record["gpt_label"] = decision.label
            if decision.confidence is not None:
                record["gpt_confidence"] = decision.confidence
            record["gpt_raw_response"] = decision.raw_response
        records.append(record)
    return records


def _write_predictions_json(records: Sequence[dict[str, Any]], json_path: Path) -> None:
    ensure_parent_dir(json_path)
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(records, fh, ensure_ascii=False, indent=2)


def _apply_final_labels(
    records: Sequence[dict[str, Any]],
    mode: str,
    threshold: float,
) -> None:
    mode_key = mode.lower()
    for record in records:
        margin = float(record.get("openclip_margin") or 0.0)
        gpt_label = record.get("gpt_label") or ""
        gpt_conf = record.get("gpt_confidence")
        openclip_label = record.get("openclip_label") or ""
        openclip_score = record.get("openclip_score")
        final_label = openclip_label or gpt_label or "unassigned"
        final_source = "openclip"
        final_score = openclip_score

        if mode_key in {"gptall", "gpt-all", "all"}:
            if gpt_label:
                final_label = gpt_label
                final_source = "gpt"
                final_score = gpt_conf
        elif mode_key in {"gpt-review", "review"}:
            if margin < threshold and gpt_label:
                final_label = gpt_label
                final_source = "gpt"
                final_score = gpt_conf
        else:  # gpt-off or default
            final_label = openclip_label or gpt_label or "unassigned"
            final_source = "openclip"
            final_score = openclip_score

        if not final_label:
            final_label = "unassigned"
        if final_source not in DECISION_SOURCES:
            final_source = "openclip"

        record["final_label"] = final_label
        record["final_source"] = final_source
        if final_score is not None:
            record["final_score"] = final_score

def _score_to_percent(value: Any) -> str:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return "—"
    clamped = min(max(score, 0.0), 1.0)
    return f"{clamped * 100:.1f}%"


def _render_gallery_html(
    records: Sequence[dict[str, Any]],
    html_path: Path,
    eval_dir: Path,
    model_order: Iterable[str],
    review_label: str,
) -> None:
    ensure_parent_dir(html_path)
    parent = html_path.parent
    models = list(dict.fromkeys(model_order))
    lines: list[str] = [
        "<!DOCTYPE html>",
        "<html lang=\"ja\">",
        "<head>",
        "  <meta charset=\"utf-8\">",
        "  <title>PhotoCategorizer Evaluation Gallery</title>",
        "  <style>",
        "    body { font-family: 'Inter', 'Helvetica Neue', Arial, sans-serif; background: #f5f7fa; margin: 0; padding: 24px; color: #1a1c1e; }",
        "    h1 { font-size: 1.6rem; margin-bottom: 16px; }",
        "    .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); gap: 20px; }",
        "    .card { background: #ffffff; border-radius: 12px; box-shadow: 0 10px 20px rgba(15, 23, 42, 0.08); overflow: hidden; display: flex; flex-direction: column; }",
        "    .thumb { position: relative; padding-top: 65%; overflow: hidden; }",
        "    .thumb img { position: absolute; inset: 0; width: 100%; height: 100%; object-fit: cover; }",
        "    .info { padding: 16px; display: flex; flex-direction: column; gap: 8px; }",
        "    .path { font-size: 0.85rem; color: #475569; word-break: break-all; display: flex; align-items: center; gap: 8px; }",
        "    .badge { background: #ef4444; color: #fff; border-radius: 12px; padding: 2px 8px; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.04em; }",
        "    .scores { display: flex; flex-direction: column; gap: 6px; font-size: 0.95rem; }",
        "    .label { font-weight: 600; color: #1e293b; }",
        "    .percent { margin-left: 6px; color: #334155; font-variant-numeric: tabular-nums; }",
        "    .muted { color: #94a3b8; }",
        "  </style>",
        "</head>",
        "<body>",
        f"  <h1>Evaluation Gallery – {escape(str(eval_dir))}</h1>",
        "  <div class=\"grid\">",
    ]
    for item in records:
        image_path = Path(item["path"])
        img_src = os.path.relpath(image_path, parent)
        display_path = escape(str(item.get("relative_path", image_path.name)))
        needs_review = bool(item.get("needs_review", False))
        lines.append("    <div class=\"card\">")
        lines.append(f"      <div class=\"thumb\"><img src=\"{escape(img_src)}\" alt=\"{escape(image_path.name)}\"></div>")
        lines.append("      <div class=\"info\">")
        badge_html = f" <span class=\"badge\">{escape(review_label)}</span>" if needs_review else ""
        lines.append(f"        <div class=\"path\">{display_path}{badge_html}</div>")
        lines.append("        <div class=\"scores\">")
        color_tag = item.get("color_bucket")
        if color_tag == "grayscale":
            lines.append("          <div><span class=\"label\">白黒</span></div>")
        final_label = item.get("final_label") or "—"
        final_source = DECISION_SOURCES.get(item.get("final_source"), item.get("final_source", ""))
        final_percent = _score_to_percent(item.get("final_score")) if item.get("final_score") is not None else "—"
        lines.append(
            f"          <div>Final: <span class=\"label\">{escape(str(final_label))}</span> "
            f"<span class=\"percent\">{final_percent}</span>"
            f" <span class=\"muted\">({escape(str(final_source or 'OpenCLIP'))})</span></div>"
        )
        for model_name in models:
            prefix = model_name.lower()
            display_name = MODEL_DISPLAY_NAMES.get(prefix, model_name.title())
            if prefix == "openclip":
                primary_label = item.get("openclip_label")
                primary_score = item.get("openclip_score")
                runner_label = item.get("openclip_runner_up_label")
                runner_score = item.get("openclip_runner_up_score")
                if primary_label:
                    lines.append(
                        f"          <div>{escape(display_name)} 1位: <span class=\"label\">{escape(str(primary_label))}</span> "
                        f"<span class=\"percent\">{_score_to_percent(primary_score)}</span></div>"
                    )
                    if runner_label:
                        lines.append(
                            f"          <div>{escape(display_name)} 2位: <span class=\"label\">{escape(str(runner_label))}</span> "
                            f"<span class=\"percent\">{_score_to_percent(runner_score)}</span></div>"
                        )
                else:
                    lines.append(
                        f"          <div class=\"muted\">{escape(display_name)}: —</div>"
                    )
            else:
                label_key = f"{prefix}_label"
                score_key = f"{prefix}_score"
                label_value = item.get(label_key)
                score_value = item.get(score_key)
                if label_value:
                    lines.append(
                        f"          <div>{escape(display_name)}: <span class=\"label\">{escape(str(label_value))}</span> "
                        f"<span class=\"percent\">{_score_to_percent(score_value)}</span></div>"
                    )
                else:
                    lines.append(
                        f"          <div class=\"muted\">{escape(display_name)}: —</div>"
                    )
        gpt_label = item.get("gpt_label")
        if gpt_label:
            percent_text = _score_to_percent(item.get("gpt_confidence")) if item.get("gpt_confidence") is not None else "—"
            lines.append(
                f"          <div>GPT: <span class=\"label\">{escape(str(gpt_label))}</span> "
                f"<span class=\"percent\">{percent_text}</span></div>"
            )
        elif "gpt_label" in item:
            lines.append("          <div class=\"muted\">GPT: —</div>")
        lines.append("        </div>")
        lines.append("      </div>")
        lines.append("    </div>")
    lines.extend(["  </div>", "</body>", "</html>"])
    html_path.write_text("\n".join(lines), encoding="utf-8")


def _render_assignment_summary(
    records: Sequence[dict[str, Any]],
    report_path: Path,
    mode: str,
    threshold: float,
    gpt_cost: Optional[GPTCostEstimate],
    reference_notes: Optional[Sequence[str]] = None,
    warnings: Optional[Sequence[str]] = None,
) -> None:
    ensure_parent_dir(report_path)
    total = len(records)
    label_counts = Counter(rec.get("final_label", "unassigned") for rec in records)
    color_counts = Counter(rec.get("color_bucket", "color") for rec in records)
    source_counts = Counter(rec.get("final_source", "openclip") for rec in records)
    lines: list[str] = [
        "# PhotoCategorizer Assignment",
        "",
        f"- モード: {mode}",
        f"- OpenCLIP差しきい値: {threshold:.3f} (≈ {threshold*100:.1f}%)",
        f"- 総画像数: {total}",
        "",
    ]
    if reference_notes:
        lines.append("## 参考情報")
        for note in reference_notes:
            lines.append(f"- {note}")
        lines.append("")
    if warnings:
        lines.append("## 警告")
        for warn in warnings:
            lines.append(f"- {warn}")
        lines.append("")
    lines.extend(
        [
            "## Final Labels",
            "| label | count |",
            "|:------|------:|",
        ]
    )
    for label, count in sorted(label_counts.items()):
        lines.append(f"| {label} | {count} |")
    lines.append("")
    lines.extend([
        "## Color Buckets",
        "| bucket | count |",
        "|:-------|------:|",
    ])
    for bucket, count in sorted(color_counts.items()):
        lines.append(f"| {bucket} | {count} |")
    lines.append("")
    lines.extend([
        "## Decision Sources",
        "| source | count |",
        "|:-------|------:|",
    ])
    for source, count in sorted(source_counts.items()):
        source_name = DECISION_SOURCES.get(source, source)
        lines.append(f"| {source_name} | {count} |")
    lines.append("")
    lines.append("## ファイル一覧")
    grouped: dict[str, list[str]] = defaultdict(list)
    for rec in records:
        grouped[rec.get("final_label", "unassigned")].append(rec.get("relative_path", rec.get("path", "")))
    for label in sorted(grouped):
        lines.append(f"### {label}")
        for rel_path in sorted(grouped[label]):
            lines.append(f"- {rel_path}")
        lines.append("")
    if gpt_cost:
        lines.extend(
            [
                "## GPT利用コスト見積",
                f"- モデル: {gpt_cost.model}",
                f"- 入力トークン: {gpt_cost.input_tokens:,}",
                f"- 出力トークン: {gpt_cost.output_tokens:,}",
                f"- 合計トークン: {gpt_cost.total_tokens:,}",
                f"- 合計コスト: ${gpt_cost.cost_usd:.5f} ≈ {gpt_cost.cost_jpy:.2f}円",
                "（1ドル=150円換算）",
            ]
        )
    report_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _relocate_inputs(
    records: Sequence[dict[str, Any]],
    color_results: dict[str, ColorDecision],
    run_dir: Path,
) -> None:
    for record in records:
        original_path_str = record.get("original_path", record["path"])
        original_path = Path(original_path_str)
        decision = color_results.get(original_path_str)
        color_bucket = "grayscale" if decision and decision.label == "mono" else "color"
        record["color_mode"] = (decision.label if decision else "")
        if decision:
            record["color_score"] = decision.score
        record["color_bucket"] = color_bucket
        category = record.get("final_label") or "unassigned"
        dest_dir = run_dir / color_bucket / category
        dest_dir.mkdir(parents=True, exist_ok=True)
        destination = dest_dir / original_path.name
        if destination.exists():
            destination = _resolve_collision(destination)
        shutil.move(str(original_path), destination)
        record["path"] = str(destination)
        record["relative_path"] = os.path.relpath(destination, run_dir)


def _write_assignment_csv(records: Sequence[dict[str, Any]], csv_path: Path) -> None:
    ensure_parent_dir(csv_path)
    rows: list[dict[str, Any]] = []
    for record in records:
        rows.append(
            {
                "filename": record.get("filename", ""),
                "path": record.get("path", ""),
                "final_label": record.get("final_label", ""),
                "color_mode": record.get("color_mode", ""),
            }
        )
    df = pd.DataFrame(rows, columns=["filename", "path", "final_label", "color_mode"])
    df.to_csv(csv_path, index=False)


@app.command()
def retrain() -> None:
    """Placeholder for future lightweight fine-tuning command."""

    raise typer.Exit("Retrain is not implemented yet.")


@app.command()
def eval(
    eval_dir: Path = typer.Argument(..., exists=True, file_okay=False, resolve_path=True),
    labels_csv: Optional[Path] = typer.Option(
        None,
        "--labels",
        exists=True,
        resolve_path=True,
        help="Optional CSV with filename,label (for reference only)",
    ),
    config_path: Path = typer.Option(Path("config.yaml"), "--config", exists=True, resolve_path=True),
    report_path: Path = typer.Option(Path("outputs/report.md"), "--report", resolve_path=True),
    model: Optional[str] = typer.Option(None, help="Override zero-shot backend for evaluation (siglip/openclip/both)"),
    checkpoint: Optional[str] = typer.Option(None, help="Override checkpoint identifier"),
    batch_size: Optional[int] = typer.Option(None, min=1),
    mode: Optional[str] = typer.Option(
        None,
        "--mode",
        "--gpt-mode",
        help="Decision mode: gpt-off | gpt-review | gptall",
    ),
    env_path: Optional[Path] = typer.Option(
        None,
        "--env-path",
        resolve_path=True,
        help="Path to alternate .env for API keys",
    ),
    gpt_batch_size: Optional[int] = typer.Option(None, "--gpt-batch-size", min=1, help="Number of images to process per GPT window"),
    gpt_rps: Optional[float] = typer.Option(None, "--gpt-rps", min=0.1, help="Max GPT requests per second"),
    gpt_max_retries: Optional[int] = typer.Option(None, "--gpt-max-retries", min=0, help="Maximum GPT retry attempts on rate limit"),
    gpt_backoff_base: Optional[float] = typer.Option(None, "--gpt-backoff-base", min=0.0, help="Base seconds for exponential backoff"),
    gpt_tpm_limit: Optional[int] = typer.Option(None, "--gpt-tpm-limit", min=1, help="Manual TPM guard (tokens per minute)"),
    gpt_rpm_limit: Optional[int] = typer.Option(None, "--gpt-rpm-limit", min=1, help="Manual RPM guard (requests per minute)"),
) -> None:
    """Run evaluation and export multi-model reports."""

    config = load_config(config_path)
    _apply_overrides(
        config,
        model,
        checkpoint,
        batch_size,
        None,
        None,
        None,
        None,
        None,
        None,
    )
    seed_everything(config.seed)

    ground_truth = _load_ground_truth(labels_csv) if labels_csv else {}
    all_image_paths = list_image_paths(eval_dir)
    processed_paths = [p for p in all_image_paths if p.suffix.lower() != ".bmp"]
    skipped_bmp_paths = [p for p in all_image_paths if p.suffix.lower() == ".bmp"]

    start_time = datetime.now()
    run_dir = Path("outputs") / start_time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    report_path = run_dir / Path(report_path).name
    json_path = run_dir / "pred.json"
    html_path = run_dir / "gallery.html"

    env_mode = _load_env_value("EVAL_MODE", env_path)
    input_mode = mode or env_mode or "gpt-off"

    env_model = _load_env_value("EVAL_MODEL", env_path)
    requested_model = (model or env_model or config.zero_shot.model).lower()
    valid_models = {"siglip", "openclip", "both"}
    if requested_model not in valid_models:
        raise typer.BadParameter(
            "--model は siglip / openclip / both のいずれかを指定してください。",
            param_hint="--model",
        )

    target_models: list[str]
    reference_models: list[str] = []
    if requested_model == "openclip":
        target_models = ["openclip"]
    elif requested_model == "both":
        target_models = ["openclip", "siglip"]
        reference_models = ["siglip"]
    else:  # requested_model == "siglip"
        typer.echo("SigLIP は参考出力として OpenCLIP と併走します。")
        target_models = ["openclip", "siglip"]
        reference_models = ["siglip"]

    predictions_by_model: dict[str, Sequence[ZeroShotResult]] = {}
    mapped_predictions: dict[str, dict[Path, ZeroShotResult]] = {}

    if processed_paths:
        for target in target_models:
            typer.echo(f"Running {target.upper()} evaluation…")
            cfg = _prepare_model_config(config, target)
            results = classify_zero_shot(cfg, processed_paths)
            if target == "openclip":
                for result in results:
                    runner_up = result.runner_up_score if result.runner_up_score is not None else 0.0
                    diff = max(float(result.score) - float(runner_up), 0.0)
                    result.margin = diff
                    result.needs_review = diff < OPENCLIP_REVIEW_THRESHOLD
            predictions_by_model[target] = results
            mapped_predictions[target] = _map_results_by_path(results)
    else:
        mapped_predictions = {name: {} for name in target_models}
    primary_model = "openclip"

    ensure_parent_dir(report_path)

    reference_notes = []
    if "siglip" in reference_models:
        reference_notes.append("SigLIPは参考出力のみ（精度集計には未反映）")

    needs_review_map = {
        result.path: result.needs_review for result in predictions_by_model.get(primary_model, [])
    }

    normalized_mode = input_mode.lower()
    if normalized_mode in {"off", ""}:
        normalized_mode = "gpt-off"
    valid_modes = {"gpt-off", "gpt-review", "gptall", "gpt-all", "review", "all"}
    if normalized_mode not in valid_modes:
        raise typer.BadParameter(
            "--mode は gpt-off / gpt-review / gptall のいずれかを指定してください。",
            param_hint="--mode",
        )

    effective_batch_size = gpt_batch_size if gpt_batch_size is not None else _load_env_int("GPT_BATCH_SIZE", env_path)
    if effective_batch_size is None:
        effective_batch_size = 8
    effective_rps = gpt_rps if gpt_rps is not None else _load_env_float("GPT_RPS", env_path)
    if effective_rps is None:
        effective_rps = 1.0
    effective_max_retries = gpt_max_retries if gpt_max_retries is not None else _load_env_int("GPT_MAX_RETRIES", env_path)
    if effective_max_retries is None:
        effective_max_retries = 6
    effective_backoff = gpt_backoff_base if gpt_backoff_base is not None else _load_env_float("GPT_BACKOFF_BASE", env_path)
    if effective_backoff is None:
        effective_backoff = 0.5
    effective_tpm = gpt_tpm_limit if gpt_tpm_limit is not None else _load_env_int("GPT_TPM_LIMIT", env_path)
    effective_rpm = gpt_rpm_limit if gpt_rpm_limit is not None else _load_env_int("GPT_RPM_LIMIT", env_path)

    gpt_decisions: dict[Path, GPTDecision] = {}
    gpt_cost = None
    if processed_paths and normalized_mode in {"gpt-review", "review"}:
        typer.echo("Running GPT adjudication (review mode)…")
        gpt_result = classify_with_gpt(
            processed_paths,
            list(config.classes.keys()),
            needs_review_map,
            "review",
            env_path,
            batch_size=effective_batch_size,
            rps_limit=effective_rps,
            max_retries=effective_max_retries,
            base_backoff=effective_backoff,
            tpm_limit=effective_tpm,
            rpm_limit=effective_rpm,
        )
        gpt_decisions = gpt_result.decisions
        gpt_cost = estimate_gpt_cost(gpt_result.usage)
    elif processed_paths and normalized_mode in {"gptall", "gpt-all", "all"}:
        typer.echo("Running GPT adjudication (all images)…")
        gpt_result = classify_with_gpt(
            processed_paths,
            list(config.classes.keys()),
            needs_review_map,
            "all",
            env_path,
            batch_size=effective_batch_size,
            rps_limit=effective_rps,
            max_retries=effective_max_retries,
            base_backoff=effective_backoff,
            tpm_limit=effective_tpm,
            rpm_limit=effective_rpm,
        )
        gpt_decisions = gpt_result.decisions
        gpt_cost = estimate_gpt_cost(gpt_result.usage)

    records = _gather_prediction_records(
        processed_paths,
        mapped_predictions,
        gpt_decisions,
        ground_truth,
        eval_dir,
        primary_model,
    )

    _apply_final_labels(records, normalized_mode, OPENCLIP_REVIEW_THRESHOLD)

    color_results = {
        str(path): judge_color_mode(path, config.color_judge) for path in processed_paths
    }

    _relocate_inputs(records, color_results, run_dir)

    warning_messages: Optional[list[str]] = None
    if skipped_bmp_paths:
        warning_messages = [
            f"BMPファイル {len(skipped_bmp_paths)} 件は未対応フォーマットのため処理されませんでした"
        ]
        for path in skipped_bmp_paths:
            relative = path.relative_to(eval_dir)
            warning_messages.append(str(relative))

    _render_assignment_summary(
        records,
        report_path,
        normalized_mode,
        OPENCLIP_REVIEW_THRESHOLD,
        gpt_cost,
        reference_notes,
        warning_messages,
    )

    _write_predictions_json(records, json_path)

    _write_assignment_csv(records, run_dir / "summary.csv")

    _render_gallery_html(records, html_path, run_dir, mapped_predictions.keys(), config.review.review_label)

    total_images = len(records)
    if skipped_bmp_paths:
        typer.echo(f"Skipped {len(skipped_bmp_paths)} BMP image(s); left in input directory.")
    gpt_applied = sum(1 for rec in records if rec.get("final_source") == "gpt")
    typer.echo(f"Processed {total_images} images (mode={normalized_mode}, GPT applied: {gpt_applied}).")
    if gpt_cost:
        typer.echo(f"GPT cost estimate: ${gpt_cost.cost_usd:.5f} ≈ {gpt_cost.cost_jpy:.2f}円")

    typer.echo(f"Report written to {report_path}")
    typer.echo(f"Predictions JSON written to {json_path}")
    typer.echo(f"Gallery HTML written to {html_path}")
    typer.echo(f"Images moved under {run_dir}")


@app.command("pixabay-download")
def pixabay_download(
    query: str = typer.Option("", "--query", "-q", help="Search query sent as q parameter"),
    preset: str = typer.Option("background", "--preset", "-p", help="Preset to seed request: background/icon/item"),
    out_dir: Path = typer.Option(
        Path("images/pixabay"), "--out-dir", file_okay=False, resolve_path=True, help="Directory to store downloads"
    ),
    limit: int = typer.Option(20, "--limit", "-n", min=1, max=500, help="Number of files to download from the response"),
    page: int = typer.Option(1, "--page", min=1, help="Starting page number"),
    per_page: int = typer.Option(40, "--per-page", min=1, max=200, help="Results per request (Pixabay max 200)"),
    order: str = typer.Option("popular", "--order", help="Result ordering: popular or latest"),
    image_type: Optional[str] = typer.Option(None, "--image-type", help="Override image_type parameter"),
    orientation: Optional[str] = typer.Option(None, "--orientation", help="Override orientation parameter"),
    colors: Optional[str] = typer.Option(None, "--colors", help="Restrict colors (e.g. transparent, red)"),
    category: Optional[str] = typer.Option(None, "--category", help="Pixabay category override"),
    min_width: Optional[int] = typer.Option(None, "--min-width", min=0),
    min_height: Optional[int] = typer.Option(None, "--min-height", min=0),
    safesearch: Optional[bool] = typer.Option(
        None, "--safesearch/--no-safesearch", help="Explicitly toggle Pixabay safesearch"
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show resolved parameters without hitting the API"),
    env_path: Optional[Path] = typer.Option(None, "--env-path", resolve_path=True, help="Path to alternate .env"),
) -> None:
    """Download sample media from Pixabay into the project workspace."""

    order_value = order.lower()
    if order_value not in {"popular", "latest"}:
        raise typer.BadParameter("order must be either 'popular' or 'latest'", param_hint="--order")
    overrides: dict[str, Any] = {}
    if image_type is not None:
        overrides["image_type"] = image_type
    if orientation is not None:
        overrides["orientation"] = orientation
    if colors is not None:
        overrides["colors"] = colors
    if category is not None:
        overrides["category"] = category
    if min_width is not None:
        overrides["min_width"] = min_width
    if min_height is not None:
        overrides["min_height"] = min_height
    if safesearch is not None:
        overrides["safesearch"] = "true" if safesearch else "false"
    params = build_params(preset, query, page, per_page, order_value, overrides)
    if dry_run:
        typer.echo("Pixabay request preview:")
        for key in sorted(params):
            typer.echo(f" - {key}={params[key]}")
        return

    api_key = load_api_key(env_path)
    client = PixabayClient(api_key)
    typer.echo("Fetching Pixabay search results…")
    try:
        response = client.search(params)
    except Exception as err:  # noqa: BLE001 - convert to CLI-friendly message
        typer.echo(f"Pixabay search failed: {err}")
        raise typer.Exit(code=1)
    hits = response.get("hits", [])
    total_hits = response.get("totalHits", 0)
    if not hits:
        typer.echo("No hits returned. Adjust your query or preset.")
        return

    typer.echo(f"Retrieved {len(hits)} hits (totalHits={total_hits}). Saving to {out_dir}.")
    try:
        downloads = client.download_hits(hits, out_dir, limit=limit)
    except Exception as err:  # noqa: BLE001 - convert to CLI-friendly message
        typer.echo(f"Download failed: {err}")
        raise typer.Exit(code=1)
    metadata_path = save_metadata(downloads, out_dir)
    typer.echo(f"Downloaded {len(downloads)} file(s). Metadata written to {metadata_path}.")
    if total_hits > len(downloads):
        typer.echo("Additional results available. Use --page and --per-page to fetch more.")


@app.command("attr-poc")
def attr_poc(
    input_dir: Path = typer.Argument(..., exists=True, file_okay=False, resolve_path=True),
    env_path: Optional[Path] = typer.Option(None, "--env-path", resolve_path=True, help="Path to alternate .env"),
    limit: int = typer.Option(100, min=1, help="Maximum number of images to process"),
    offset: int = typer.Option(0, min=0, help="Number of images to skip from the start"),
    out_dir: Path = typer.Option(Path("outputs/attr_poc"), "--out-dir", resolve_path=True),
    sleep: float = typer.Option(0.0, min=0.0, help="Seconds to sleep between images"),
    qwen_model: Optional[str] = typer.Option(None, help="Override Qwen model identifier"),
    gemma_model: Optional[str] = typer.Option(None, help="Override Gemma model identifier"),
    qwen_base_url: Optional[str] = typer.Option(None, help="Override Qwen server base URL"),
    gemma_base_url: Optional[str] = typer.Option(None, help="Override Gemma server base URL"),
    use_qwen: bool = typer.Option(True, "--use-qwen/--no-use-qwen", help="Include Qwen2.5-VL in extraction"),
    use_gemma: bool = typer.Option(True, "--use-gemma/--no-use-gemma", help="Include Gemma-3 in extraction"),
) -> None:
    """Run multi-model attribute extraction PoC and build an HTML comparison report."""

    from openai import OpenAI

    def _client_from_env(prefix: str, override_base: Optional[str]) -> tuple[OpenAI, str]:
        base_url = (
            override_base
            or _load_env_value(f"{prefix}_BASE_URL", env_path)
            or _load_env_value("LMSTUDIO_BASE_URL", env_path)
            or "http://127.0.0.1:1234/v1"
        )
        api_key = _load_env_value(f"{prefix}_API_KEY", env_path) or _load_env_value("LMSTUDIO_API_KEY", env_path) or "lm-studio"
        return OpenAI(base_url=base_url, api_key=api_key), base_url

    info_lines: list[str] = []
    model_specs: list[ModelSpec] = []

    if use_qwen:
        qwen_client, qwen_url = _client_from_env("QWEN", qwen_base_url)
        effective_qwen_model = (
            qwen_model
            or _load_env_value("QWEN_MODEL", env_path)
            or _load_env_value("LMSTUDIO_QWEN_MODEL", env_path)
            or "qwen/qwen2.5-vl-7b"
        )
        info_lines.append(f" - Qwen model: {effective_qwen_model} @ {qwen_url}")
        model_specs.append(
            ModelSpec(key="qwen", label="Qwen2.5-VL", client=qwen_client, model=effective_qwen_model)
        )

    if use_gemma:
        gemma_client, gemma_url = _client_from_env("GEMMA", gemma_base_url)
        effective_gemma_model = (
            gemma_model
            or _load_env_value("GEMMA_MODEL", env_path)
            or _load_env_value("LMSTUDIO_GEMMA_MODEL", env_path)
            or "google/gemma-3-12b"
        )
        info_lines.append(f" - Gemma model: {effective_gemma_model} @ {gemma_url}")
        model_specs.append(
            ModelSpec(key="gemma", label="Gemma-3", client=gemma_client, model=effective_gemma_model)
        )

    if not model_specs:
        raise typer.BadParameter(
            "少なくとも1つのモデルを有効にしてください。",
            param_hint="--use-qwen/--use-gemma",
        )

    info_lines.append(f" - Images: {input_dir} (offset={offset}, limit={limit})")
    typer.echo("Running attribute extraction PoC with:\n" + "\n".join(info_lines))

    try:
        records = run_attribute_extraction(
            input_dir,
            model_specs,
            limit=limit,
            offset=offset,
            sleep=sleep,
        )
    except FileNotFoundError as err:
        raise typer.BadParameter(str(err), param_hint="input_dir") from err

    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "attr_results.json"
    html_path = out_dir / "attr_report.html"

    typer.echo(f"Saving structured outputs to {json_path}")
    json_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

    typer.echo(f"Rendering HTML comparison to {html_path}")
    generate_html_report(records, html_path)

    typer.echo("Done. Open the HTML report in a browser to review model outputs.")


@app.command("attr-index")
def attr_index(
    attr_json: Path = typer.Argument(..., exists=True, dir_okay=False, resolve_path=True),
    index_dir: Path = typer.Option(Path("outputs/vector_index"), "--index-dir", resolve_path=True),
    env_path: Optional[Path] = typer.Option(None, "--env-path", resolve_path=True, help="Path to alternate .env"),
    text_model: Optional[str] = typer.Option(None, help="Embedding model identifier"),
    text_base_url: Optional[str] = typer.Option(None, help="Embedding server base URL"),
    text_api_key: Optional[str] = typer.Option(None, help="Embedding server API key"),
    image_model: str = typer.Option("ViT-B-32", help="OpenCLIP image encoder"),
    image_pretrained: str = typer.Option("openai", help="Pretrained weights name"),
    device: str = typer.Option("cpu", help="Torch device for image encoder"),
    reset: bool = typer.Option(False, help="Reset index directory before ingest"),
    attr_key: List[str] = typer.Option(
        None,
        "--attr-key",
        help="Attribute key(s) to ingest (repeatable). Default: all",
        show_default=False,
    ),
    thumbnail_size: int = typer.Option(64, help="Max edge length for generated thumbnails"),
    no_thumbnails: bool = typer.Option(False, help="Skip thumbnail generation"),
    save_attr_embeddings: bool = typer.Option(
        True,
        "--save-attr-embeddings/--no-save-attr-embeddings",
        help="Persist attr embeddings in compressed .npz format",
    ),
) -> None:
    """Build FAISS + SQLite indices from attr_poc JSON results."""

    embed_base_url = (
        text_base_url
        or _load_env_value("EMBED_BASE_URL", env_path)
        or _load_env_value("LMSTUDIO_BASE_URL", env_path)
        or "http://127.0.0.1:1234/v1"
    )
    embed_api_key = (
        text_api_key or _load_env_value("EMBED_API_KEY", env_path) or _load_env_value("LMSTUDIO_API_KEY", env_path) or "lm-studio"
    )
    model_name = (
        text_model
        or _load_env_value("EMBED_MODEL", env_path)
        or _load_env_value("LMSTUDIO_EMBED_MODEL", env_path)
        or "text-embedding-nomic-embed-text-v1.5"
    )

    client = create_text_client(embed_base_url, embed_api_key)
    typer.echo(
        "Building attribute index with:\n"
        f" - attr JSON: {attr_json}\n"
        f" - index dir: {index_dir}\n"
        f" - text model: {model_name}\n"
        f" - image encoder: {image_model}/{image_pretrained} (device={device})"
    )
    if attr_key:
        typer.echo(" - attr keys: " + ", ".join(attr_key))
    if no_thumbnails:
        typer.echo(" - thumbnails: disabled")
    else:
        typer.echo(f" - thumbnail size: {thumbnail_size}")
    typer.echo(f" - save attr embeddings: {'yes' if save_attr_embeddings else 'no'}")

    records = load_attr_records(attr_json)
    if not records:
        typer.echo("No records present in attr JSON. Nothing to do.")
        return

    if attr_key:
        normalised_keys: list[str] = []
        allowed = {key: key for key in ATTR_KEYS}
        for key in attr_key:
            candidate = key.strip()
            if candidate not in allowed:
                raise typer.BadParameter(
                    f"Unsupported attr key '{key}'. Choose from: {', '.join(ATTR_KEYS)}",
                    param_hint="--attr-key",
                )
            normalised_keys.append(allowed[candidate])
        selected_keys: Sequence[str] = tuple(normalised_keys)
    else:
        selected_keys = ATTR_KEYS

    if thumbnail_size <= 0:
        raise typer.BadParameter("thumbnail-size must be >= 1", param_hint="--thumbnail-size")

    indexer = AttributeIndexer(
        IndexPaths(index_dir),
        text_client=client,
        text_model=model_name,
        image_model=image_model,
        image_pretrained=image_pretrained,
        device=device,
        thumbnail_size=thumbnail_size,
        generate_thumbnails=not no_thumbnails,
        save_embeddings=save_attr_embeddings,
    )

    if reset:
        typer.echo("Resetting index directory…")
        indexer.reset()

    typer.echo(f"Ingesting {len(records)} records…")
    indexer.ingest_records(records, attr_keys=selected_keys)
    typer.echo("Index build completed.")


@app.command("attr-serve")
def attr_serve(
    index_dir: Path = typer.Option(Path("outputs/vector_index"), "--index-dir", resolve_path=True),
    env_path: Optional[Path] = typer.Option(None, "--env-path", resolve_path=True, help="Path to alternate .env"),
    text_model: Optional[str] = typer.Option(None, help="Embedding model identifier"),
    text_base_url: Optional[str] = typer.Option(None, help="Embedding server base URL"),
    text_api_key: Optional[str] = typer.Option(None, help="Embedding server API key"),
    host: str = typer.Option("127.0.0.1", help="Host to bind"),
    port: int = typer.Option(8000, help="Port to bind"),
    reload: bool = typer.Option(False, help="Enable auto-reload (development only)"),
) -> None:
    """Launch the attribute search web UI."""

    embed_base_url = (
        text_base_url
        or _load_env_value("EMBED_BASE_URL", env_path)
        or _load_env_value("LMSTUDIO_BASE_URL", env_path)
        or "http://127.0.0.1:1234/v1"
    )
    embed_api_key = (
        text_api_key or _load_env_value("EMBED_API_KEY", env_path) or _load_env_value("LMSTUDIO_API_KEY", env_path) or "lm-studio"
    )
    model_name = (
        text_model
        or _load_env_value("EMBED_MODEL", env_path)
        or _load_env_value("LMSTUDIO_EMBED_MODEL", env_path)
        or "text-embedding-nomic-embed-text-v1.5"
    )

    typer.echo(
        "Starting attribute search UI with:\n"
        f" - index dir: {index_dir}\n"
        f" - text model: {model_name}\n"
        f" - base URL: {embed_base_url}\n"
        f" - host: {host}:{port}"
    )

    app_instance = create_app(
        index_dir,
        embed_base_url=embed_base_url,
        embed_api_key=embed_api_key,
        embed_model=model_name,
    )
    uvicorn.run(app_instance, host=host, port=port, reload=reload)


@app.command("cluster-precompute")
def cluster_precompute_cli(
    index_dir: Path = typer.Option(Path("outputs/vector_index"), "--index-dir", resolve_path=True),
    env_path: Optional[Path] = typer.Option(None, "--env-path", resolve_path=True, help="Path to alternate .env"),
    mode: List[str] = typer.Option([], "--mode", "-m", help="対象モード（location/subject/image）。省略時はすべて。"),
    limit: Optional[int] = typer.Option(None, "--limit", help="先頭からの上限件数。省略時は全件。"),
    n_neighbors: int = typer.Option(30, help="UMAP の近傍数"),
    min_dist: float = typer.Option(0.15, help="UMAP の min_dist"),
    metric: str = typer.Option("cosine", help="UMAP の距離関数"),
    random_state: int = typer.Option(42, help="UMAP の乱数シード"),
    min_samples: int = typer.Option(10, help="DBSCAN の min_samples"),
    eps: Optional[float] = typer.Option(None, help="DBSCAN の eps。省略時は自動推定"),
    eps_percentile: float = typer.Option(90.0, help="eps自動推定時のパーセンタイル"),
    text_model: Optional[str] = typer.Option(None, help="テキスト埋め込みモデル名"),
    text_base_url: Optional[str] = typer.Option(None, help="埋め込みAPIのベースURL"),
    text_api_key: Optional[str] = typer.Option(None, help="埋め込みAPIのキー"),
) -> None:
    """UMAP + DBSCAN の前処理を実行してクラスタマップ用データを生成する。"""

    embed_base_url = (
        text_base_url
        or _load_env_value("EMBED_BASE_URL", env_path)
        or _load_env_value("LMSTUDIO_BASE_URL", env_path)
        or "http://127.0.0.1:1234/v1"
    )
    embed_api_key = (
        text_api_key or _load_env_value("EMBED_API_KEY", env_path) or _load_env_value("LMSTUDIO_API_KEY", env_path) or "lm-studio"
    )
    model_name = (
        text_model
        or _load_env_value("EMBED_MODEL", env_path)
        or _load_env_value("LMSTUDIO_EMBED_MODEL", env_path)
        or "text-embedding-nomic-embed-text-v1.5"
    )

    selected_modes = [item.lower() for item in mode if item.strip()]
    if not selected_modes:
        selected_modes = list(CLUSTER_ALLOWED_MODES)
    invalid = sorted(set(selected_modes) - set(CLUSTER_ALLOWED_MODES))
    if invalid:
        raise typer.BadParameter(
            f"サポートされていないモード: {', '.join(invalid)}",
            param_hint="--mode",
        )

    config = ClusterJobConfig(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
        min_samples=min_samples,
        eps=eps,
        eps_percentile=eps_percentile,
        limit=limit,
    )

    typer.echo(
        "クラスタ前処理を実行します:\n"
        f" - index dir: {index_dir}\n"
        f" - modes: {', '.join(selected_modes)}\n"
        f" - UMAP: n_neighbors={config.n_neighbors}, min_dist={config.min_dist}, metric={config.metric}, seed={config.random_state}\n"
        f" - DBSCAN: min_samples={config.min_samples}, eps={'auto' if config.eps is None else config.eps} (percentile={config.eps_percentile})"
        + (f"\n - limit: {config.limit}" if config.limit else "")
    )

    indexer = AttributeIndexer(
        IndexPaths(index_dir),
        text_client=create_text_client(embed_base_url, embed_api_key),
        text_model=model_name,
    )

    for mode_name in selected_modes:
        typer.echo(f"\n[{mode_name}] 前処理を実行中…", nl=False)
        try:
            artifacts = precompute_cluster(indexer, mode_name, config)
        except Exception as err:  # noqa: BLE001 - CLIでユーザーに見せるためまとめて捕捉
            typer.echo(" 失敗")
            raise typer.BadParameter(str(err), param_hint="--mode") from err

        save_artifacts(indexer.paths.cluster_mode_dir(mode_name), artifacts)
        typer.echo(" 完了")
        typer.echo(
            f"  - 総件数 {artifacts.meta['total']} 件 / クラスタ {artifacts.meta['cluster_count']} / ノイズ {artifacts.meta['noise_count']}"
        )


def main() -> None:
    """Entry point for console_scripts."""

    app()


if __name__ == "__main__":
    main()
