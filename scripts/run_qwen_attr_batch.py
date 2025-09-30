#!/usr/bin/env python3
"""Resumable Qwen-only attribute extraction with progress tracking."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from photocat.attr_poc import ModelSpec, _call_model, _encode_image
from photocat.cli import _load_env_value
from photocat.utils import ensure_parent_dir, list_image_paths


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def _append_jsonl(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _write_json(path: Path, data: Iterable[dict]) -> None:
    ensure_parent_dir(path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(list(data), handle, ensure_ascii=False, indent=2)


def _write_processed_list(path: Path, processed: Iterable[str]) -> None:
    ensure_parent_dir(path)
    with path.open("w", encoding="utf-8") as handle:
        for item in sorted(processed):
            handle.write(f"{item}\n")


def _write_targets(path: Path, targets: Iterable[str]) -> None:
    ensure_parent_dir(path)
    with path.open("w", encoding="utf-8") as handle:
        for item in targets:
            handle.write(f"{item}\n")


def _write_status(path: Path, all_images: list[str], target_set: set[str], processed_set: set[str]) -> None:
    ensure_parent_dir(path)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("image_path,target,processed\n")
        for image in all_images:
            target = "yes" if image in target_set else "no"
            processed = "yes" if image in processed_set else "no"
            handle.write(f"{image},{target},{processed}\n")


def _create_qwen_spec(
    *,
    env_path: Optional[Path],
    model_name: Optional[str],
    base_url: Optional[str],
    api_key: Optional[str],
) -> tuple[ModelSpec, str]:
    resolved_base_url = (
        base_url
        or _load_env_value("QWEN_BASE_URL", env_path)
        or _load_env_value("LMSTUDIO_BASE_URL", env_path)
        or "http://127.0.0.1:1234/v1"
    )
    resolved_api_key = (
        api_key
        or _load_env_value("QWEN_API_KEY", env_path)
        or _load_env_value("LMSTUDIO_API_KEY", env_path)
        or "lm-studio"
    )
    resolved_model = (
        model_name
        or _load_env_value("QWEN_MODEL", env_path)
        or _load_env_value("LMSTUDIO_QWEN_MODEL", env_path)
        or "qwen/qwen2.5-vl-7b"
    )
    client = OpenAI(base_url=resolved_base_url, api_key=resolved_api_key)
    spec = ModelSpec(key="qwen", label="Qwen2.5-VL", client=client, model=resolved_model)
    return spec, resolved_base_url


def run_batch(args: argparse.Namespace) -> None:
    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    env_path = Path(args.env_path).resolve() if args.env_path else None

    all_images = [str(path) for path in list_image_paths(input_dir)]
    if args.limit > len(all_images):
        raise ValueError(f"limit {args.limit} exceeds available images {len(all_images)}")

    targets = all_images[: args.limit]
    target_set = set(targets)

    jsonl_path = out_dir / "attr_results.jsonl"
    json_path = out_dir / "attr_results.json"
    processed_list_path = out_dir / "processed_images.txt"
    targets_path = out_dir / "targets.txt"
    status_path = out_dir / "status.csv"
    log_path = out_dir / "run.log"

    out_dir.mkdir(parents=True, exist_ok=True)

    _write_targets(targets_path, targets)

    existing_records = _load_jsonl(jsonl_path)
    processed_records = list(existing_records)
    processed_set = {record.get("image_path") for record in processed_records if record.get("image_path")}
    processed_set.discard(None)

    pending = [image for image in targets if image not in processed_set]
    if args.max_new is not None:
        pending = pending[: args.max_new]

    spec, resolved_base_url = _create_qwen_spec(
        env_path=env_path,
        model_name=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
    )

    with log_path.open("a", encoding="utf-8") as log_handle:
        log_handle.write(
            f"[{_iso_now()}] Starting run: total={len(targets)} already={len(processed_set)} pending={len(pending)} base_url={resolved_base_url}\n"
        )

    if not pending:
        _write_status(status_path, all_images, target_set, processed_set)
        _write_processed_list(processed_list_path, processed_set)
        _write_json(json_path, processed_records)
        return

    for image_path in pending:
        path_obj = Path(image_path)
        started_at = _iso_now()
        model_output = _call_model(path_obj, spec)
        model_output["model_label"] = spec.label
        image_b64, image_mime = _encode_image(path_obj)
        record = {
            "image_path": image_path,
            spec.key: model_output,
            "image_base64": image_b64,
            "image_mime": image_mime,
            "processed_at": started_at,
        }
        processed_records.append(record)
        processed_set.add(image_path)
        _append_jsonl(jsonl_path, record)
        _write_json(json_path, processed_records)
        _write_processed_list(processed_list_path, processed_set)
        _write_status(status_path, all_images, target_set, processed_set)
        with log_path.open("a", encoding="utf-8") as log_handle:
            log_handle.write(f"[{_iso_now()}] processed {image_path}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run resumable Qwen attribute extraction")
    parser.add_argument("input_dir", help="Directory that contains images")
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Number of images (in sorted order) to target",
    )
    parser.add_argument(
        "--max-new",
        type=int,
        default=None,
        help="Maximum number of new images to process on this run",
    )
    parser.add_argument(
        "--out-dir",
        default="outputs/attr_poc_qwen",
        help="Directory for outputs and progress files",
    )
    parser.add_argument(
        "--env-path",
        default=None,
        help="Optional custom .env path",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override Qwen model identifier",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Override Qwen server base URL",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Override Qwen API key",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_batch(args)


if __name__ == "__main__":
    main()
