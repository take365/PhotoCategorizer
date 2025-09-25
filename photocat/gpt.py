"""GPT-based classification helpers."""

from __future__ import annotations

import base64
import json
import os
import random
import re
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from collections import deque
from typing import Iterable, Mapping, MutableMapping, Optional

from PIL import Image
from openai import APIError, OpenAI, RateLimitError
from typer import BadParameter

__all__ = [
    "GPTDecision",
    "GPTBatchResult",
    "GPTUsageSummary",
    "GPTCostEstimate",
    "classify_with_gpt",
    "estimate_gpt_cost",
]


@dataclass(slots=True)
class GPTDecision:
    """Represents a GPT-based classification decision."""

    label: str
    confidence: float | None
    raw_response: str


@dataclass(slots=True)
class GPTUsageSummary:
    """Aggregated token usage for a batch of GPT calls."""

    model: str
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


@dataclass(slots=True)
class GPTBatchResult:
    """Container combining GPT decisions and token usage."""

    decisions: dict[Path, GPTDecision]
    usage: GPTUsageSummary


@dataclass(slots=True)
class GPTCostEstimate:
    """Computed cost for GPT usage."""

    model: str
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_usd: float
    cost_jpy: float


_PRICING_TABLE: dict[str, tuple[float, float, float]] = {
    "gpt-4o-mini": (0.15, 0.075, 0.60),  # input, cached input, output ($ / 1M tokens)
    "gpt-5-mini": (0.25, 0.025, 2.00),
    "gpt-5": (1.25, 0.125, 10.00),
}


def _load_env_value(name: str, env_path: Path | None) -> str | None:
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


def _load_int_env(name: str, env_path: Path | None) -> int | None:
    raw = _load_env_value(name, env_path)
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _ensure_openai_settings(env_path: Path | None) -> tuple[str, str]:
    api_key = _load_env_value("OPENAI_API_KEY", env_path)
    if not api_key:
        raise BadParameter("OPENAI_API_KEY が見つかりません。--gpt-mode を使う場合は設定してください。")
    model = _load_env_value("BATCH_MODEL", env_path) or "gpt-4o-mini"
    return api_key, model


def _encode_image(path: Path) -> tuple[str, str]:
    with Image.open(path) as image:
        image = image.convert("RGB")
        image.thumbnail((1024, 1024))
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=85)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return encoded, "image/jpeg"


def _parse_response(labels: Iterable[str], text: str) -> tuple[str | None, float | None]:
    cleaned = text.strip()
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        json_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if json_match:
            try:
                payload = json.loads(json_match.group(0))
            except json.JSONDecodeError:
                payload = {}
        else:
            payload = {}
    label_candidates = [str(lbl).strip() for lbl in labels]
    chosen_label = None
    if isinstance(payload, Mapping):
        raw = payload.get("label")
        if isinstance(raw, str):
            chosen_label = _normalise_label(raw, label_candidates)
        confidence = payload.get("confidence")
        try:
            conf_value = float(confidence)
        except (TypeError, ValueError):
            conf_value = None
    else:
        conf_value = None
    if not chosen_label:
        for candidate in label_candidates:
            pattern = re.compile(rf"\b{re.escape(candidate)}\b", re.IGNORECASE)
            if pattern.search(cleaned):
                chosen_label = candidate
                break
    return chosen_label, conf_value


def _normalise_label(raw: str, candidates: Iterable[str]) -> str | None:
    lowered = raw.strip().lower()
    for candidate in candidates:
        if lowered == candidate.lower():
            return candidate
    return None


def classify_with_gpt(
    image_paths: Iterable[Path],
    labels: list[str],
    needs_review_map: Mapping[Path, bool],
    mode: str,
    env_path: Path | None,
    *,
    batch_size: int = 8,
    rps_limit: float = 1.0,
    max_retries: int = 6,
    base_backoff: float = 0.5,
    tpm_limit: int | None = None,
    rpm_limit: int | None = None,
) -> GPTBatchResult:
    mode_key = mode.lower()
    if mode_key not in {"all", "review"}:
        raise BadParameter("--gpt-mode は 'all' か 'review' を指定してください。")
    api_key, model = _ensure_openai_settings(env_path)
    client = OpenAI(api_key=api_key)

    batch_size = max(1, batch_size)
    rps_limit = max(0.0, rps_limit)
    max_retries = max(0, max_retries)
    base_backoff = base_backoff if base_backoff > 0 else 0.1

    if tpm_limit is None:
        tpm_limit = _load_int_env("GPT_TPM_LIMIT", env_path)
    if rpm_limit is None:
        rpm_limit = _load_int_env("GPT_RPM_LIMIT", env_path)

    prompt_text = (
        "You are a photo classification assistant. Choose the single best matching label "
        "from the following list and respond with JSON: {\"label\": <label>, \"confidence\": <0-1 float>}. "
        "Labels: " + ", ".join(labels)
    )

    usage = GPTUsageSummary(model=model)
    decisions: dict[Path, GPTDecision] = {}

    targets: list[Path] = []
    for path in image_paths:
        if mode_key == "review" and not needs_review_map.get(path, False):
            continue
        targets.append(path)

    if not targets:
        return GPTBatchResult(decisions=decisions, usage=usage)

    request_history: deque[float] = deque()
    token_history: deque[tuple[float, int]] = deque()
    current_tokens = 0

    EST_PROMPT_TOKENS = 900
    EST_COMPLETION_TOKENS = 200

    def _prune_histories(now: float) -> None:
        nonlocal current_tokens
        while request_history and now - request_history[0] >= 60:
            request_history.popleft()
        while token_history and now - token_history[0][0] >= 60:
            _, tokens = token_history.popleft()
            current_tokens = max(0, current_tokens - tokens)

    def _wait_for_limits(estimated_tokens: int) -> None:
        nonlocal current_tokens
        while True:
            now = time.time()
            _prune_histories(now)
            waits: list[float] = []
            if rpm_limit and rpm_limit > 0 and len(request_history) >= rpm_limit:
                waits.append(60 - (now - request_history[0]))
            if tpm_limit and tpm_limit > 0 and current_tokens + estimated_tokens > tpm_limit:
                if token_history:
                    waits.append(60 - (now - token_history[0][0]))
            if rps_limit and rps_limit > 0 and request_history:
                interval = 1.0 / rps_limit
                waits.append(request_history[-1] + interval - now)
            waits = [w for w in waits if w and w > 0]
            if not waits:
                break
            time.sleep(min(max(waits), 60.0))

    def _register_tokens(tokens: int) -> None:
        nonlocal current_tokens
        if tokens <= 0:
            return
        token_history.append((time.time(), tokens))
        current_tokens += tokens

    def _extract_text_from_response(response) -> str:
        output_text = getattr(response, "output_text", None)
        if output_text:
            return output_text.strip()
        output = getattr(response, "output", None) or getattr(response, "outputs", None)
        if output and isinstance(output, list):
            pieces: list[str] = []
            for item in output:
                content = getattr(item, "content", None)
                if not content:
                    continue
                if isinstance(content, list):
                    for chunk in content:
                        text_val = getattr(chunk, "text", None)
                        if text_val:
                            pieces.append(str(text_val))
                else:
                    text_val = getattr(content, "text", None)
                    if text_val:
                        pieces.append(str(text_val))
            if pieces:
                return "\n".join(pieces).strip()
        return ""

    def _extract_total_tokens(response) -> int:
        usage_block = getattr(response, "usage", None)
        if usage_block is None:
            return 0
        total = getattr(usage_block, "total_tokens", None)
        if total is None:
            prompt = getattr(usage_block, "prompt_tokens", 0)
            completion = getattr(usage_block, "completion_tokens", 0)
            total = (prompt or 0) + (completion or 0)
        return int(total or 0)

    def _extract_retry_after(error: Exception) -> Optional[float]:
        header_sources = []
        if hasattr(error, "response") and getattr(error, "response") is not None:
            header_sources.append(getattr(error.response, "headers", None))
        header_sources.append(getattr(error, "headers", None))
        for headers in header_sources:
            if not headers:
                continue
            for key in ("Retry-After", "retry-after"):
                if key in headers:
                    try:
                        return float(headers[key])
                    except (TypeError, ValueError):
                        try:
                            return float(str(headers[key]).strip("s"))
                        except (TypeError, ValueError):
                            continue
        message = str(getattr(error, "message", "") or error)
        match = re.search(r"in (\d+(?:\.\d+)?)\s*ms", message)
        if match:
            return float(match.group(1)) / 1000.0
        match = re.search(r"in (\d+(?:\.\d+)?)\s*s", message)
        if match:
            return float(match.group(1))
        return None

    def _process_single(path: Path) -> tuple[bool, bool]:
        rate_limit_hit = False
        retries = 0
        image_b64, mime = _encode_image(path)
        content_payload = [
            {"type": "input_text", "text": prompt_text},
            {
                "type": "input_image",
                "detail": "auto",
                "image_url": f"data:{mime};base64,{image_b64}",
            },
        ]
        use_completion_tokens_param = False
        while True:
            estimated_tokens = EST_PROMPT_TOKENS + EST_COMPLETION_TOKENS
            _wait_for_limits(estimated_tokens)
            request_kwargs = {
                "model": model,
                "input": [
                    {
                        "role": "user",
                        "content": content_payload,
                    }
                ],
            }
            if use_completion_tokens_param:
                request_kwargs["max_completion_tokens"] = 200
            else:
                request_kwargs["max_output_tokens"] = 200
            try:
                response = client.responses.create(**request_kwargs)
                request_history.append(time.time())
            except RateLimitError as err:
                request_history.append(time.time())
                rate_limit_hit = True
                wait_time = _extract_retry_after(err)
                if wait_time is None:
                    wait_time = min(base_backoff * (2 ** retries), 30.0)
                wait_time = min(wait_time + random.uniform(0.0, 0.25), 30.0)
                time.sleep(max(wait_time, 0.01))
                retries += 1
                if retries > max_retries:
                    decisions[path] = GPTDecision(label="", confidence=None, raw_response=str(err))
                    return False, True
                continue
            except APIError as err:
                request_history.append(time.time())
                error_message = str(getattr(err, "message", "") or err)
                if not use_completion_tokens_param and "max_output_tokens" in error_message:
                    use_completion_tokens_param = True
                    continue
                decisions[path] = GPTDecision(label="", confidence=None, raw_response=str(err))
                return False, rate_limit_hit
            except Exception as err:  # noqa: BLE001
                request_history.append(time.time())
                decisions[path] = GPTDecision(label="", confidence=None, raw_response=str(err))
                return False, rate_limit_hit

            _accumulate_usage(usage, getattr(response, "usage", None))
            total_tokens = _extract_total_tokens(response)
            _register_tokens(total_tokens)

            output_text = _extract_text_from_response(response)
            label, confidence = _parse_response(labels, output_text)
            decisions[path] = GPTDecision(
                label=label or "",
                confidence=confidence,
                raw_response=output_text,
            )
            return True, rate_limit_hit

    current_batch_size = batch_size
    index = 0
    total = len(targets)
    while index < total:
        subset = targets[index : index + current_batch_size]
        rate_limit_encountered = False
        for path in subset:
            _, rate_flag = _process_single(path)
            if rate_flag:
                rate_limit_encountered = True
        if rate_limit_encountered and current_batch_size > 1:
            current_batch_size = max(1, current_batch_size // 2)
        index += len(subset)

    return GPTBatchResult(decisions=decisions, usage=usage)

def _accumulate_usage(summary: GPTUsageSummary, usage_data: Optional[object]) -> None:
    if usage_data is None:
        return

    def _get(obj: object, key: str, default: int = 0) -> int:
        if isinstance(obj, MutableMapping):
            value = obj.get(key, default)
            return int(value) if value is not None else default
        if hasattr(obj, key):
            value = getattr(obj, key)
            return int(value) if value is not None else default
        return default

    input_tokens = _get(usage_data, "input_tokens", _get(usage_data, "prompt_tokens", 0))
    output_tokens = _get(usage_data, "output_tokens", _get(usage_data, "completion_tokens", 0))
    cached_tokens = _get(usage_data, "input_cached_tokens")
    if cached_tokens == 0:
        details = _get_details(usage_data, "prompt_tokens_details")
        cached_tokens = _get(details, "cached_tokens") if details is not None else 0
    total_tokens = _get(usage_data, "total_tokens", input_tokens + output_tokens)

    summary.input_tokens += input_tokens
    summary.cached_input_tokens += cached_tokens
    summary.output_tokens += output_tokens
    summary.total_tokens += total_tokens


def _get_details(obj: object, key: str) -> Optional[object]:
    if isinstance(obj, MutableMapping):
        return obj.get(key)
    return getattr(obj, key, None)


def estimate_gpt_cost(
    usage: GPTUsageSummary,
    usd_to_jpy: float = 150.0,
) -> GPTCostEstimate | None:
    pricing = _PRICING_TABLE.get(usage.model.lower())
    if pricing is None:
        return None
    input_rate, cached_rate, output_rate = pricing
    unit = 1_000_000
    billable_input = max(usage.input_tokens - usage.cached_input_tokens, 0)
    cost_input = billable_input * input_rate / unit
    cost_cached = usage.cached_input_tokens * cached_rate / unit
    cost_output = usage.output_tokens * output_rate / unit
    total_cost = cost_input + cost_cached + cost_output
    cost_jpy = total_cost * usd_to_jpy
    return GPTCostEstimate(
        model=usage.model,
        input_tokens=usage.input_tokens,
        cached_input_tokens=usage.cached_input_tokens,
        output_tokens=usage.output_tokens,
        total_tokens=usage.total_tokens,
        cost_usd=total_cost,
        cost_jpy=cost_jpy,
    )
