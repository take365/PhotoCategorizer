"""Attribute extraction PoC pipeline using local vision-capable LLMs."""

from __future__ import annotations

import base64
import json
import re
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable, Sequence

from openai import APIError, OpenAI
from PIL import Image

from .utils import ensure_parent_dir, list_image_paths

__all__ = ["ModelSpec", "run_attribute_extraction", "generate_html_report"]


PROMPT_TEXT = (
    "あなたは画像解析アシスタントです。入力画像を見て、必ず JSON 形式で属性を出力してください。"
    "出力は日本語で、短い単語ではなく自然な文章を使ってください。"
    "余計な説明やコメントは書かず、以下のキーのみを返してください。"
    "\n\n出力フォーマット:\n"
    "{\n"
    '  "location": "海辺の砂浜が広がり、遠くには水平線と沈みかけた夕日が見える情景",\n'
    '  "subject": "白い毛並みの犬が元気よく砂浜を走りながら波打ち際で遊んでいる様子",\n'
    '  "tone": "暖かい雰囲気",\n'
    '  "style": "シネマ風",\n'
    '  "composition": "引きの構図"\n'
    "}\n\n"
    "location と subject は日本語の文章で 30〜50 文字程度。"
    "tone, style, composition は12文字以内程度の短い表現。"
    "出力は JSON のみで、それ以外の文字は含めないでください。"
)

USER_PROMPT = "次の画像から属性を抽出してください。"


@dataclass(slots=True)
class ModelSpec:
    """Configuration describing how to call a local model."""

    key: str
    label: str
    client: OpenAI
    model: str
    temperature: float = 0.2
    max_output_tokens: int = 400
    request_timeout: float | None = 60.0


def _encode_image(path: Path, max_size: int = 1024) -> tuple[str, str]:
    """Return base64 JPEG representation of *path* resized for prompting."""

    with Image.open(path) as image:
        image = image.convert("RGB")
        image.thumbnail((max_size, max_size))
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=90)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return encoded, "image/jpeg"


_JSON_PATTERN = re.compile(r"\{.*\}", re.DOTALL)


def _parse_json_payload(text: str) -> tuple[dict[str, Any] | None, str | None]:
    """Try to parse JSON object from *text*; return (payload, error)."""

    cleaned = text.strip()
    if not cleaned:
        return None, "empty response"
    try:
        value = json.loads(cleaned)
        if isinstance(value, dict):
            return value, None
    except json.JSONDecodeError:
        pass

    match = _JSON_PATTERN.search(cleaned)
    if match:
        try:
            value = json.loads(match.group(0))
            if isinstance(value, dict):
                return value, "trimmed"
        except json.JSONDecodeError:
            return None, "invalid json"
    return None, "invalid json"


def _call_model(path: Path, spec: ModelSpec) -> dict[str, Any]:
    """Invoke *spec* on *path* and return structured result."""

    image_b64, mime = _encode_image(path)

    system_message = {
        "role": "system",
        "content": [{"type": "text", "text": PROMPT_TEXT}],
    }
    user_message = {
        "role": "user",
        "content": [
            {"type": "text", "text": USER_PROMPT},
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{image_b64}", "detail": "auto"},
            },
        ],
    }

    try:
        response = spec.client.chat.completions.create(
            model=spec.model,
            messages=[system_message, user_message],
            temperature=spec.temperature,
            max_tokens=spec.max_output_tokens,
            timeout=spec.request_timeout,
        )
        text = response.choices[0].message.content if response.choices else ""
    except APIError as err:
        return {"ok": False, "error": str(err)}
    except Exception as err:  # noqa: BLE001
        return {"ok": False, "error": str(err)}

    text = text or ""
    parsed, parse_error = _parse_json_payload(text)

    result: dict[str, Any] = {
        "ok": parsed is not None,
        "raw": text,
    }
    if parsed is not None:
        result["data"] = parsed
        if parse_error:
            result["parse_note"] = parse_error
    else:
        result["error"] = parse_error or "unknown"
    return result


def run_attribute_extraction(
    input_dir: Path,
    model_specs: Sequence[ModelSpec],
    *,
    limit: int = 100,
    sleep: float = 0.0,
) -> list[dict[str, Any]]:
    """Run attribute extraction on images under *input_dir* with *model_specs*."""

    image_paths = list_image_paths(input_dir)
    targets = image_paths[:limit]

    records: list[dict[str, Any]] = []
    for index, path in enumerate(targets, start=1):
        entry: dict[str, Any] = {
            "image_path": str(path),
        }
        for spec in model_specs:
            model_output = _call_model(path, spec)
            model_output["model_label"] = spec.label
            entry[spec.key] = model_output
        entry["image_base64"], entry["image_mime"] = _encode_image(path)
        records.append(entry)
        if sleep > 0.0:
            time.sleep(sleep)
    return records


def generate_html_report(records: Iterable[dict[str, Any]], output_path: Path) -> None:
    """Generate an HTML report comparing model outputs."""

    ensure_parent_dir(output_path)

    html_parts = [
        "<!DOCTYPE html>",
        "<html lang=\"ja\">",
        "<head>",
        "  <meta charset=\"utf-8\">",
        "  <title>属性抽出 PoC レポート</title>",
        "  <style>",
        "body { font-family: 'Segoe UI',sans-serif; margin: 24px; background: #f7f7f7; }",
        ".item { background: #fff; padding: 16px; margin-bottom: 20px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }",
        ".item h2 { margin-top: 0; font-size: 18px; }",
        ".content { display: flex; gap: 16px; flex-wrap: wrap; }",
        ".image { flex: 0 0 320px; }",
        ".image img { max-width: 320px; border-radius: 6px; }",
        ".responses { flex: 1 1 360px; display: flex; gap: 16px; flex-wrap: wrap; }",
        ".response { flex: 1 1 320px; background: #fafafa; border: 1px solid #e0e0e0; border-radius: 6px; padding: 12px; }",
        ".response h3 { margin-top: 0; font-size: 16px; }",
        "pre { white-space: pre-wrap; word-break: break-word; font-size: 13px; }",
        "  </style>",
        "</head>",
        "<body>",
        "  <h1>属性抽出 PoC レポート</h1>",
    ]

    records_list = list(records)
    for idx, record in enumerate(records_list, start=1):
        image_b64 = record.get("image_base64")
        image_mime = record.get("image_mime", "image/jpeg")
        image_tag = (
            f'<img src="data:{image_mime};base64,{image_b64}" alt="image {idx}">' if image_b64 else "<div>画像なし</div>"
        )
        html_parts.append("  <div class=\"item\">")
        html_parts.append(f"    <h2>{idx:03d}: {Path(record['image_path']).name}</h2>")
        html_parts.append("    <div class=\"content\">")
        html_parts.append("      <div class=\"image\">")
        html_parts.append(f"        {image_tag}")
        html_parts.append("      </div>")
        html_parts.append("      <div class=\"responses\">")

        for key, value in record.items():
            if key in {"image_path", "image_base64", "image_mime"}:
                continue
            if not isinstance(value, dict):
                continue
            title = value.get("model_label", key)
            raw_json = json.dumps(value.get("data", value), ensure_ascii=False, indent=2)
            html_parts.append("        <div class=\"response\">")
            html_parts.append(f"          <h3>{_escape_html(str(title))}</h3>")
            html_parts.append("          <pre>")
            html_parts.append(_escape_html(raw_json))
            html_parts.append("          </pre>")
            html_parts.append("        </div>")

        html_parts.append("      </div>")
        html_parts.append("    </div>")
        html_parts.append("  </div>")

    html_parts.extend(["</body>", "</html>"])

    output_path.write_text("\n".join(html_parts), encoding="utf-8")


def _escape_html(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\"", "&quot;")
        .replace("'", "&#39;")
    )
