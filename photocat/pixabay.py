"""Utilities for downloading assets from the Pixabay API."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional
from urllib import parse, request

from typer import BadParameter

API_URL = "https://pixabay.com/api/"
DEFAULT_USER_AGENT = "PhotoCategorizerPixabayClient/1.0"

_PRESETS: dict[str, dict[str, Any]] = {
    "background": {
        "category": "backgrounds",
        "orientation": "horizontal",
        "image_type": "all",
        "min_width": 1920,
        "min_height": 1080,
        "safesearch": "true",
    },
    "icon": {
        "image_type": "vector",
        "colors": "transparent",
        "min_width": 256,
        "min_height": 256,
        "safesearch": "true",
    },
    "item": {
        "image_type": "illustration",
        "colors": "transparent",
        "min_width": 600,
        "min_height": 600,
        "safesearch": "true",
    },
}


@dataclass
class PixabayDownloadResult:
    path: Path
    hit: dict[str, Any]


class PixabayClient:
    """Thin wrapper around Pixabay's REST API."""

    def __init__(self, api_key: str, user_agent: str = DEFAULT_USER_AGENT) -> None:
        if not api_key:
            raise ValueError("Pixabay API key must not be empty")
        self.api_key = api_key
        self.user_agent = user_agent

    def search(self, params: dict[str, Any]) -> dict[str, Any]:
        full_params = {"key": self.api_key}
        full_params.update({k: v for k, v in params.items() if v not in (None, "")})
        query = parse.urlencode(full_params)
        url = f"{API_URL}?{query}"
        req = request.Request(url, headers={"User-Agent": self.user_agent})
        with request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        if "hits" not in data:
            raise RuntimeError("Unexpected Pixabay response structure")
        return data

    def download_hits(
        self,
        hits: Iterable[dict[str, Any]],
        output_dir: Path,
        limit: Optional[int] = None,
        skip_existing: bool = True,
    ) -> list[PixabayDownloadResult]:
        output_dir.mkdir(parents=True, exist_ok=True)
        results: list[PixabayDownloadResult] = []
        for idx, hit in enumerate(hits, start=1):
            if limit and idx > limit:
                break
            url = hit.get("largeImageURL") or hit.get("fullHDURL") or hit.get("imageURL")
            if not url:
                continue
            filename = self._suggest_filename(hit, url)
            path = output_dir / filename
            if path.exists() and skip_existing:
                results.append(PixabayDownloadResult(path=path, hit=hit))
                continue
            req = request.Request(url, headers={"User-Agent": self.user_agent})
            with request.urlopen(req) as resp, path.open("wb") as fh:
                shutil.copyfileobj(resp, fh)
            results.append(PixabayDownloadResult(path=path, hit=hit))
        return results

    @staticmethod
    def _suggest_filename(hit: dict[str, Any], url: str) -> str:
        suffix = Path(parse.urlparse(url).path).suffix or ".jpg"
        return f"pixabay_{hit.get('id', 'unknown')}{suffix}"


def load_api_key(env_path: Path | None = None) -> str:
    key = os.getenv("PIXABAY_API_KEY")
    if key:
        return key
    env_path = env_path or Path(".env")
    if env_path.exists():
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            name, value = line.split("=", 1)
            if name.strip().lower() in {"pixabay_api_key", "key", "api_key"}:
                stripped = value.strip()
                val = stripped.strip('"').strip("'")
                if val:
                    return val
    raise BadParameter("Pixabay API key not found; set PIXABAY_API_KEY or add key=... to .env")


def build_params(
    preset: str,
    query: str,
    page: int,
    per_page: int,
    order: str,
    overrides: dict[str, Any],
) -> dict[str, Any]:
    preset_key = preset.lower()
    if preset_key not in _PRESETS:
        raise BadParameter(f"Preset must be one of {', '.join(sorted(_PRESETS))}")
    params = dict(_PRESETS[preset_key])
    params.update(overrides)
    params["q"] = query
    params["page"] = page
    params["per_page"] = per_page
    params["order"] = order
    return params


def save_metadata(results: list[PixabayDownloadResult], output_dir: Path) -> Path:
    metadata_path = output_dir / "pixabay_metadata.json"
    payload: list[dict[str, Any]] = []
    for item in results:
        hit = dict(item.hit)
        hit["local_path"] = str(item.path)
        payload.append(hit)
    with metadata_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    return metadata_path
