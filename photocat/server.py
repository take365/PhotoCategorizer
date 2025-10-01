"""FastAPI web UI for multi-attribute image search."""

from __future__ import annotations

import base64
import json
import time
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
from fastapi import Body, FastAPI, HTTPException, Query, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from PIL import Image
from pydantic import BaseModel, Field, validator

from .attr_index import ATTR_KEYS, AttributeIndexer, IndexPaths, WeightedResult, create_text_client
from .cluster_map import CLUSTER_MODES, ClusterDatasetManager
from .scatter import AxisDefinition, ScatterProjector

ATTR_LABELS = {
    "location": "ロケーション",
    "subject": "被写体",
    "tone": "トーン",
    "style": "スタイル",
    "composition": "構図",
}

DEFAULT_TOP_K = 30
THUMB_SIZE = 260
FULL_SIZE = 720
SCATTER_THUMB_SIZE = 96
CLUSTER_THUMB_SIZE = 96


class AxisModel(BaseModel):
    key: str = Field(example="location")
    positives: list[str]
    negatives: list[str]

    @validator("key")
    def _validate_key(cls, value: str) -> str:  # noqa: N805,N804
        if value not in ATTR_KEYS:
            raise ValueError(f"key must be one of: {', '.join(ATTR_KEYS)}")
        return value

    @validator("positives", "negatives", pre=True, each_item=False)
    def _clean_terms(cls, value: Iterable[str]) -> list[str]:  # noqa: N805,N804
        cleaned: list[str] = []
        for item in value:
            text = str(item).strip()
            if text:
                cleaned.append(text)
        return cleaned


class ScatterRequestModel(BaseModel):
    axis_x: AxisModel
    axis_y: AxisModel
    limit: int = Field(200, ge=10, le=1000)
    scaling: str = Field("robust")


def _default_form_values() -> dict[str, Any]:
    return {
        "general_query": "",
        "attr_queries": {key: "" for key in ATTR_KEYS},
        "attr_weights": {key: 50 for key in ATTR_KEYS},  # 0-100 scale
        "image_weight": 0,
        "top_k": DEFAULT_TOP_K,
    }


@lru_cache(maxsize=512)
def _image_to_base64(path: str, size: int) -> str:
    file_path = Path(path)
    if not file_path.exists():
        return ""
    with Image.open(file_path) as img:
        img = img.convert("RGB")
        img.thumbnail((size, size))
        buffer = BytesIO()
        img.save(buffer, format="JPEG", quality=85)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _prepare_results(indexer: AttributeIndexer, results: Iterable[WeightedResult]) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    results = list(results)
    if not results:
        return prepared
    max_score = results[0].score or 1.0
    for item in results:
        path = indexer.get_image_path(item.image_id)
        if path is None or not path.exists():
            continue
        attrs = indexer.get_attributes(item.image_id)
        thumb = _image_to_base64(str(path), THUMB_SIZE)
        full = _image_to_base64(str(path), FULL_SIZE)
        attr_json = json.dumps(attrs, ensure_ascii=False, indent=2)
        contributions = []
        total = item.score
        for key, value in sorted(item.contributions.items(), key=lambda kv: kv[1], reverse=True):
            label = "画像" if key == "image" else ATTR_LABELS.get(key, key)
            percent = (value / total * 100.0) if total else 0.0
            contributions.append({
                "key": key,
                "label": label,
                "value": value,
                "percent": percent,
            })
        prepared.append(
            {
                "image_id": item.image_id,
                "score": total,
                "score_percent": (total / max_score * 100.0) if max_score else 0.0,
                "image_path": str(path),
                "thumb": thumb,
                "full": full,
                "attributes": attrs,
                "attr_json": attr_json,
                "contributions": contributions,
            }
        )
    return prepared


def _select_result(items: list[dict[str, Any]], selected_id: Optional[int]) -> Optional[dict[str, Any]]:
    if not items:
        return None
    if selected_id is None:
        target = items[0]
    else:
        target = next((item for item in items if item["image_id"] == selected_id), items[0])
    full = _image_to_base64(target["image_path"], FULL_SIZE)
    target = dict(target)
    target["full"] = full
    target["attr_json"] = json.dumps(target["attributes"], ensure_ascii=False, indent=2)
    return target


def _run_weighted_search(
    indexer: AttributeIndexer,
    form_values: dict[str, Any],
    *,
    image_vector: Optional[np.ndarray] = None,
) -> tuple[list[dict[str, Any]], Optional[dict[str, Any]], float, Optional[str]]:
    general_query: str = form_values.get("general_query", "").strip()
    attr_queries: Dict[str, str] = form_values.get("attr_queries", {})
    attr_weights_raw: Dict[str, int] = form_values.get("attr_weights", {})
    image_weight_raw: int = int(form_values.get("image_weight", 0))
    top_k: int = int(form_values.get("top_k", DEFAULT_TOP_K))

    attr_weights: dict[str, float] = {}
    queries: dict[str, str] = {}
    for key in ATTR_KEYS:
        raw_weight = int(attr_weights_raw.get(key, 0)) if attr_weights_raw.get(key) is not None else 0
        text = (attr_queries.get(key) or "").strip()
        weight = max(min(raw_weight, 100), 0) / 100.0
        if not text:
            attr_weights[key] = 0.0
            queries[key] = ""
            continue
        attr_weights[key] = weight if weight > 0 else 0.0
        queries[key] = text

    image_weight = max(min(image_weight_raw, 100), 0) / 100.0
    if all(weight <= 0 for weight in attr_weights.values()) and image_weight <= 0:
        if general_query:
            primary = "subject"
            attr_weights[primary] = 1.0
            queries[primary] = general_query
        else:
            return [], None, 0.0, "検索条件または重みを入力してください。"

    total_weight = sum(attr_weights.values()) + image_weight
    if total_weight <= 0:
        return [], None, 0.0, "有効な重みが設定されていません。"

    start = time.perf_counter()
    try:
        results = indexer.weighted_search(
            queries,
            attr_weights,
            image_vector=image_vector,
            image_weight=image_weight,
            top_k=top_k,
        )
    except Exception as err:  # noqa: BLE001
        return [], None, 0.0, f"埋め込みの取得に失敗しました: {err}"
    elapsed = time.perf_counter() - start
    prepared = _prepare_results(indexer, results)
    selected = _select_result(prepared, None)
    return prepared, selected, elapsed, None


def create_app(
    index_dir: Path,
    *,
    embed_base_url: str,
    embed_api_key: str,
    embed_model: str,
) -> FastAPI:
    index_paths = IndexPaths(index_dir)
    client = create_text_client(embed_base_url, embed_api_key)
    indexer = AttributeIndexer(index_paths, text_client=client, text_model=embed_model)
    templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent.parent / "templates"))
    projector = ScatterProjector(indexer)
    cluster_manager = ClusterDatasetManager(indexer, index_paths.cluster_dir)

    app = FastAPI(title="Photo Attribute Search")

    def render(
        request: Request,
        form_values: dict[str, Any],
        results: list[dict[str, Any]],
        selected: Optional[dict[str, Any]],
        elapsed: float,
        message: Optional[str] = None,
    ) -> HTMLResponse:
        context = {
            "request": request,
            "attr_labels": ATTR_LABELS,
            "form_values": form_values,
            "results": results,
            "selected": selected,
            "elapsed": elapsed,
            "message": message,
        }
        return templates.TemplateResponse("attr_search.html", context)

    @app.get("/", response_class=HTMLResponse)
    async def root(request: Request) -> HTMLResponse:
        values = _default_form_values()
        return render(request, values, [], None, 0.0)

    @app.post("/search", response_class=HTMLResponse)
    async def search(request: Request) -> HTMLResponse:
        form = await request.form()
        mode = form.get("mode", "search")
        form_values = _default_form_values()
        message: Optional[str] = None
        selected: Optional[dict[str, Any]] = None
        results: list[dict[str, Any]] = []
        elapsed = 0.0

        try:
            if mode == "search":
                form_values["general_query"] = str(form.get("general_query", "")).strip()
                for key in ATTR_KEYS:
                    form_values["attr_queries"][key] = str(form.get(f"{key}_query", "")).strip()
                    form_values["attr_weights"][key] = int(form.get(f"{key}_weight", form_values["attr_weights"][key]))
                form_values["image_weight"] = int(form.get("image_weight", form_values["image_weight"]))
                form_values["top_k"] = max(int(form.get("top_k", DEFAULT_TOP_K)), 1)

                upload = form.get("image_file")
                image_vector = None
                if isinstance(upload, UploadFile) and upload.filename:
                    data = await upload.read()
                    if data:
                        try:
                            image_vector = indexer.encode_image_bytes(data)
                        except Exception:
                            message = "画像ベクトルの生成に失敗しました。"
                results, selected, elapsed, warn = _run_weighted_search(indexer, form_values, image_vector=image_vector)
                if warn and not message:
                    message = warn
            elif mode == "from_image":
                ref_id = int(form.get("ref_image_id", 0))
                path = indexer.get_image_path(ref_id)
                if not path:
                    message = "参照画像が見つかりません。"
                else:
                    form_values = _default_form_values()
                    form_values["image_weight"] = 100
                    image_vector = indexer.encode_image_path(path)
                    results, selected, elapsed, warn = _run_weighted_search(indexer, form_values, image_vector=image_vector)
                    if warn and not message:
                        message = warn
            elif mode == "from_attr":
                ref_id = int(form.get("ref_image_id", 0))
                attr_key = str(form.get("ref_attr_key", ""))
                attrs = indexer.get_attributes(ref_id)
                text = attrs.get(attr_key, "")
                if not text:
                    message = "参照属性が見つかりません。"
                else:
                    form_values = _default_form_values()
                    for key in ATTR_KEYS:
                        form_values["attr_weights"][key] = 0
                        form_values["attr_queries"][key] = ""
                    form_values["attr_queries"][attr_key] = text
                    form_values["attr_weights"][attr_key] = 100
                    form_values["general_query"] = text
                    results, selected, elapsed, warn = _run_weighted_search(indexer, form_values)
                    if warn and not message:
                        message = warn
            else:
                message = "不明なリクエストです。"
        except Exception as err:  # noqa: BLE001
            message = f"検索中にエラーが発生しました: {err}"

        if results and selected is None:
            selected = _select_result(results, None)
        return render(request, form_values, results, selected, elapsed, message)

    @app.get("/scatter", response_class=HTMLResponse)
    async def scatter_view(request: Request) -> HTMLResponse:
        context = {
            "request": request,
            "default_axis_x": {
                "key": "location",
                "positives": "outdoor\nopen air\nlandscape",
                "negatives": "indoor\nroom interior",
                "positives_ja": "屋外\n野外\n自然の風景",
                "negatives_ja": "屋内\n室内",
            },
            "default_axis_y": {
                "key": "subject",
                "positives": "people\nfamily\nportrait",
                "negatives": "objects\nstill life",
                "positives_ja": "人\n家族\n人物",
                "negatives_ja": "物体\n静物",
            },
        }
        return templates.TemplateResponse("scatter.html", context)

    @app.post("/scatter/project", response_class=JSONResponse)
    async def scatter_project(payload: ScatterRequestModel = Body(...)) -> JSONResponse:
        try:
            axis_x = AxisDefinition(
                key=payload.axis_x.key,
                positives=payload.axis_x.positives,
                negatives=payload.axis_x.negatives,
            )
            axis_y = AxisDefinition(
                key=payload.axis_y.key,
                positives=payload.axis_y.positives,
                negatives=payload.axis_y.negatives,
            )
            result = projector.project(axis_x, axis_y, limit=payload.limit, scaling=payload.scaling)
        except ValueError as err:
            raise HTTPException(status_code=400, detail=str(err)) from err
        except Exception as err:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=str(err)) from err

        response_points: list[dict[str, Any]] = []
        for point in result.points:
            attrs = indexer.get_attributes(point.image_id)
            thumb_path = indexer.paths.thumbnail_path(point.image_id)
            if thumb_path.exists():
                thumb_b64 = _image_to_base64(str(thumb_path), SCATTER_THUMB_SIZE)
            else:
                original = indexer.get_image_path(point.image_id)
                thumb_b64 = _image_to_base64(str(original), SCATTER_THUMB_SIZE) if original else ""
            response_points.append(
                {
                    "image_id": point.image_id,
                    "x": point.x,
                    "y": point.y,
                    "raw_x": point.raw_x,
                    "raw_y": point.raw_y,
                    "magnitude": point.magnitude,
                    "thumbnail": thumb_b64,
                    "attributes": attrs,
                    "image_path": str(indexer.get_image_path(point.image_id) or ""),
                }
            )

        payload = {
            "points": response_points,
            "stats": result.stats,
        }
        return JSONResponse(payload)

    @app.get("/scatter/image/{image_id}", response_class=JSONResponse)
    async def scatter_image(image_id: int) -> JSONResponse:
        path = indexer.get_image_path(image_id)
        if not path or not path.exists():
            raise HTTPException(status_code=404, detail="画像が見つかりません")
        image_b64 = _image_to_base64(str(path), FULL_SIZE)
        attrs = indexer.get_attributes(image_id)
        return JSONResponse({
            "image_id": image_id,
            "image": image_b64,
            "attributes": attrs,
            "path": str(path),
        })

    @app.get("/clusters", response_class=HTMLResponse)
    async def clusters_view(request: Request) -> HTMLResponse:
        context = {
            "request": request,
            "modes": CLUSTER_MODES,
            "default_mode": "location",
        }
        return templates.TemplateResponse("cluster.html", context)

    @app.get("/clusters/{mode}/meta", response_class=JSONResponse)
    async def clusters_meta(mode: str) -> JSONResponse:
        try:
            payload = cluster_manager.dataset_meta(mode)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="クラスタデータが見つかりません。") from None
        except ValueError as err:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=str(err)) from err
        return JSONResponse(payload)

    @app.get("/clusters/{mode}/chunks/{chunk}", response_class=JSONResponse)
    async def clusters_chunk(
        mode: str,
        chunk: int,
        size: int = Query(200, ge=10, le=2000),
    ) -> JSONResponse:
        try:
            payload = cluster_manager.chunk_points(mode, chunk, size)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="クラスタデータが見つかりません。") from None
        except ValueError as err:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=str(err)) from err

        enriched: list[dict[str, Any]] = []
        for point in payload.get("points", []):
            image_id = int(point.get("image_id", 0))
            image_path = indexer.get_image_path(image_id)
            thumb_path = indexer.paths.thumbnail_path(image_id)
            if thumb_path.exists():
                thumb_b64 = _image_to_base64(str(thumb_path), CLUSTER_THUMB_SIZE)
            elif image_path:
                thumb_b64 = _image_to_base64(str(image_path), CLUSTER_THUMB_SIZE)
            else:
                thumb_b64 = ""
            attrs = indexer.get_attributes(image_id)
            enriched.append(
                {
                    **point,
                    "thumbnail": thumb_b64,
                    "attributes": attrs,
                    "image_path": str(image_path) if image_path else "",
                    "color": cluster_manager.cluster_color(int(point.get("cluster_id", -1))),
                }
            )
        payload["points"] = enriched
        return JSONResponse(payload)

    @app.get("/clusters/{mode}/detail/{cluster_id}", response_class=JSONResponse)
    async def clusters_detail(mode: str, cluster_id: int) -> JSONResponse:
        try:
            detail = cluster_manager.cluster_detail(mode, cluster_id)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="クラスタデータが見つかりません。") from None
        except ValueError as err:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=str(err)) from err

        representatives: list[dict[str, Any]] = []
        for image_id in detail.pop("representative_image_ids", []):
            image_path = indexer.get_image_path(image_id)
            thumb_path = indexer.paths.thumbnail_path(image_id)
            if thumb_path.exists():
                thumb_b64 = _image_to_base64(str(thumb_path), THUMB_SIZE)
            elif image_path:
                thumb_b64 = _image_to_base64(str(image_path), THUMB_SIZE)
            else:
                thumb_b64 = ""
            attrs = indexer.get_attributes(image_id)
            representatives.append(
                {
                    "image_id": image_id,
                    "thumbnail": thumb_b64,
                    "image_path": str(image_path) if image_path else "",
                    "attributes": attrs,
                }
            )
        detail["representative_images"] = representatives
        return JSONResponse(detail)

    return app
