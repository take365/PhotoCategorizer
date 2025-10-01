"""Embedding ingestion and FAISS indexing for attribute search."""

from __future__ import annotations

import json
import math
import os
import platform
import shutil
import sqlite3
import time
from collections import defaultdict
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Iterator, Mapping, MutableMapping, Sequence

import faiss  # type: ignore[import]
import numpy as np
import open_clip
import torch
from openai import OpenAI
from PIL import Image

from .utils import ensure_parent_dir

ATTR_KEYS = ("location", "subject", "tone", "style", "composition")
DEFAULT_ATTR_TOPK = 200

TAG_LABELS_JA = {
    "bright": "明るい",
    "dark": "暗い",
    "backlit": "逆光",
    "golden-hour": "夕暮れ／朝焼け",
    "night": "夜",
    "warm": "暖色",
    "cool": "寒色",
    "neutral": "中立",
    "monochrome": "白黒",
    "sepia": "セピア",
    "cinematic": "映画風",
    "vintage": "レトロ",
    "minimal": "ミニマル",
    "dramatic": "ドラマチック",
    "close-up": "寄り",
    "wide": "引き",
    "top-view": "真上",
    "low-angle": "ローアングル",
    "centered": "中央",
    "thirds": "三分割",
}


@dataclass(slots=True)
class WeightedResult:
    """Container for weighted similarity scoring."""

    image_id: int
    score: float
    contributions: dict[str, float]


@dataclass(slots=True)
class IndexPaths:
    """Filesystem paths for the FAISS+SQLite index bundle."""

    root: Path

    @property
    def sqlite_path(self) -> Path:
        return self.root / "meta.db"

    def attribute_index_path(self, key: str) -> Path:
        return self.root / f"attr_{key}.faiss"

    @property
    def image_index_path(self) -> Path:
        return self.root / "images.faiss"

    @property
    def thumbnail_dir(self) -> Path:
        return self.root / "thumbnails"

    def thumbnail_path(self, image_id: int) -> Path:
        return self.thumbnail_dir / f"{image_id}.jpg"

    def attr_embedding_path(self, key: str) -> Path:
        return self.root / f"attr_{key}_embeddings.npz"

    @property
    def cluster_dir(self) -> Path:
        return self.root / "clusters"

    def cluster_mode_dir(self, mode: str) -> Path:
        return self.cluster_dir / mode

    @property
    def image_embedding_path(self) -> Path:
        return self.root / "image_embeddings.npz"


class AttributeIndexer:
    """Builds FAISS indices from attribute extraction results."""

    def __init__(
        self,
        paths: IndexPaths,
        *,
        text_client: OpenAI,
        text_model: str,
        image_model: str = "ViT-B-32",
        image_pretrained: str = "openai",
        device: str = "cpu",
        thumbnail_size: int = 64,
        generate_thumbnails: bool = True,
        save_embeddings: bool = True,
    ) -> None:
        self.paths = paths
        self.text_client = text_client
        self.text_model = text_model
        self.device = torch.device(device)
        self._image_model_id = image_model
        self._image_pretrained = image_pretrained
        self._image_model = None
        self._image_preprocess = None
        self._sqlite_conn: sqlite3.Connection | None = None
        self._attr_indexes: dict[str, faiss.IndexIDMap] = {}
        self._image_index: faiss.IndexIDMap | None = None
        self._text_dim: int | None = None
        self._image_dim: int | None = None
        self._thumbnail_size = int(thumbnail_size)
        self._generate_thumbnails = bool(generate_thumbnails)
        self._save_embeddings = bool(save_embeddings)
        self._attr_embedding_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    # ------------------------------------------------------------------
    # Setup helpers

    def reset(self) -> None:
        """Remove existing index files and database."""

        if self.paths.root.exists():
            for path in self.paths.root.glob("*.faiss"):
                path.unlink(missing_ok=True)
            if self.paths.sqlite_path.exists():
                self.paths.sqlite_path.unlink()
            if self.paths.thumbnail_dir.exists():
                shutil.rmtree(self.paths.thumbnail_dir, ignore_errors=True)
            for key in ATTR_KEYS:
                embed_path = self.paths.attr_embedding_path(key)
                embed_path.unlink(missing_ok=True)
        self._attr_embedding_cache.clear()
        ensure_parent_dir(self.paths.sqlite_path)
        with self.sqlite_conn:
            self._create_schema()

    @property
    def sqlite_conn(self) -> sqlite3.Connection:
        if self._sqlite_conn is None:
            ensure_parent_dir(self.paths.sqlite_path)
            self._sqlite_conn = sqlite3.connect(self.paths.sqlite_path)
            self._sqlite_conn.execute("PRAGMA journal_mode=WAL")
            self._create_schema()
        return self._sqlite_conn

    def _create_schema(self) -> None:
        conn = self.sqlite_conn
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT UNIQUE,
                added_at REAL
            );

            CREATE TABLE IF NOT EXISTS attributes (
                image_id INTEGER,
                key TEXT,
                text TEXT,
                PRIMARY KEY (image_id, key),
                FOREIGN KEY (image_id) REFERENCES images(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS metadata (
                name TEXT PRIMARY KEY,
                value TEXT
            );
            """
        )

    # ------------------------------------------------------------------
    # Index loading helpers

    def _ensure_attr_index(self, key: str, dim: int) -> faiss.IndexIDMap:
        if key in self._attr_indexes:
            return self._attr_indexes[key]

        path = self.paths.attribute_index_path(key)
        if path.exists():
            index = faiss.read_index(str(path))
        else:
            index = faiss.IndexFlatIP(dim)
        if not isinstance(index, faiss.IndexIDMap):
            index = faiss.IndexIDMap(index)
        self._attr_indexes[key] = index
        return index

    def _ensure_image_index(self, dim: int) -> faiss.IndexIDMap:
        if self._image_index is not None:
            return self._image_index
        path = self.paths.image_index_path
        if path.exists():
            index = faiss.read_index(str(path))
        else:
            index = faiss.IndexFlatIP(dim)
        if not isinstance(index, faiss.IndexIDMap):
            index = faiss.IndexIDMap(index)
        self._image_index = index
        return index

    # ------------------------------------------------------------------
    # Embedding helpers

    def _text_embedding_dim(self) -> int:
        if self._text_dim is None:
            for key in ATTR_KEYS:
                path = self.paths.attribute_index_path(key)
                if not path.exists():
                    continue
                index = faiss.read_index(str(path))
                dim = index.d
                if not isinstance(index, faiss.IndexIDMap):
                    index = faiss.IndexIDMap(index)
                self._attr_indexes.setdefault(key, index)
                self._text_dim = dim
                break
        if self._text_dim is None:
            try:
                sample = self.text_client.embeddings.create(
                    model=self.text_model,
                    input=["テスト文"],
                    encoding_format="float",
                )
            except Exception as err:  # noqa: BLE001
                raise RuntimeError("text embedding backend is unavailable") from err
            vector = sample.data[0].embedding
            self._text_dim = len(vector)
        return self._text_dim

    def _image_embedding_dim(self) -> int:
        if self._image_dim is None:
            path = self.paths.image_index_path
            if path.exists():
                index = faiss.read_index(str(path))
                dim = index.d
                if not isinstance(index, faiss.IndexIDMap):
                    index = faiss.IndexIDMap(index)
                self._image_index = index
                self._image_dim = dim
        if self._image_dim is None:
            model, _ = self._load_image_model()
            if hasattr(model, "visual") and hasattr(model.visual, "output_dim"):
                self._image_dim = int(model.visual.output_dim)
            else:
                # Fallback: run a dummy tensor with standard 224 size
                dummy = torch.zeros(1, 3, 224, 224)
                with torch.no_grad():
                    emb = model.encode_image(dummy.to(self.device))
                self._image_dim = emb.shape[-1]
        return self._image_dim

    def _load_image_model(self):  # type: ignore[no-untyped-def]
        if self._image_model is None or self._image_preprocess is None:
            model, _, preprocess = open_clip.create_model_and_transforms(
                self._image_model_id,
                pretrained=self._image_pretrained,
                device=self.device,
            )
            model.eval()
            self._image_model = model
            self._image_preprocess = preprocess
        return self._image_model, self._image_preprocess

    # ------------------------------------------------------------------
    # Public embedding helpers

    def encode_text(self, text: str) -> np.ndarray:
        response = self.text_client.embeddings.create(
            model=self.text_model,
            input=[text],
            encoding_format="float",
        )
        return np.array(response.data[0].embedding, dtype="float32")

    def encode_image_path(self, path: Path) -> np.ndarray:
        model, preprocess = self._load_image_model()
        with Image.open(path) as img:
            image = preprocess(img.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = model.encode_image(image)
        vector = embedding.cpu().numpy()[0].astype("float32")
        norm = np.linalg.norm(vector)
        if not math.isfinite(norm) or norm == 0.0:
            raise ValueError(f"Invalid norm for image embedding: {path}")
        return vector / norm

    def encode_image_bytes(self, data: bytes) -> np.ndarray:
        model, preprocess = self._load_image_model()
        with Image.open(BytesIO(data)) as img:
            image = preprocess(img.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = model.encode_image(image)
        vector = embedding.cpu().numpy()[0].astype("float32")
        norm = np.linalg.norm(vector)
        if not math.isfinite(norm) or norm == 0.0:
            raise ValueError("Invalid norm for image embedding (bytes input)")
        return vector / norm

    # ------------------------------------------------------------------
    # Public API

    def ingest_records(
        self,
        records: Sequence[Mapping[str, object]],
        *,
        attr_keys: Sequence[str] | None = None,
    ) -> None:
        text_dim = self._text_embedding_dim()
        image_dim = self._image_embedding_dim()

        target_keys = tuple(attr_keys) if attr_keys else ATTR_KEYS
        normalised_keys = tuple(key.strip() for key in target_keys if key.strip())
        if not normalised_keys:
            raise ValueError("attr_keys must contain at least one key")

        allowed_keys = set(ATTR_KEYS)
        for key in normalised_keys:
            if key not in allowed_keys:
                raise ValueError(f"Unsupported attr key: {key}")

        attr_vectors: dict[str, list[tuple[int, np.ndarray]]] = {key: [] for key in normalised_keys}
        image_vectors: list[tuple[int, np.ndarray]] = []

        for record in records:
            image_path = Path(str(record["image_path"]))
            image_id = self._get_or_create_image(image_path)
            attr_payload = self._select_attribute_payload(record)
            if not attr_payload:
                continue

            for key in normalised_keys:
                text_value = attr_payload.get(key)
                if not text_value or not isinstance(text_value, str) or not text_value.strip():
                    continue
                embedding = self.encode_text(text_value)
                if embedding.shape[-1] != text_dim:
                    raise ValueError(f"Text embedding dimension mismatch for {key}")
                attr_vectors[key].append((image_id, embedding))
                self._upsert_attribute_text(image_id, key, text_value)

            image_vector = self.encode_image_path(image_path)
            if image_vector.shape[-1] != image_dim:
                raise ValueError("Image embedding dimension mismatch")
            image_vectors.append((image_id, image_vector))

            if self._generate_thumbnails:
                self._save_thumbnail(image_id, image_path)

        # Persist to FAISS
        for key, pairs in attr_vectors.items():
            if not pairs:
                continue
            ids = np.array([pair[0] for pair in pairs], dtype="int64")
            vecs = np.stack([pair[1] for pair in pairs]).astype("float32")
            vecs_norm = vecs.copy()
            faiss.normalize_L2(vecs_norm)
            index = self._ensure_attr_index(key, text_dim)
            index.remove_ids(ids)
            index.add_with_ids(vecs_norm, ids)
            if self._save_embeddings:
                self._write_attr_embeddings(key, ids, vecs_norm)

        if image_vectors:
            ids = np.array([pair[0] for pair in image_vectors], dtype="int64")
            vecs = np.stack([pair[1] for pair in image_vectors]).astype("float32")
            faiss.normalize_L2(vecs)
            index = self._ensure_image_index(image_dim)
            index.remove_ids(ids)
            index.add_with_ids(vecs, ids)

        self.flush()

    def flush(self) -> None:
        if self._sqlite_conn is not None:
            self._sqlite_conn.commit()
        for key, index in self._attr_indexes.items():
            faiss.write_index(index, str(self.paths.attribute_index_path(key)))
        if self._image_index is not None:
            faiss.write_index(self._image_index, str(self.paths.image_index_path))

    # ------------------------------------------------------------------
    # Thumbnail helpers

    def _save_thumbnail(self, image_id: int, image_path: Path) -> None:
        try:
            thumb_path = self.paths.thumbnail_path(image_id)
            ensure_parent_dir(thumb_path)
            if thumb_path.exists():
                return
            with Image.open(image_path) as img:
                img = img.convert("RGB")
                img.thumbnail((self._thumbnail_size, self._thumbnail_size))
                img.save(thumb_path, format="JPEG", quality=80)
        except Exception:  # noqa: BLE001
            # サムネ生成失敗時は検索機能に致命的でないためスキップ
            thumb_path.unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # Embedding persistence helpers

    def _write_attr_embeddings(self, key: str, ids: np.ndarray, vectors: np.ndarray) -> None:
        path = self.paths.attr_embedding_path(key)
        ensure_parent_dir(path)
        new_ids = ids.astype("int64")
        new_vecs = vectors.astype("float16")

        combined: dict[int, np.ndarray] = {}
        if path.exists():
            try:
                with np.load(path, allow_pickle=False) as data:
                    prev_ids = data["ids"].astype("int64")
                    prev_vecs = data["embeddings"].astype("float16")
                for prev_id, prev_vec in zip(prev_ids, prev_vecs, strict=False):
                    combined[int(prev_id)] = prev_vec
            except Exception:  # noqa: BLE001
                combined = {}

        for image_id, vec in zip(new_ids, new_vecs, strict=False):
            combined[int(image_id)] = vec

        ordered_ids = np.array(sorted(combined.keys()), dtype="int64")
        ordered_vecs = np.stack([combined[idx] for idx in ordered_ids]).astype("float16")
        np.savez_compressed(path, ids=ordered_ids, embeddings=ordered_vecs)
        self._attr_embedding_cache[key] = (ordered_ids, ordered_vecs.astype("float32"))

    def load_attr_embeddings(self, key: str) -> tuple[np.ndarray, np.ndarray]:
        if key not in ATTR_KEYS:
            raise ValueError(f"Unsupported attr key: {key}")
        cached = self._attr_embedding_cache.get(key)
        if cached is not None:
            ids, vectors = cached
            return ids.copy(), vectors.copy()
        path = self.paths.attr_embedding_path(key)
        if not path.exists():
            raise FileNotFoundError(f"Attribute embeddings not found for key '{key}'")
        with np.load(path, allow_pickle=False) as data:
            ids = data["ids"].astype("int64")
            vectors = data["embeddings"].astype("float32")
        self._attr_embedding_cache[key] = (ids, vectors)
        return ids.copy(), vectors.copy()

    def load_image_embeddings(self) -> tuple[np.ndarray, np.ndarray]:
        dim = self._image_embedding_dim()
        index = self._ensure_image_index(dim)
        ntotal = int(index.ntotal)
        if ntotal == 0:
            return np.empty(0, dtype="int64"), np.empty((0, dim), dtype="float32")

        cache_path = self.paths.image_embedding_path
        if cache_path.exists():
            with np.load(cache_path, allow_pickle=False) as data:
                ids = data["ids"].astype("int64")
                vectors = data["embeddings"].astype("float32")
            if ids.shape[0] == ntotal:
                return ids, vectors

        id_map = faiss.vector_to_array(index.id_map).astype("int64")
        vectors = np.empty((ntotal, dim), dtype="float32")

        reconstruct_supported = hasattr(index, "reconstruct")
        if reconstruct_supported:
            try:
                sample = index.reconstruct(0)
                if isinstance(sample, np.ndarray) and sample.shape[0] == dim:
                    vectors[0] = sample.astype("float32")
                    for row in range(1, ntotal):
                        vectors[row] = index.reconstruct(row).astype("float32")
                    np.savez_compressed(cache_path, ids=id_map, embeddings=vectors)
                    return id_map, vectors
            except Exception:  # noqa: BLE001 - fallback to re-encoding
                pass

        # reconstruct() not available → re-encode from original files
        for pos, image_id in enumerate(id_map):
            path = self.get_image_path(int(image_id))
            if not path or not path.exists():
                raise FileNotFoundError(f"画像ファイルが見つかりません: {image_id}")
            vector = self.encode_image_path(path)
            vectors[pos] = vector
        np.savez_compressed(cache_path, ids=id_map, embeddings=vectors.astype("float32"))
        return id_map, vectors

    # ------------------------------------------------------------------
    # Retrieval helpers

    def _get_or_create_image(self, path: Path) -> int:
        conn = self.sqlite_conn
        cursor = conn.execute("SELECT id FROM images WHERE path = ?", (str(path),))
        row = cursor.fetchone()
        if row:
            return int(row[0])
        cursor = conn.execute(
            "INSERT INTO images (path, added_at) VALUES (?, ?)",
            (str(path), time.time()),
        )
        return int(cursor.lastrowid)

    def _upsert_attribute_text(self, image_id: int, key: str, text: str) -> None:
        with self.sqlite_conn:
            self.sqlite_conn.execute(
                "INSERT INTO attributes (image_id, key, text) VALUES (?, ?, ?)"
                " ON CONFLICT(image_id, key) DO UPDATE SET text = excluded.text",
                (image_id, key, text),
            )

    def _select_attribute_payload(self, record: Mapping[str, object]) -> Mapping[str, str] | None:
        # prefer qwen data by default
        for candidate in ("qwen", "gemma"):
            payload = record.get(candidate)
            if isinstance(payload, Mapping) and payload.get("ok") and isinstance(payload.get("data"), Mapping):
                return self._format_attribute_payload(payload["data"])  # type: ignore[arg-type]
        return None

    def _format_attribute_payload(self, data: Mapping[str, object]) -> Mapping[str, str]:
        location = str(data.get("location", "")).strip()
        subject = str(data.get("subject", "")).strip()
        formatted: dict[str, str] = {
            "location": location,
            "subject": subject,
        }

        tags = data.get("tags")
        if isinstance(tags, Mapping):
            for group in ("tone", "style", "composition"):
                values = tags.get(group, [])
                if isinstance(values, (list, tuple)):
                    ja_values = [
                        TAG_LABELS_JA.get(str(value), str(value))
                        for value in values
                        if str(value).strip()
                    ]
                    formatted[group] = "、".join(ja_values)
                elif isinstance(values, str):
                    formatted[group] = values
        else:
            for group in ("tone", "style", "composition"):
                raw = data.get(group)
                if raw:
                    formatted[group] = str(raw)

        return formatted

    # ------------------------------------------------------------------
    # Query helpers

    def search_attributes(
        self,
        key: str,
        query: str,
        *,
        top_k: int = 10,
    ) -> list[tuple[int, float]]:
        dim = self._text_embedding_dim()
        index = self._ensure_attr_index(key, dim)
        vector = self.encode_text(query)
        faiss.normalize_L2(vector.reshape(1, -1))
        scores, ids = index.search(vector.reshape(1, -1), top_k)
        return [(int(idx), float(score)) for idx, score in zip(ids[0], scores[0]) if idx != -1]

    def search_images(self, vector: np.ndarray, *, top_k: int = 10) -> list[tuple[int, float]]:
        dim = self._image_embedding_dim()
        index = self._ensure_image_index(dim)
        query = vector.astype("float32").reshape(1, -1)
        faiss.normalize_L2(query)
        scores, ids = index.search(query, top_k)
        return [(int(idx), float(score)) for idx, score in zip(ids[0], scores[0]) if idx != -1]

    def iter_attribute_texts(self, image_ids: Sequence[int]) -> Iterator[tuple[int, dict[str, str]]]:
        placeholders = ",".join("?" for _ in image_ids)
        query = (
            f"SELECT image_id, key, text FROM attributes "
            f"WHERE image_id IN ({placeholders})"
        )
        rows = self.sqlite_conn.execute(query, tuple(image_ids))
        mapping: dict[int, dict[str, str]] = {}
        for image_id, key, text in rows:
            mapping.setdefault(int(image_id), {})[str(key)] = str(text)
        for image_id in image_ids:
            yield image_id, mapping.get(int(image_id), {})

    def get_image_path(self, image_id: int) -> Path | None:
        row = self.sqlite_conn.execute("SELECT path FROM images WHERE id = ?", (image_id,)).fetchone()
        if not row:
            return None
        raw = str(row[0])
        resolved = self._resolve_path(raw)
        if resolved.exists():
            return resolved
        return Path(raw)

    def get_attributes(self, image_id: int) -> dict[str, str]:
        rows = self.sqlite_conn.execute(
            "SELECT key, text FROM attributes WHERE image_id = ?",
            (image_id,),
        ).fetchall()
        return {str(key): str(text) for key, text in rows}

    def weighted_search(
        self,
        attr_queries: Mapping[str, str],
        attr_weights: Mapping[str, float],
        *,
        image_vector: np.ndarray | None = None,
        image_weight: float = 0.0,
        top_k: int = 20,
        per_attr_top_k: int = DEFAULT_ATTR_TOPK,
    ) -> list[WeightedResult]:
        contributions: MutableMapping[int, dict[str, float]] = defaultdict(dict)
        candidate_ids: set[int] = set()

        text_dim = self._text_embedding_dim()
        for key in ATTR_KEYS:
            weight = float(attr_weights.get(key, 0.0) or 0.0)
            query_text = (attr_queries.get(key) or "").strip()
            if not query_text:
                continue
            if weight <= 0:
                continue
            index = self._ensure_attr_index(key, text_dim)
            try:
                vector = self.encode_text(query_text)
            except Exception:
                continue
            vector = vector.reshape(1, -1).astype("float32")
            faiss.normalize_L2(vector)
            scores, ids = index.search(vector, per_attr_top_k)
            for idx, score in zip(ids[0], scores[0]):
                if idx == -1:
                    continue
                mapped = contributions.setdefault(int(idx), {})
                mapped[key] = mapped.get(key, 0.0) + float(score) * weight
                candidate_ids.add(int(idx))

        if image_vector is not None and image_weight > 0.0:
            dim = self._image_embedding_dim()
            index = self._ensure_image_index(dim)
            query = image_vector.astype("float32").reshape(1, -1)
            faiss.normalize_L2(query)
            scores, ids = index.search(query, per_attr_top_k)
            for idx, score in zip(ids[0], scores[0]):
                if idx == -1:
                    continue
                mapped = contributions.setdefault(int(idx), {})
                mapped["image"] = mapped.get("image", 0.0) + float(score) * image_weight
                candidate_ids.add(int(idx))

        results: list[WeightedResult] = []
        for image_id in candidate_ids:
            parts = contributions.get(image_id, {})
            total = sum(parts.values())
            if total <= 0:
                continue
            results.append(WeightedResult(image_id=image_id, score=total, contributions=dict(parts)))

        results.sort(key=lambda item: item.score, reverse=True)
        return results[:top_k]

    # ------------------------------------------------------------------
    # Path helpers

    def _resolve_path(self, raw: str) -> Path:
        candidate = Path(raw)
        if candidate.exists():
            return candidate

        # Windows host reading WSL-style path (/mnt/x/...)
        if os.name == "nt" and raw.startswith("/mnt/") and len(raw) > 6:
            drive = raw[5]
            rest = raw[7:]
            windows_path = Path(f"{drive.upper()}:/{rest}")
            if windows_path.exists():
                return windows_path

        # WSL host reading Windows path (D:\...)
        if os.name != "nt" and ":" in raw and raw[1:3] == ":\\":
            if self._is_wsl():
                drive = raw[0].lower()
                rest = raw[3:].replace("\\", "/")
                wsl_path = Path(f"/mnt/{drive}/{rest}")
                if wsl_path.exists():
                    return wsl_path

        return Path(raw)

    @staticmethod
    def _is_wsl() -> bool:
        if os.name != "posix":
            return False
        return "microsoft" in platform.release().lower()


def load_attr_records(json_path: Path) -> list[dict[str, object]]:
    """Load attr_poc JSON results."""

    raw = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("attr_results.json must contain a list")
    return [dict(item) for item in raw]


def create_text_client(base_url: str | None, api_key: str | None) -> OpenAI:
    base = base_url or "http://127.0.0.1:1234/v1"
    key = api_key or "lm-studio"
    return OpenAI(base_url=base, api_key=key)
