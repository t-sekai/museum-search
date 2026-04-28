from __future__ import annotations

import json
import logging
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from schemas import SAFE_MUSEUM_SLUG_RE, validate_museum_slug_value


LOG = logging.getLogger(__name__)
SAFE_VERSION_RE = re.compile(r"^[A-Za-z0-9_.-]+$")

REQUIRED_ARTWORK_COLUMNS = {
    "museum_slug",
    "object_id",
    "artwork_key",
    "faiss_id",
    "embedding_row",
    "title",
    "artist_display_name",
    "primary_image",
    "primary_image_small",
    "image_url_used",
    "source_url",
    "embedding_model",
    "embedding_dim",
    "index_version",
    "is_searchable",
}


class InvalidMuseumSlug(ValueError):
    pass


class MuseumNotFoundError(FileNotFoundError):
    pass


class MuseumArtifactNotFoundError(FileNotFoundError):
    pass


class MuseumIndexLoadError(RuntimeError):
    pass


def _json_scalar(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, np.generic):
        return value.item()
    return value


def _optional_string(value: Any) -> str | None:
    value = _json_scalar(value)
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _first_non_empty(rows: Iterable[dict[str, Any]], column: str) -> str | None:
    for row in rows:
        value = _optional_string(row.get(column))
        if value:
            return value
    return None


@dataclass(frozen=True)
class MuseumIndex:
    museum_slug: str
    version: str
    index: Any
    artworks: pd.DataFrame
    manifest: dict[str, Any]
    by_faiss_id: dict[int, dict[str, Any]]
    embedding_model: str
    embedding_dim: int | None

    def search(self, query_embedding: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        query = np.asarray(query_embedding, dtype=np.float32)
        if query.ndim == 1:
            query = query.reshape(1, -1)
        if query.ndim != 2 or query.shape[0] != 1:
            raise ValueError(f"Expected query embedding shape (dim,) or (1, dim), got {query.shape}")

        index_dim = getattr(self.index, "d", None)
        if index_dim is not None and int(index_dim) != int(query.shape[1]):
            raise ValueError(
                f"Query embedding dimension {query.shape[1]} does not match index dimension {index_dim}"
            )

        scores, ids = self.index.search(query.astype("float32", copy=False), int(top_k))
        return scores[0], ids[0]

    def lookup(self, ids: np.ndarray, scores: np.ndarray) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for raw_id, raw_score in zip(ids, scores):
            faiss_id = int(raw_id)
            if faiss_id == -1:
                continue

            row = self.by_faiss_id.get(faiss_id)
            if row is None:
                LOG.warning("FAISS returned unknown id %s for museum %s", faiss_id, self.museum_slug)
                continue

            results.append(
                {
                    "artwork_key": str(_json_scalar(row.get("artwork_key"))),
                    "object_id": str(_json_scalar(row.get("object_id"))),
                    "title": _optional_string(row.get("title")),
                    "artist_display_name": _optional_string(row.get("artist_display_name")),
                    "primary_image": _optional_string(row.get("primary_image")),
                    "primary_image_small": _optional_string(row.get("primary_image_small")),
                    "image_url_used": _optional_string(row.get("image_url_used")),
                    "source_url": _optional_string(row.get("source_url")),
                    "score": float(raw_score),
                }
            )
        return results


class MuseumIndexStore:
    def __init__(self, root: str = "/runpod-volume/visual_search/museums") -> None:
        self.root = Path(root)
        self.cache: dict[str, MuseumIndex] = {}
        self._lock = threading.RLock()

    def _safe_slug(self, museum_slug: str) -> str:
        try:
            return validate_museum_slug_value(museum_slug)
        except ValueError as exc:
            raise InvalidMuseumSlug(str(exc)) from exc

    def _museum_dir(self, museum_slug: str) -> Path:
        slug = self._safe_slug(museum_slug)
        root = self.root.resolve()
        candidate = (self.root / slug).resolve()
        if candidate.parent != root:
            raise InvalidMuseumSlug("museum_slug resolved outside the museum artifact root")
        return candidate

    def _safe_version(self, active_version: Any) -> str:
        if not isinstance(active_version, str):
            raise MuseumIndexLoadError("active_version must be a string")
        version = active_version.strip()
        if not version or not SAFE_VERSION_RE.fullmatch(version):
            raise MuseumIndexLoadError("active_version contains invalid path characters")
        return version

    def available_museums(self) -> list[str]:
        if not self.root.exists() or not self.root.is_dir():
            LOG.warning("Museum root does not exist or is not a directory: %s", self.root)
            return []

        museums = []
        for child in self.root.iterdir():
            if child.is_dir() and (child / "current.json").is_file():
                try:
                    museums.append(self._safe_slug(child.name))
                except InvalidMuseumSlug:
                    LOG.warning("Skipping unsafe museum directory name: %s", child.name)
        return sorted(museums)

    def diagnostics(self) -> dict[str, Any]:
        root_exists = self.root.exists()
        root_is_dir = self.root.is_dir()
        children: list[dict[str, Any]] = []

        if root_exists and root_is_dir:
            for child in sorted(self.root.iterdir(), key=lambda path: path.name):
                if not child.is_dir():
                    continue
                children.append(
                    {
                        "name": child.name,
                        "safe_slug": bool(SAFE_MUSEUM_SLUG_RE.fullmatch(child.name)),
                        "has_current_json": (child / "current.json").is_file(),
                        "has_versions_dir": (child / "versions").is_dir(),
                    }
                )

        return {
            "museum_root": str(self.root),
            "museum_root_exists": root_exists,
            "museum_root_is_dir": root_is_dir,
            "children": children,
        }

    def loaded_museums(self) -> list[str]:
        with self._lock:
            return sorted(self.cache.keys())

    def clear(self) -> None:
        with self._lock:
            self.cache.clear()

    def get(self, museum_slug: str) -> MuseumIndex:
        slug = self._safe_slug(museum_slug)
        with self._lock:
            cached = self.cache.get(slug)
            if cached is not None:
                return cached

            loaded = self._load(slug)
            self.cache[slug] = loaded
            return loaded

    def _load(self, museum_slug: str) -> MuseumIndex:
        museum_dir = self._museum_dir(museum_slug)
        current_path = museum_dir / "current.json"
        if not current_path.is_file():
            raise MuseumNotFoundError(f"unknown museum: {museum_slug}")

        try:
            current = json.loads(current_path.read_text(encoding="utf-8"))
            active_version = self._safe_version(current["active_version"])
        except KeyError as exc:
            raise MuseumIndexLoadError(f"{current_path} is missing active_version") from exc
        except Exception as exc:
            raise MuseumIndexLoadError(f"could not read {current_path.name}: {exc}") from exc

        version_dir = museum_dir / "versions" / active_version
        index_path = version_dir / "index.faiss"
        artworks_path = version_dir / "artworks.parquet"
        manifest_path = version_dir / "manifest.json"

        missing = [
            path.name
            for path in (index_path, artworks_path, manifest_path)
            if not path.is_file()
        ]
        if missing:
            raise MuseumArtifactNotFoundError(
                f"missing artifact(s) for museum {museum_slug}: {', '.join(missing)}"
            )

        try:
            import faiss

            index = faiss.read_index(str(index_path))
        except Exception as exc:
            raise MuseumIndexLoadError(f"could not load FAISS index for {museum_slug}: {exc}") from exc

        try:
            artworks = pd.read_parquet(artworks_path)
        except Exception as exc:
            raise MuseumIndexLoadError(
                f"could not load artwork metadata for {museum_slug}: {exc}"
            ) from exc

        missing_columns = sorted(REQUIRED_ARTWORK_COLUMNS.difference(artworks.columns))
        if missing_columns:
            raise MuseumIndexLoadError(
                f"artworks.parquet is missing required columns: {', '.join(missing_columns)}"
            )

        artworks = artworks.copy()
        try:
            if artworks["faiss_id"].isna().any():
                raise ValueError("faiss_id contains null values")
            artworks["faiss_id"] = pd.to_numeric(
                artworks["faiss_id"], errors="raise"
            ).astype("int64")
        except Exception as exc:
            raise MuseumIndexLoadError(f"faiss_id must be convertible to int64: {exc}") from exc

        if artworks["faiss_id"].duplicated().any():
            raise MuseumIndexLoadError("artworks.parquet contains duplicate faiss_id values")

        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise MuseumIndexLoadError(f"could not load manifest for {museum_slug}: {exc}") from exc

        rows = artworks.to_dict(orient="records")
        by_faiss_id = {int(row["faiss_id"]): row for row in rows}

        embedding_model = (
            _optional_string(manifest.get("embedding_model"))
            or _first_non_empty(rows, "embedding_model")
            or ""
        )
        embedding_dim_value = _json_scalar(manifest.get("embedding_dim"))
        if embedding_dim_value is None:
            embedding_dim_value = _json_scalar(_first_non_empty(rows, "embedding_dim"))
        embedding_dim = int(embedding_dim_value) if embedding_dim_value is not None else None

        LOG.info(
            "Loaded museum index museum=%s version=%s rows=%d ntotal=%s",
            museum_slug,
            active_version,
            len(rows),
            getattr(index, "ntotal", "unknown"),
        )

        return MuseumIndex(
            museum_slug=museum_slug,
            version=active_version,
            index=index,
            artworks=artworks,
            manifest=manifest,
            by_faiss_id=by_faiss_id,
            embedding_model=embedding_model,
            embedding_dim=embedding_dim,
        )
