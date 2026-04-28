from __future__ import annotations

import logging
import os

from fastapi import FastAPI, HTTPException

from image_io import (
    ImageDownloadError,
    ImageDownloadTimeout,
    ImageIOError,
    ImageTooLargeError,
    InvalidImageError,
    load_image_from_url,
)
from index_store import (
    InvalidMuseumSlug,
    MuseumArtifactNotFoundError,
    MuseumIndexLoadError,
    MuseumIndexStore,
    MuseumNotFoundError,
)
from model import DEFAULT_MODEL_NAME, ImageEmbedder, ModelLoadError
from schemas import SearchRequest, SearchResponse


logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
LOG = logging.getLogger(__name__)

VOLUME_ROOT = os.getenv("VISUAL_SEARCH_VOLUME_ROOT", "/runpod-volume/visual_search")
MUSEUM_ROOT = os.getenv(
    "VISUAL_SEARCH_MUSEUM_ROOT",
    os.path.join(VOLUME_ROOT, "museums"),
)
MAX_IMAGE_BYTES = int(os.getenv("MAX_IMAGE_DOWNLOAD_BYTES", "10000000"))
IMAGE_TIMEOUT_SECONDS = int(os.getenv("IMAGE_DOWNLOAD_TIMEOUT_SECONDS", "15"))
PRELOAD_MODEL = os.getenv("PRELOAD_MODEL", "1").lower() not in {"0", "false", "no"}

app = FastAPI(title="Runpod Visual Artwork Search Worker")
embedder = ImageEmbedder(model_name=os.getenv("DINO_MODEL_NAME", DEFAULT_MODEL_NAME))
index_store = MuseumIndexStore(root=MUSEUM_ROOT)


@app.on_event("startup")
async def startup() -> None:
    if not PRELOAD_MODEL:
        LOG.info("Skipping DINOv3 startup preload because PRELOAD_MODEL is disabled")
        return

    try:
        embedder.load()
    except Exception:
        LOG.exception("DINOv3 model preload failed; /search will retry and return 500 if it still fails")


@app.get("/health")
async def health() -> dict[str, object]:
    volume_diagnostics = index_store.diagnostics()
    return {
        "status": "healthy",
        "model_loaded": embedder.is_loaded,
        "volume_root_exists": os.path.exists(VOLUME_ROOT),
        "volume_root": VOLUME_ROOT,
        "museum_root": MUSEUM_ROOT,
        "museum_root_exists": volume_diagnostics["museum_root_exists"],
        "loaded_museums": index_store.loaded_museums(),
    }


@app.get("/museums")
async def museums() -> dict[str, list[str]]:
    return {"available_museums": index_store.available_museums()}


@app.get("/debug/volume")
async def debug_volume() -> dict[str, object]:
    return {
        "volume_root": VOLUME_ROOT,
        "volume_root_exists": os.path.exists(VOLUME_ROOT),
        **index_store.diagnostics(),
    }


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest) -> SearchResponse:
    try:
        museum_index = index_store.get(request.museum_slug)
    except InvalidMuseumSlug as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (MuseumNotFoundError, MuseumArtifactNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except MuseumIndexLoadError as exc:
        LOG.exception("Museum index loading failed for museum=%s", request.museum_slug)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        LOG.exception("Unexpected museum index loading failure for museum=%s", request.museum_slug)
        raise HTTPException(status_code=500, detail="museum index loading failed") from exc

    try:
        image = load_image_from_url(
            request.image_url,
            max_bytes=MAX_IMAGE_BYTES,
            timeout=IMAGE_TIMEOUT_SECONDS,
        )
    except (ImageDownloadTimeout, ImageTooLargeError, InvalidImageError, ImageDownloadError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ImageIOError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        query_embedding = embedder.embed_image(image)
    except (ModelLoadError, ValueError) as exc:
        LOG.exception("Model embedding failed")
        raise HTTPException(status_code=500, detail=f"model embedding failed: {exc}") from exc
    except Exception as exc:
        LOG.exception("Unexpected model embedding failure")
        raise HTTPException(status_code=500, detail="model embedding failed") from exc

    try:
        scores, ids = museum_index.search(query_embedding, request.top_k)
        results = museum_index.lookup(ids, scores)
    except Exception as exc:
        LOG.exception("FAISS search failed for museum=%s", request.museum_slug)
        raise HTTPException(status_code=500, detail=f"museum search failed: {exc}") from exc

    return SearchResponse(
        museum_slug=request.museum_slug,
        index_version=museum_index.version,
        embedding_model=museum_index.embedding_model or embedder.model_name,
        top_k=request.top_k,
        results=results,
    )


@app.post("/reload")
async def reload() -> dict[str, str]:
    index_store.clear()
    return {"status": "cleared"}
