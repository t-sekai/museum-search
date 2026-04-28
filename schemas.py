from __future__ import annotations

import re
from typing import List, Optional
from urllib.parse import urlparse

from pydantic import BaseModel, Field, validator


SAFE_MUSEUM_SLUG_RE = re.compile(r"^[a-z0-9_-]+$")


def validate_museum_slug_value(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("museum_slug must be a string")

    slug = value.strip()
    if not slug:
        raise ValueError("museum_slug is required")
    if not SAFE_MUSEUM_SLUG_RE.fullmatch(slug):
        raise ValueError(
            "museum_slug may only contain lowercase letters, numbers, underscores, and hyphens"
        )
    return slug


def validate_http_url_value(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("image_url must be a string")

    url = value.strip()
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("image_url must be an http or https URL")
    return url


class SearchRequest(BaseModel):
    museum_slug: str
    image_url: str
    top_k: int = Field(default=10, ge=1, le=50)

    @validator("museum_slug")
    def validate_museum_slug(cls, value: str) -> str:
        return validate_museum_slug_value(value)

    @validator("image_url")
    def validate_image_url(cls, value: str) -> str:
        return validate_http_url_value(value)

    class Config:
        extra = "forbid"


class SearchResult(BaseModel):
    artwork_key: str
    object_id: str
    title: Optional[str] = None
    artist_display_name: Optional[str] = None
    primary_image: Optional[str] = None
    primary_image_small: Optional[str] = None
    image_url_used: Optional[str] = None
    source_url: Optional[str] = None
    score: float


class SearchResponse(BaseModel):
    museum_slug: str
    index_version: str
    embedding_model: str
    top_k: int
    results: List[SearchResult]

