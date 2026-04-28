from __future__ import annotations

from io import BytesIO
from urllib.parse import urlparse

import requests
from PIL import Image, UnidentifiedImageError


class ImageIOError(Exception):
    """Base exception for image download and decode failures."""


class ImageDownloadTimeout(ImageIOError):
    pass


class ImageDownloadError(ImageIOError):
    pass


class ImageTooLargeError(ImageIOError):
    pass


class InvalidImageError(ImageIOError):
    pass


def _validate_url(url: str) -> str:
    candidate = str(url).strip()
    parsed = urlparse(candidate)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise InvalidImageError("image_url must be an http or https URL")
    return candidate


def load_image_from_url(
    url: str,
    max_bytes: int = 10_000_000,
    timeout: int = 15,
) -> Image.Image:
    """Stream an image download, enforce a size limit, and decode it as RGB."""

    image_url = _validate_url(url)
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")

    try:
        with requests.get(image_url, stream=True, timeout=timeout) as response:
            if response.status_code != 200:
                raise ImageDownloadError(
                    f"image download returned HTTP {response.status_code}"
                )

            content_type = response.headers.get("content-type")
            if content_type:
                media_type = content_type.split(";", 1)[0].strip().lower()
                if media_type and not media_type.startswith("image/"):
                    raise InvalidImageError(
                        f"image download returned non-image content-type: {media_type}"
                    )

            content_length = response.headers.get("content-length")
            if content_length:
                try:
                    if int(content_length) > max_bytes:
                        raise ImageTooLargeError(
                            f"image exceeds maximum size of {max_bytes} bytes"
                        )
                except ValueError:
                    pass

            buffer = BytesIO()
            total_bytes = 0
            for chunk in response.iter_content(chunk_size=64 * 1024):
                if not chunk:
                    continue
                total_bytes += len(chunk)
                if total_bytes > max_bytes:
                    raise ImageTooLargeError(
                        f"image exceeds maximum size of {max_bytes} bytes"
                    )
                buffer.write(chunk)
    except ImageIOError:
        raise
    except requests.Timeout as exc:
        raise ImageDownloadTimeout("image download timed out") from exc
    except requests.RequestException as exc:
        raise ImageDownloadError("image download failed") from exc

    try:
        buffer.seek(0)
        image = Image.open(buffer)
        image.load()
        return image.convert("RGB")
    except (OSError, UnidentifiedImageError) as exc:
        raise InvalidImageError("downloaded content is not a valid image") from exc

