from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


LOG = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "facebook/dinov3-vits16-pretrain-lvd1689m"
DEFAULT_VOLUME_MODEL_DIR = Path("/runpod-volume/visual_search/models/dinov3-vits16")
VALID_MODEL_DEVICES = {"auto", "cpu", "cuda"}


class ModelLoadError(RuntimeError):
    pass


def resolve_torch_device(torch_module: Any) -> Any:
    requested_device = os.getenv("MODEL_DEVICE", "auto").strip().lower()
    if requested_device not in VALID_MODEL_DEVICES:
        raise ModelLoadError(
            "MODEL_DEVICE must be one of: auto, cpu, cuda"
        )

    if requested_device == "cpu":
        return torch_module.device("cpu")

    if requested_device == "cuda":
        if not torch_module.cuda.is_available():
            raise ModelLoadError("MODEL_DEVICE=cuda was requested, but CUDA is not available")
        return torch_module.device("cuda")

    return torch_module.device("cuda" if torch_module.cuda.is_available() else "cpu")


def extract_embeddings(outputs: object):
    import torch

    if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
        return outputs.last_hidden_state[:, 0, :]
    if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
        return outputs.pooler_output
    if isinstance(outputs, tuple) and len(outputs) > 0 and torch.is_tensor(outputs[0]):
        first = outputs[0]
        if first.ndim == 3:
            return first[:, 0, :]
        if first.ndim == 2:
            return first
    raise ValueError("Unsupported model output format for embedding extraction.")


class ImageEmbedder:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        volume_model_dir: str | Path = DEFAULT_VOLUME_MODEL_DIR,
    ) -> None:
        self.model_name = model_name
        self.volume_model_dir = Path(volume_model_dir)
        self.device: Any = None
        self.model: Any = None
        self.processor: Any = None
        self._embedding_dim: int | None = None

    @property
    def is_loaded(self) -> bool:
        return self.model is not None and self.processor is not None

    @property
    def embedding_dim(self) -> int | None:
        return self._embedding_dim

    def _load_pair(self, source: str | Path, **kwargs: Any) -> tuple[Any, Any]:
        from transformers import AutoImageProcessor, AutoModel

        processor = AutoImageProcessor.from_pretrained(source, **kwargs)
        model = AutoModel.from_pretrained(source, **kwargs)
        return processor, model

    def _load_from_candidates(self) -> tuple[Any, Any]:
        explicit_path = os.getenv("DINO_MODEL_PATH")
        candidates: list[tuple[str | Path, dict[str, Any], str]] = []

        if explicit_path:
            explicit = Path(explicit_path)
            kwargs = {"local_files_only": True} if explicit.exists() else {}
            candidates.append((explicit_path, kwargs, "DINO_MODEL_PATH"))

        if self.volume_model_dir.exists():
            if (self.volume_model_dir / "config.json").exists():
                candidates.append(
                    (
                        self.volume_model_dir,
                        {"local_files_only": True},
                        "volume model directory",
                    )
                )
            candidates.append(
                (
                    self.model_name,
                    {
                        "cache_dir": str(self.volume_model_dir),
                        "local_files_only": True,
                    },
                    "volume Hugging Face cache",
                )
            )

        candidates.append((self.model_name, {}, "model name"))

        errors: list[str] = []
        for source, kwargs, label in candidates:
            try:
                LOG.info("Loading DINOv3 model from %s", label)
                return self._load_pair(source, **kwargs)
            except Exception as exc:
                errors.append(f"{label}: {exc}")
                LOG.warning("Could not load DINOv3 model from %s: %s", label, exc)

        raise ModelLoadError("Could not load DINOv3 model. " + " | ".join(errors))

    def load(self) -> None:
        if self.is_loaded:
            return

        try:
            import torch
        except ImportError as exc:
            raise ModelLoadError(
                "Embedding requires torch and transformers in the runtime environment."
            ) from exc

        try:
            self.device = resolve_torch_device(torch)
            self.processor, self.model = self._load_from_candidates()
            self.model = self.model.to(self.device)
            self.model.eval()
            LOG.info("DINOv3 model loaded on %s", self.device)
        except ModelLoadError:
            raise
        except Exception as exc:
            self.model = None
            self.processor = None
            raise ModelLoadError(f"Could not initialize DINOv3 model: {exc}") from exc

    def embed_image(self, image: Image.Image) -> np.ndarray:
        self.load()

        if self.model is None or self.processor is None or self.device is None:
            raise ModelLoadError("DINOv3 model is not loaded.")

        import torch

        inputs = self.processor(images=[image.convert("RGB")], return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.inference_mode():
            outputs = self.model(**inputs)
            embedding_tensor = extract_embeddings(outputs)

        embedding = embedding_tensor[0].detach().cpu().numpy().astype(np.float32, copy=False)
        norm = float(np.linalg.norm(embedding))
        if norm <= 1e-12:
            raise ValueError("DINOv3 returned a zero-norm embedding.")

        embedding = (embedding / norm).astype(np.float32, copy=False)
        self._embedding_dim = int(embedding.shape[0])
        return embedding
