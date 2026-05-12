# CPU Online Service

This packaging path runs only the online FastAPI search service on CPU. It does
not include the offline indexing scripts.

Build from the `museum-search` directory:

```bash
docker build -f Dockerfile.cpu -t museum-search:cpu .
```

Run with the same museum artifact volume used by the GPU image:

```bash
docker run --rm -p 8000:8000 \
  -e VISUAL_SEARCH_VOLUME_ROOT=/runpod-volume/visual_search \
  -e VISUAL_SEARCH_MUSEUM_ROOT=/runpod-volume/visual_search/museums \
  -v /path/to/visual_search:/runpod-volume/visual_search \
  museum-search:cpu
```

The CPU image sets `MODEL_DEVICE=cpu`. Other supported values are:

- `MODEL_DEVICE=auto`: use CUDA if available, otherwise CPU.
- `MODEL_DEVICE=cuda`: require CUDA and fail startup/search if unavailable.

For CPU deployments, keep the DINO model cached or mounted through
`DINO_MODEL_PATH` or `/runpod-volume/visual_search/models/dinov3-vits16` to avoid
slow cold starts.
