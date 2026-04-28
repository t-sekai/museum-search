# Runpod Visual Search Worker

FastAPI worker for Runpod load-balanced serverless visual artwork search.

## Local Run

```bash
cd runpod_worker
uvicorn app:app --host 0.0.0.0 --port 8000
```

## Health Check

```bash
curl http://localhost:8000/health
```

## List Museums

```bash
curl http://localhost:8000/museums
```

## Search Test

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{
    "museum_slug": "met",
    "image_url": "https://example.com/test.jpg",
    "top_k": 5
  }'
```

The worker expects artifacts under `/runpod-volume/visual_search/museums`.
Each request searches exactly one museum index selected by `museum_slug`.

