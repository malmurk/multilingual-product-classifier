# Production wrapper

> ⚠️ **Partly superseded (Phase L.6, 2026-05-11).** The *shipped* service
> (wrapper 3.0.1) now runs **INT8 ONNX on CPU only** — torch/CUDA/bf16 removed
> from the serving path; T=0.5 calibration carries over. For the current
> deploy/server-spec/cost story see the internal deployment runbook (not included in this repository).
> The GPU/PyTorch details below describe the older dev path.

FastAPI + Prometheus + Grafana stack wrapping the v3 hierarchical classifier.
Lives at `<data-boost-project>/production/`.

## What it provides

- REST API: `POST /predict`, `POST /predict/batch`, `GET /health`, `GET /metrics`
- Calibrated inference baked in: bf16 autocast + SDPA attention + temperature 0.5
- Optional E5 retrieval fallback (toggle via `RETRIEVAL_FALLBACK=true`)
- Prometheus metrics: request count, latency histogram, confidence distribution,
  auto-assign rate, retrieval-fired count
- Grafana dashboard auto-provisioned via Docker Compose
- Pytest smoke tests (6 tests, all passing on real GPU inference)

## Why a separate wrapper instead of extending `gated_predictor.py`

`gated_predictor.py` in this project is the legacy ONNX-based predictor with
the broken fp32+eager calibration baked in. The production wrapper:
1. Uses PT checkpoints directly (skips ONNX export — easier to update)
2. Has the calibration bug fixed (bf16+SDPA+T=0.5)
3. Adds REST + monitoring + Docker that the parent project lacks
4. Doesn't touch the legacy production path until user authorizes the swap

Eventually `gated_predictor.py` should be retired and ONNX export updated
to match the new calibration. That's part of Phase L of the correction plan.

## Architecture

```
                    ┌────────────────────────────────────────┐
                    │          FastAPI (port 8000)           │
                    │                                         │
   ┌────────┐       │  /predict  ──┐                          │
   │ client │ ─────►│  /predict/batch ──► InferenceEngine     │
   └────────┘       │  /health     │       │                  │
                    │  /metrics ──►│       ▼                  │
                    │              │   ┌──────────────┐       │
                    │              │   │  v3 PT model │       │
                    │              │   │  (3 stages)  │       │
                    │              │   └──────┬───────┘       │
                    │              │          │               │
                    │              │   if conf < gate:        │
                    │              │          ▼               │
                    │              │   ┌──────────────┐       │
                    │              │   │  E5 retrieval │ (opt) │
                    │              │   └──────────────┘       │
                    │              │                          │
                    │   Prometheus  ──► /metrics scrape       │
                    └──────────────┴──────────────────────────┘
                                          │
                                          ▼
                              ┌──────────────────────┐
                              │  Prometheus + Grafana │
                              │     (Docker stack)    │
                              └──────────────────────┘
```

## Where to deploy

**Direct uvicorn (local dev, incl. Windows + ROCm GPU):**

```bash
cd "<data-boost-project>"
python -m uvicorn production.app.main:app \
    --host 0.0.0.0 --port 8000
```

**Docker Compose (Linux production target):**

```bash
cd "<data-boost-project>/production"
docker compose up -d --build
```

Services: API (8000), Prometheus (9090), Grafana (3000).

## Calibration details — DO NOT change casually

The defaults (bf16 + SDPA + T=0.5) come from the 2026-05-10 audit. See
the internal calibration decision record (not included in this repository) for the full rationale. To
change them, run the val_pruned precision sweep (methodology in
the precision-sweep script in `<data-boost-project>`, not included here) and update the calibration note.

## Observability

Prometheus metrics exposed at `/metrics`:

| Metric | Labels | What it tracks |
|---|---|---|
| `classifier_predict_requests_total` | endpoint, outcome | Auto/manual/error counts |
| `classifier_inference_seconds` | endpoint | Latency histogram |
| `classifier_prediction_confidence` | strategy | Confidence distribution |
| `classifier_auto_assign_rate` | — | Auto-assign rate (last batch gauge) |
| `classifier_model_loaded` | — | 1 if engine ready |
| `classifier_retrieval_used_total` | outcome | E5-fallback fire count |

Grafana dashboard panels: auto-assign rate stat, model-loaded indicator,
request rate per outcome, latency p50/p95/p99 timeseries, confidence
heatmap, retrieval-fired stat.

## What's NOT in the wrapper (yet)

- Brand catalog gate (Phase L)
- Human review queue UI (Phase L)
- Multimodal image features (Phase M)
- ROCm Docker base image (Linux/CPU only currently — see `Dockerfile` TODO)
