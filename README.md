# multilingual-product-classifier

Three-stage multilingual transformer cascade that maps free-text product listings (Russian / Romanian titles, brands, descriptions) onto a 3,000+-leaf category taxonomy, with a confidence-gated split between automatic assignment and a human-review queue.

Built for a production e-commerce catalog. The client is an Eastern-European marketplace (reference under NDA); code is published with their data, weights, and taxonomy removed. This is a scrubbed re-publication: the original development history contained client-identifying data, so it was squashed for the public release.

## Problem

Marketplace catalogs grow faster than anyone can categorize them. Supplier feeds arrive as messy free text — mixed Russian and Romanian, inconsistent brand naming, no attributes, no category hints — and every uncategorized SKU is invisible to filtered search and category browsing. Manual sorting is the default answer, and it does not scale: a taxonomy with thousands of leaf categories means even a trained human spends real time per product, and a 500k-SKU catalog means years of that time.

The engineering problem underneath: a flat classifier over 3,000+ classes is both slow to converge and unreliable at the tail, while the taxonomy's tree structure (super-category → parent → leaf) is known, machine-readable, and free to exploit.

## Results

| Metric | Value |
|---|---|
| Auto-classification rate | 90.6% on a 500k-SKU production catalog at an Eastern-European marketplace (reference under NDA) |
| Cost | ~$0.05 per 1,000 SKUs |
| Auto-assign threshold | leaf confidence ≥ 0.93 → assigned automatically; below → human-review queue |

Products below the threshold are not dropped — they go to a `manual_sorting` queue with the model's full prediction path (super / parent / leaf labels, confidences, and a machine-readable reason), so a reviewer confirms a suggestion instead of classifying from scratch. Every decision, auto or manual, is appended to a corrections log that feeds the retraining loop.

The 0.93 threshold is deliberately conservative and re-evaluated after every retrain: growing the leaf head or shifting the training-data mix changes the confidence distribution, so the threshold is swept against a held-out set before each deployment rather than trusted indefinitely.

## Architecture

Three independent classifiers share one encoder architecture and run as a cascade at inference time.

**Per-stage model** (`src/model.py:CategoryClassifier`):

- Encoder: `intfloat/multilingual-e5-base` — XLM-RoBERTa-base architecture (12 layers, 768 hidden)
- Pooling: attention-mask-weighted mean over all token embeddings (pad tokens contribute nothing)
- Head: `Dropout(0.1)` → `Linear(768, num_classes)`

**Stages:** super (~20 classes) → parent (a few hundred) → leaf (3,000+, varies by taxonomy version). Each stage has its own checkpoint; the leaf stage warm-starts from the parent stage's fine-tuned encoder (`--init-encoder-from`) instead of re-learning from the pretrained base.

**Structural masking at inference.** Each stage constrains the next: after the super stage picks a label, every parent logit that is not a child of that super is set to `NEG_INF` (-1e10) before softmax; the parent's prediction masks the leaf logits the same way. A 3,000+-way leaf head therefore only ever competes among the ~10–20 leaves that are structurally valid under the predicted parent. One caveat this design carries: post-mask confidences are renormalized within the surviving branch, so "0.95" means 95% among the valid candidates, not among the full class space — the auto-assign threshold is calibrated against exactly this quantity.

**Flat softmax during training.** The masking is inference-only. Each stage trains with plain cross-entropy over all of its classes, so the encoder learns to separate every leaf globally rather than only within pre-narrowed branches.

**Production runtime.** Checkpoints are exported to ONNX and dynamically quantized to INT8 (`src/export_onnx.py`, `onnxruntime.quantization.quantize_dynamic` with `QuantType.QInt8`); inference runs one `onnxruntime.InferenceSession` per stage on CPU. The export path traces a wrapper module that calls the encoder's embedding and layer stack directly with a legacy additive attention mask, because transformers ≥ 4.46 SDPA mask helpers break under `torch.jit.trace` — numerically identical to the training forward, weights untouched.

**Serving** (`classifier/`): a FastAPI service and a DB worker, two containers off one image.

- `service.py` — `POST /classify` (single product, synchronous), `POST /classify/batch` (up to 500), `GET /health`, `GET /hierarchy` (taxonomy tree for review-UI dropdowns).
- `worker.py` — polls an `unsorted` queue table in your database, classifies in batches, writes `products.category_id` directly for predictions at or above the auto-threshold (after resolving the predicted leaf name to a live category id through an hourly-refreshed cache), and routes everything else to `manual_sorting` with a reason (`non_leaf`, `low_confidence`, `leaf_not_in_db`). It also sweeps the manual queue each tick, removing rows a human has already resolved, and shuts down cleanly on SIGTERM/SIGINT.

```
                     product text (title + brand + attrs + description)
                                        │
                                        ▼
                          ┌─────────────────────────┐
                          │  Stage 1: super (~20)   │  full softmax
                          └────────────┬────────────┘
                                       │ argmax → super label
                                       ▼
                          ┌─────────────────────────┐
                          │  Stage 2: parent        │  logits of parents NOT under
                          │  (a few hundred)        │  the super → NEG_INF, then softmax
                          └────────────┬────────────┘
                                       │ argmax → parent label
                                       ▼
                          ┌─────────────────────────┐
                          │  Stage 3: leaf (3,000+) │  logits of leaves NOT under
                          │                         │  the parent → NEG_INF, then softmax
                          └────────────┬────────────┘
                                       │ leaf label + confidence
                                       ▼
                     ┌─────────────────┴──────────────────┐
                     │ conf ≥ 0.93                        │ conf < 0.93 (or non-leaf stop,
                     ▼                                    ▼  or leaf unknown to the DB)
          UPDATE products.category_id          INSERT INTO manual_sorting
              (auto-assigned)                     (human-review queue)
```

## Repository layout

```
├── src/                     # training + offline evaluation (PyTorch)
│   ├── model.py             # CategoryClassifier: encoder + masked mean-pool + linear head
│   ├── train.py             # per-stage trainer: bf16 autocast, warmup+cosine, early stopping
│   ├── dataset.py           # JSONL dataset + tokenization
│   ├── taxonomy.py          # taxonomy CSV loader (super/parent/leaf tree)
│   ├── predict.py           # cascaded masked prediction against PyTorch checkpoints
│   ├── export_onnx.py       # trace-based ONNX export + dynamic INT8 quantization
│   ├── evaluate.py          # held-out metrics per stage
│   └── ...                  # data building, augmentation, eval experiments
├── classifier/              # production service (ONNX Runtime, no PyTorch dependency)
│   ├── predictor.py         # three-stage masked predictor over InferenceSessions
│   ├── hierarchy.py         # taxonomy tree + masked_softmax
│   ├── preprocessor.py      # builds the model input text from product fields
│   ├── service.py           # FastAPI: /classify, /classify/batch, /health, /hierarchy
│   ├── worker.py            # DB queue worker: auto-assign vs manual_sorting routing
│   ├── docker-compose.yml   # api + worker containers off one image
│   └── .env.example
├── docs/
│   ├── architecture/        # pipeline overview, data flow, text format, production wrapper
│   └── features/            # cascaded prediction, auto-assign threshold, class weights
├── tests/                   # pytest: hierarchy masking, taxonomy, ONNX export parity,
│                            #         worker resolver, dataset/preprocess/train units
├── requirements*.txt        # training / inference / docker splits
└── docker-compose.yml       # training container
```

## Training

Training ran on a single consumer AMD GPU via ROCm — no CUDA hardware, no cloud training budget. The trainer accounts for the platform explicitly:

- One dummy forward+backward before epoch 1 to trigger MIOpen HIP-kernel compilation (`MIOPEN_FIND_MODE=FAST`), so the first real batch isn't a multi-minute stall.
- bf16 autocast throughout, with a non-finite-loss guard that skips a diverged batch instead of poisoning the run.

The parts that made a 3,000+-way head trainable at all:

- **Split learning rates:** the fresh classifier head trains at 50× the encoder LR (`--head-lr-mult`) — millions of randomly-initialized head parameters need far more gradient signal than a pretrained encoder should receive.
- **Step-level linear warmup → cosine decay**, so the first few hundred updates on a huge fresh head don't diverge.
- **Leaf warm-start:** the leaf stage copies the parent stage's fine-tuned encoder weights before training (`--init-encoder-from`), skipping a full re-fine-tune of the backbone.
- Gradient clipping, early stopping on validation loss, resumable checkpoints with optimizer state.

```bash
python -m src.train --stage super  --epochs 20
python -m src.train --stage parent --epochs 20
python -m src.train --stage leaf   --epochs 20 \
    --init-encoder-from models/checkpoints/parent/best_model.pt \
    --head-lr-mult 50
python -m src.export_onnx --stage leaf --checkpoint models/checkpoints/leaf/best_model.pt
```

## What is not included

This repository is the code, not the model. The following are client property and are **not included in this repository**:

- Trained model weights and ONNX exports
- Taxonomy CSVs (`taxonomy_full.csv`, `taxonomy_live.csv`, `taxonomy_pruned.csv`, `taxonomy_universe.csv`, `taxonomy_v3.csv`)
- Training data, evaluation sets, and database dumps

To run the pipeline end to end you would supply your own taxonomy CSV and labeled product data in the documented formats (`docs/architecture/text-format.md`), train the three stages, and export. The code paths are complete; the artifacts are yours to produce.

## Running the service

The production wrapper under `classifier/` expects trained ONNX models in a mounted directory and (for the worker) a MariaDB/MySQL database with `products`, `categories`, and the two queue tables (`unsorted`, `manual_sorting`).

```bash
cd classifier
cp .env.example .env      # set DB_URL to your database, adjust thresholds if needed
docker-compose up -d      # starts: api (127.0.0.1:8000) + worker
```

- The API binds to localhost only — it is designed to be called by a backend on the same host, not exposed publicly.
- Models are volume-mounted rather than baked into the image, so a retrain deploys as file sync + container restart, not an image rebuild.
- All knobs are environment variables (see `classifier/.env.example`): `THRESHOLD_LEAF_AUTO` (default 0.93), worker `BATCH_SIZE` / `POLL_INTERVAL`, resolver refresh interval, corrections-log path.

Smoke test:

```bash
curl -s localhost:8000/health
curl -s localhost:8000/classify -X POST -H 'Content-Type: application/json' \
     -d '{"title": "Ноутбук Lenovo IdeaPad 3 15.6\" 8GB RAM"}'
```

## License

MIT.
