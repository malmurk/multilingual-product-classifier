# Multilingual product classifier — deployment package

Drop-in hierarchical categorizer for an e-commerce catalog. Takes a product
title (+ optional brand / attributes / description) in Russian or
Romanian and returns a category from the 3-level taxonomy
(illustratively **~20 super → ~100 parent → ~600 leaf** — exact counts vary by taxonomy version) defined in
`taxonomy_live.csv` (your production taxonomy — the source of
truth your backend assigns categories against). Built and proven on a
500k-SKU production catalog at an Eastern-European marketplace
(reference under NDA): 90.6% auto-classification at ~$0.05 per 1,000
SKUs. `taxonomy_live.csv` and the trained model weights are specific
to that deployment and are **not included in this repository** —
bring your own taxonomy and train against it (see the repo root for
the training pipeline).

## What's inside

```
classifier/
  predictor.py       three-stage ONNX inference with logit masking
  preprocessor.py    text cleaning + field concatenation
  hierarchy.py       loads labels + builds the parent/leaf masks
  service.py         FastAPI HTTP endpoint  (real-time)
  worker.py          background DB poller   (bulk imports)
  Dockerfile         python:3.11-slim; final image ~3.5 GB (driven by ONNX models)
  docker-compose.yml runs api + worker together
  requirements.txt
  .env.example
```

## How the narrowing works

This is the part the design spec calls "logit masking" — at each stage
the classifier is physically forbidden from picking anything outside
the branch chosen by the previous stage:

| Stage | Classes in model | Choices after masking |
|-------|------------------|----------------------|
| 1. Super  | 19   | 19 (unrestricted) |
| 2. Parent | ~100 | **~6 on average, ~15 max** — only parents under the predicted super |
| 3. Leaf   | ~600 | **~5 on average** — only leaves under the predicted parent |

Example for `"Компьютеры"`: stage 2 chooses among a handful of parents (not all ~100),
stage 3 among a handful of leaves (not all ~600). Each stage's confidence is
re-normalized within the narrowed set, so thresholds stay meaningful.

## Integration — 3 ways

### 1. HTTP (recommended for the product-import pipeline)

```bash
docker compose up api
```

```bash
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Ноутбук Lenovo IdeaPad 3 15IAU7",
    "brand": "Lenovo",
    "attributes": {"RAM": "16GB", "SSD": "512GB"},
    "description": "15.6 inch, Intel Core i5-1235U"
  }'
```

Response:
```json
{
  "category": "Ноутбуки",
  "category_level": "leaf",
  "needs_review": false,
  "super_label": "Компьютеры",
  "parent_label": "Ноутбуки и аксессуары",
  "leaf_label": "Ноутбуки",
  "super_confidence": 0.99,
  "parent_confidence": 0.94,
  "leaf_confidence": 0.88,
  "stage2_choices": 10,
  "stage3_choices": 7
}
```

`stage2_choices` / `stage3_choices` tell you exactly how much the
taxonomy was narrowed for that product — useful for the admin panel.

### 2. Background worker (queue-driven, for the import pipeline)

The worker is a queue consumer, not a table scanner.  Two new tables
(`unsorted` and `manual_sorting`) drive the flow — create them in your
database with the migration SQL described below (not included in this
repository; it's a couple of `CREATE TABLE` statements against your own
`products`/`categories` schema):

```
                  Your app inserts new product → INSERT INTO unsorted (product_id)
                                              │
                                              ▼
  +------------------------------+    every POLL_INTERVAL (default 1800s):
  |  classifier worker (idle)    | ──▶ SELECT FROM unsorted JOIN products
  +------------------------------+        │
                                          ▼  classify each row
                       ┌──────────────────┴──────────────────┐
                       │                                     │
        leaf conf ≥ 0.93 AND name in DB         leaf conf < 0.93, non-leaf,
                       │                          or unknown leaf name
                       ▼                                     ▼
       UPDATE products.category_id           INSERT INTO manual_sorting
       DELETE FROM unsorted                  DELETE FROM unsorted
                                                            │
                                              dashboard reviewer fixes
                                              products.category_id
                                                            │
                                              next worker tick sweeps:
                                              DELETE FROM manual_sorting
                                              WHERE product is now categorized
```

The auto/manual decision is governed by `THRESHOLD_LEAF_AUTO`
(default `0.93`).  Three cases force a row to `manual_sorting`:

| Reason | When |
|--------|------|
| `low_confidence` | predicted leaf, but confidence < `THRESHOLD_LEAF_AUTO` |
| `non_leaf`       | model couldn't reach a leaf — stage stopped at parent or super |
| `leaf_not_in_db` | model predicted a leaf name that doesn't exist in `categories` (taxonomy drift — retrain or rename) |

Manual rows include the model's best guess (`predicted_category`,
`super_label`, `parent_label`, `leaf_label`, `confidence`) so the
human reviewer sees the narrowed branch instead of the full leaf list.

Every prediction (auto-assigned, manual, or DB error) is also appended
to `corrections.jsonl` — that's the source for the active-learning
retrain loop.

#### Running it

```bash
cp classifier/.env.example classifier/.env
# edit DB_URL  (mysql+pymysql://user:pass@your-db-host:3306/catalog_db?charset=utf8mb4)
# (one-time) create the unsorted/manual_sorting tables on your DB —
# see "Background worker" above for the schema

docker compose --env-file classifier/.env up worker
```

### 3. Python library (if your backend is Python)

```python
from classifier import HierarchicalPredictor

predictor = HierarchicalPredictor(model_dir="models/onnx")
pred = predictor.predict(
    title="Ноутбук Lenovo IdeaPad 3",
    brand="Lenovo",
    attributes={"RAM": "16GB"},
)
print(pred.category, pred.needs_review, pred.leaf_confidence)
```

## Confidence thresholds

The worker has one decision threshold:

| Env var                | Default | Effect |
|------------------------|---------|--------|
| `THRESHOLD_LEAF_AUTO`  | `0.93`  | Leaf confidence ≥ this → auto-assign `products.category_id`. Below → row goes to `manual_sorting`. |

The predictor's per-stage thresholds (`THRESHOLD_SUPER`, `THRESHOLD_PARENT`,
`THRESHOLD_LEAF`) are kept low (`0.50` each) so the model always emits a
leaf-level guess.  That guess is shown to the human reviewer alongside the
super/parent labels, so they can fix it with one click instead of picking
from the full leaf list blind.

The HTTP service (`/classify`) still honours the per-stage thresholds —
it returns `needs_review=true` and falls back to a parent/super category
when stages stop early.  That's separate from the worker's auto-assign
threshold.

Every prediction is appended to `corrections.jsonl` — mount that
volume outside the container and use it to feed the active-learning
retrain loop.

## Model artifacts expected in `MODEL_DIR`

```
super_classifier.onnx     # 19-class   (unrestricted)        — ~1.1 GB INT8
parent_classifier.onnx    # parent-level  (masked at inference) — ~1.1 GB INT8
leaf_classifier.onnx      # leaf-level  (masked at inference) — ~1.1 GB INT8
super_labels.json         # generated from taxonomy_live.csv
parent_labels.json
leaf_labels.json
hierarchy.json            # super_to_parents + parent_to_leaves + idx maps
```

None of these are included in this repository — they're trained
weights and taxonomy-derived sidecars specific to a given deployment.
The four JSON sidecars are produced by `regen_classifier_labels.py` at
the repo root — re-run it any time your taxonomy CSV changes so the
indices stay aligned with the ONNX output dimensions.

## Health check

```bash
curl http://localhost:8000/health        # {"status":"ok"}
curl http://localhost:8000/hierarchy     # full taxonomy, for admin UI dropdowns
```

## Tests

```bash
pytest tests/test_hierarchy_masking.py -v
```

The mask test verifies the key invariant: picking a super at stage 1
narrows stage 2 to only that super's parents and stage 3 to only that
parent's leaves — not the full 196 / 4072.
