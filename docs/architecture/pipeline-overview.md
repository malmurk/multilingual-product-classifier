# Architecture: three-stage cascade pipeline overview

The classifier is three independent models sharing the same XLM-RoBERTa-base encoder structure, run in a cascade at inference time.

## Model structure (per stage)

- **Encoder:** `xlm-roberta-base` (12 layers, 768 hidden dim, ~278M params)
- **Pooling:** attention-mask-weighted mean pool over all token embeddings
- **Head:** `nn.Dropout(0.1)` → `nn.Linear(768, num_classes)`
- **Implemented in:** `src/model.py:CategoryClassifier`

## Class counts

| Stage | Classes |
|---|---|
| super | ~20 |
| parent | ~190 |
| leaf | 3000+ (varies by taxonomy version) |

Each stage has its own independent `best_model.pt` checkpoint (weights not included in this repository). Encoder weights are shared in structure but not in values (trained independently, except leaf warm-starts from parent's encoder via `--init-encoder-from`).

## Inference runtime

Production runs INT8 ONNX — one `onnxruntime.InferenceSession` per stage. Stages are called sequentially; each stage's argmax result constrains the next stage's logit mask.

## Training-time loss

Flat softmax cross-entropy over all classes in the stage (class-weighted via `--class-weights` flag). The cascaded masking used at inference is **not** applied during training.

## See also

- `docs/features/cascaded-prediction.md` — masked softmax detail
- `docs/features/class-weights.md` — rare-class weighting
- `docs/vendors/xlm-roberta.md` — encoder choice rationale
- `docs/vendors/onnxruntime.md` — INT8 quantization
