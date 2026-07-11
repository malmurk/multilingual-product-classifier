# Feature: cascaded masked-softmax prediction

The classifier uses a three-stage cascade at inference time. Each stage constrains the next by masking out logits that are structurally impossible given the previous prediction.

## Inference sequence

1. **Super stage**
   - Run encoder on the input text.
   - Softmax over all ~20 super classes.
   - `argmax` → `super_name`.

2. **Parent stage**
   - Reuse the same encoder output.
   - Build `valid_parents`: indices of all parents that are children of `super_name` in the taxonomy.
   - Set all other parent logits to `NEG_INF = -1e10`.
   - Softmax → `argmax` → `parent_name`.

3. **Leaf stage**
   - Build `valid_leaves`: indices of all leaves that are children of `parent_name`.
   - Set all other leaf logits to `NEG_INF = -1e10`.
   - Softmax → `argmax` → `leaf_name`.

Code: `src/predict.py:predict_one()`, lines 88–133.

## Why this works

The leaf head may have 3,000+ output dimensions. At any single query only 10–20 leaf candidates survive the parent-constrained mask. This makes large taxonomies tractable at inference without any special training-time changes.

## Training-time behaviour

Training-time loss is **flat softmax** over all classes in the stage — the masking is inference-only. This is by design: the encoder must learn to separate all leaves, not just the constrained subset.

## Confidence interpretation

Post-mask confidences are NOT calibrated against the full class space:
- A "95% parent confidence" means 95% among ~10 parent candidates, not among ~190.
- Do not report these to users as raw calibration numbers.

## See also

- `docs/architecture/pipeline-overview.md` — model structure
- `docs/features/auto-assign-threshold.md` — production confidence threshold
