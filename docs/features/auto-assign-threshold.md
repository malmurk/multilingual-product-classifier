# Feature: auto-assign confidence threshold

The production pipeline uses a confidence threshold to decide whether a prediction is sent straight to the catalog (auto-assign) or queued for human review (manual sorting).

## Rule

```
leaf_conf >= 0.93  →  auto-assign (no human review required)
leaf_conf  < 0.93  →  manual sorting queue
```

`leaf_conf` is the post-mask softmax probability of the predicted leaf — i.e. the argmax probability after logits for non-children of the predicted parent have been set to `NEG_INF`.

## Implementation

Defined as `THRESH = 0.93` in `<data-boost-project>/build_rich_results.py` (line 16). The script merges prediction JSONL with the input XLSX and writes an audit workbook with an `auto_assign_at_0.93` column (boolean) and an `outcome` column (`"auto-assign"` or `"manual_sorting"`).

A summary sheet reports the auto/manual split per product section.

## Calibration note

This threshold was tuned for an earlier model snapshot (pruned leaf set, `val_pruned`). It should be re-evaluated after every retrain because:
1. A larger leaf head (more classes) tends to lower per-class peak confidence.
2. Adding training data from a new external marketplace source changes the confidence distribution for shop-shape inputs.

After v3 ships, run a threshold sweep on `val_v3` or `val_pruned` before declaring 0.93 still valid.

## See also

- `docs/runbooks/inference.md` — how leaf_conf is computed
- `docs/features/cascaded-prediction.md` — confidence interpretation warning (post-mask, not calibrated)
