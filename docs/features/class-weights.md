# Feature: class-weighting for rare leaves

The training loss is weighted per class so rare leaves receive proportionally stronger gradient signal.

## Algorithm

```python
weight[c] = 1.0 / sqrt(count[c])   # for classes with count > 0
weight[c] = 1.0                      # for zero-count classes
# Then mean-normalize so mean(weights) = 1.0
weight = weight / weight.mean()
```

An optional max-clip (`max = 10 × median`) is applied before normalisation in later pipeline iterations to prevent extreme outliers from destabilising training.

## Storage format

A 1-D `torch.Tensor` of shape `[num_classes]`, indexed by the class index from `load_taxonomy()`. Saved as a `.pt` file, one per stage (weights not included in this repository):

```
data/class_weights_super_v3.pt
data/class_weights_parent_v3.pt
data/class_weights_leaf_v3.pt
```

## Training integration

Loaded via `--class-weights <path>` in `src/train.py`. The tensor is passed directly to `nn.CrossEntropyLoss(weight=class_weight, ...)` (line 124). The flag is named `--class-weights` in argparse and stored as `cfg.class_weights_path` in `TrainingConfig`.

Sanity check expected: `mean ≈ 1.0`, `min > 0`, `max ≤ 10`. Shape must match the taxonomy dimension for the stage.

## Why sqrt

- `1/count` over-weights extreme tail classes and causes instability.
- `1/sqrt(count)` is a well-known compromise that boosts rare classes without letting a single 2-row leaf dominate the gradient.

## See also

- `docs/runbooks/training.md` — how to pass class weights during training
- `docs/architecture/pipeline-overview.md` — where the loss is computed
