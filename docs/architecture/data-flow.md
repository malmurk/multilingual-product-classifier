# Architecture: data flow — marketplace DB to training set

## Source of truth

SQL exports from your database — product catalog, categories, parents, super-categories — under `data/real/*.sql`. `src/build_real_training_data.py` converts these into JSONL training rows. Raw exports and processed datasets are not included in this repository.

## Processed splits

| File | Size | Split |
|---|---|---|
| `data/processed/train.jsonl` | 26,089 rows (baseline) | 85% |
| `data/processed/val.jsonl` | ~3,000 rows | 10% |
| `data/processed/test.jsonl` | ~1,500 rows | 5% |

Data-boost v3 equivalents live in `<data-boost-project>`: `data/train_v3.jsonl`, `data/val_v3.jsonl`, `data/test_v3.jsonl`. None of these files ship with this repository.

## Taxonomy CSVs

| File | Leaves | When to use |
|---|---|---|
| `taxonomy_live.csv` | hundreds (deployed subset) | Current production |
| `taxonomy_universe.csv` | thousands (full universe) | Revival reference |
| `taxonomy_pruned.csv` | hundreds (pruned baseline) | Early-phase baseline |
| `data/taxonomy_v3.csv` | thousands (v3) | v3 retrain (`<data-boost-project>`) |

(Exact leaf counts are deployment-specific and removed for publication.)

**Always set `TAXONOMY_CSV` env var explicitly.** Never rely on `train.py` default (`taxonomy_full.csv`).

## JSONL row schema (key fields)

```json
{
  "text": "title | brand | price_band | description[:200]",
  "super": "Детские товары",
  "parent": "Игрушки",
  "leaf": "Конструкторы",
  "match": "shop",
  "product_id": 14579,
  "price": 89,
  "brand": "SLUBAN"
}
```

The `text` field is pre-assembled via `build_input_text`. Phase B / Phase G new rows must use the same function.

## See also

- `docs/architecture/text-format.md` — `build_input_text` signature
- `docs/runbooks/training.md` — how to feed these files into `train.py`
