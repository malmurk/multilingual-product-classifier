# Architecture: input text format

The classifier consumes a single string per product. The format is defined by one authoritative function — never manually replicate it.

## Authoritative function

```python
# src/preprocess.py
def build_input_text(
    title: str,
    attributes: Optional[Dict[str, Any]] = None,
    description: Optional[str] = None,
    brand: Optional[str] = None,
    extra_fields: Optional[Iterable[Any]] = None,
    description_chars: int = 200,
) -> str:
```

**Output format:** `"title | brand | attr_value_1 | ... | extra_field_1 | ... | description[:200]"`

Empty fields are **silently dropped** — segment positions are NOT fixed. A product with no brand and no description produces `"title"`. A product with brand and price_band but no description produces `"title | brand | цена_средняя"`.

## Rules

1. Always call `build_input_text()` for any text you generate — training data or inference input.
2. Never manually join fields with `|`.
3. `clean_text()` is applied internally to each field: lowercases, strips non-word chars (preserves Cyrillic, Latin, digits, hyphen, underscore), collapses whitespace.
4. `description_chars=200` truncates the description **before** `clean_text` is applied.

## price_band tokens

`price_band(price)` returns one of: `цена_дешево` / `цена_низкая` / `цена_средняя` / `цена_высокая` / `цена_премиум` (or `None`). Thresholds are calibrated for the marketplace's primary currency. Prices ingested from a secondary marketplace channel are converted to the primary currency first, using a fixed conversion constant.

## Secondary-channel extension (Phase G)

Decision A2: rows from the secondary marketplace channel append `marketplace_secondary` as an extra_field so the model can condition on input shape (these rows typically have no description, giving 3 segments vs the usual 4–5).

## Train/serve parity

`src/predict.py` calls `build_input_text` via `_assemble_text()` (lines 23–42). Any regression here causes train/serve skew — the most impactful defect in the v2 regression (Phase G defect 1).
