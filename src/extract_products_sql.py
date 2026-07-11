"""Extract real product rows from the `products.sql` MySQL dump.

Output: data/real/products_raw.jsonl — one line per row with the columns
we actually need for training + inference validation:

    {
      "id": 1234,
      "category_id": 42,
      "title": "Wireless Mouse X200",
      "title_ru": null,
      "title_ro": "mouse wireless x200",
      "brand": "ExampleBrand",
      "price": 199,
      "price_discount": null,
      "description": ".",
      "keywords": null,
      "red_line": {"ro": null, "ru": null},
      "active": 0,
      "quantity": 12,
      "created_at": "2026-01-01 12:00:00"
    }

The SQL dump has a single `products` table and NO categories table — so
the category_id here is still a raw integer. Use
`build_real_training_data.py` to join it with a categories dump and
produce the final training JSONL with the full super/parent/leaf path.

Usage:
    python -m src.extract_products_sql --sql products.sql \
        --out data/real/products_raw.jsonl
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Iterator, List, Optional

logger = logging.getLogger("extract_products_sql")


# --- Columns as declared in the INSERT header (generated columns are
# skipped by mysqldump, so they're NOT in the data rows).
# NOTE: trimmed and genericized for publication — the original deployment
# used the client's full column list. Set this to YOUR dump's INSERT column
# order before running; EXPECTED_COLS is derived from it.
COLUMNS: List[str] = [
    "id", "active", "sku", "category_id", "link", "title", "brand",
    "brand_id", "price", "price_discount", "price_purchase", "quantity",
    "total_quantity_", "description", "red_line", "keywords", "created_at",
    "updated_at", "title_", "description_", "mainImg", "product_code",
]
EXPECTED_COLS = len(COLUMNS)

# We emit only these to the JSONL (training doesn't care about SKU / stock etc.)
# mainImg kept for multimodal eval/training (CDN image URL per product).
KEEP: List[str] = [
    "id", "category_id", "mainImg", "title", "brand", "price", "price_discount",
    "price_purchase", "description", "keywords", "active", "quantity",
    "total_quantity_", "red_line", "title_", "description_", "link",
    "product_code", "created_at",
]


# ---------------------------------------------------------------- SQL tokeniser

class SQLRowParser:
    """Parse a MySQL VALUES row '(val1, val2, ...)' into a list of
    Python values. Handles:

      - NULL          -> None
      - integers      -> int
      - floats        -> float
      - 'strings'     -> str (with \\' \\" \\\\ \\n \\r \\t \\0 escapes)
      - X'68657861'   -> bytes decoded as UTF-8 (mysqldump uses hex for
                         utf8mb4_bin columns like title_, description_)
      - nested parens inside strings are kept as-is
    """

    def __init__(self, text: str):
        self.text = text
        self.i = 0
        self.n = len(text)

    def _peek(self) -> str:
        return self.text[self.i] if self.i < self.n else ""

    def _skip_ws(self) -> None:
        while self.i < self.n and self.text[self.i] in " \t\n\r":
            self.i += 1

    def parse_row(self) -> Optional[List[Any]]:
        self._skip_ws()
        if self.i >= self.n:
            return None
        if self.text[self.i] != "(":
            return None
        self.i += 1  # consume '('
        values: List[Any] = []
        while True:
            self._skip_ws()
            if self.i >= self.n:
                raise ValueError("Unterminated row")
            if self.text[self.i] == ")":
                self.i += 1
                return values
            values.append(self._read_value())
            self._skip_ws()
            if self.i < self.n and self.text[self.i] == ",":
                self.i += 1
            elif self.i < self.n and self.text[self.i] == ")":
                self.i += 1
                return values
            else:
                raise ValueError(f"Expected ',' or ')' at pos {self.i}")

    def _read_value(self) -> Any:
        c = self._peek()
        if c == "'":
            return self._read_string()
        if c == "X" and self.i + 1 < self.n and self.text[self.i + 1] == "'":
            return self._read_hex()
        # Bareword: NULL / number / true / false
        start = self.i
        while self.i < self.n and self.text[self.i] not in ",)":
            self.i += 1
        raw = self.text[start:self.i].strip()
        if raw.upper() == "NULL" or raw == "":
            return None
        if raw.upper() == "TRUE":
            return True
        if raw.upper() == "FALSE":
            return False
        # number
        try:
            if "." in raw or "e" in raw.lower():
                return float(raw)
            return int(raw)
        except ValueError:
            return raw  # fallback

    def _read_string(self) -> str:
        assert self.text[self.i] == "'"
        self.i += 1  # opening quote
        out: List[str] = []
        while self.i < self.n:
            c = self.text[self.i]
            if c == "\\" and self.i + 1 < self.n:
                nxt = self.text[self.i + 1]
                mapping = {
                    "'": "'", '"': '"', "\\": "\\",
                    "n": "\n", "r": "\r", "t": "\t",
                    "0": "\x00", "b": "\b", "Z": "\x1a",
                }
                out.append(mapping.get(nxt, nxt))
                self.i += 2
                continue
            if c == "'":
                # could be a closing quote, OR a doubled '' inside string
                if self.i + 1 < self.n and self.text[self.i + 1] == "'":
                    out.append("'")
                    self.i += 2
                    continue
                self.i += 1
                return "".join(out)
            out.append(c)
            self.i += 1
        raise ValueError("Unterminated string")

    def _read_hex(self) -> bytes:
        # X'ABCD...'
        assert self.text[self.i:self.i + 2] == "X'"
        self.i += 2
        start = self.i
        while self.i < self.n and self.text[self.i] != "'":
            self.i += 1
        hex_str = self.text[start:self.i]
        self.i += 1  # closing quote
        try:
            return bytes.fromhex(hex_str)
        except ValueError:
            return b""


def _decode_hex_json(b: Optional[bytes]) -> Any:
    """title_, description_, link_ columns store JSON like
    {"ro":"...","ru":"..."} as utf8mb4_bin, dumped by mysqldump as X'...'.
    Decode to text, then parse as JSON; fall back to the raw string.
    """
    if b is None:
        return None
    try:
        text = b.decode("utf-8")
    except UnicodeDecodeError:
        try:
            text = b.decode("utf-8", errors="replace")
        except Exception:
            return None
    text = text.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


# ---------------------------------------------------------------- SQL streaming

def iter_rows(sql_path: Path) -> Iterator[dict]:
    """Stream through the SQL file, yielding one dict per row. Memory-safe
    for 150MB dumps: we accumulate one INSERT block at a time.
    """
    in_insert = False
    buf: List[str] = []

    with open(sql_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("INSERT INTO `products`"):
                in_insert = True
                # Header lines usually end with "VALUES" on the next line.
                continue
            if not in_insert:
                continue
            stripped = line.strip()
            if stripped == "VALUES":
                continue

            # Find where the VALUES list ends. mysqldump ends the block with
            # a line ending in ';' (after the last row).
            buf.append(line)
            if stripped.endswith(";"):
                yield from _yield_rows_from_block("".join(buf))
                buf = []
                in_insert = False

        # Flush any trailing buffer (shouldn't happen with valid dumps)
        if buf:
            yield from _yield_rows_from_block("".join(buf))


def _yield_rows_from_block(block: str) -> Iterator[dict]:
    # Trim trailing ';' if present
    block = block.strip()
    if block.endswith(";"):
        block = block[:-1]
    parser = SQLRowParser(block)
    while True:
        try:
            row = parser.parse_row()
        except ValueError as e:
            logger.warning("parse error: %s", e)
            return
        if row is None:
            return
        if len(row) != EXPECTED_COLS:
            logger.warning(
                "skipping row with %d cols (expected %d)", len(row), EXPECTED_COLS
            )
            # consume a ',' if present and continue
            parser._skip_ws()
            if parser.i < parser.n and parser.text[parser.i] == ",":
                parser.i += 1
            continue
        yield dict(zip(COLUMNS, row))
        parser._skip_ws()
        if parser.i < parser.n and parser.text[parser.i] == ",":
            parser.i += 1


# ---------------------------------------------------------------- main

def normalise(row: dict) -> dict:
    out = {k: row.get(k) for k in KEEP}
    out["title_"] = _decode_hex_json(row.get("title_"))
    out["description_"] = _decode_hex_json(row.get("description_"))
    # red_line is stored as a plain JSON string, not hex
    rl = row.get("red_line")
    if isinstance(rl, str) and rl.strip().startswith("{"):
        try:
            out["red_line"] = json.loads(rl)
        except json.JSONDecodeError:
            out["red_line"] = rl

    # Pull Russian/Romanian titles out of the JSON blob for convenience
    title_bundle = out.get("title_") if isinstance(out.get("title_"), dict) else {}
    out["title_ru"] = (title_bundle or {}).get("ru")
    out["title_ro"] = (title_bundle or {}).get("ro")
    desc_bundle = out.get("description_") if isinstance(out.get("description_"), dict) else {}
    out["description_ru"] = (desc_bundle or {}).get("ru")
    out["description_ro"] = (desc_bundle or {}).get("ro")
    # Drop the bulky JSON originals once we've unpacked them
    out.pop("title_", None)
    out.pop("description_", None)
    # Drop fields that are empty strings
    for k in list(out.keys()):
        if isinstance(out[k], str) and not out[k].strip():
            out[k] = None
    return out


def run(sql_path: Path, out_path: Path) -> dict:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_total = 0
    n_written = 0
    n_with_category = 0
    n_active = 0
    categories_seen = set()

    with open(out_path, "w", encoding="utf-8") as fh:
        for row in iter_rows(sql_path):
            n_total += 1
            rec = normalise(row)
            if rec.get("category_id") is not None:
                n_with_category += 1
                categories_seen.add(int(rec["category_id"]))
            if rec.get("active"):
                n_active += 1
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_written += 1

    stats = {
        "rows_read": n_total,
        "rows_written": n_written,
        "rows_with_category_id": n_with_category,
        "rows_active": n_active,
        "unique_category_ids": len(categories_seen),
    }
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sql", default="products.sql")
    ap.add_argument("--out", default="data/real/products_raw.jsonl")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    stats = run(Path(args.sql), Path(args.out))
    logger.info("Done: %s", json.dumps(stats, indent=2))
    # Print JSON to stdout too so callers can machine-read it
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
