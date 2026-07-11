"""Build a labeled, multimodal eval set from shop-categorized products.

Joins two files on product id:
  - products_shop.jsonl : ground-truth leaf labels (match=="shop" means the
                          label is the shop's own category_id->taxonomy path,
                          NOT a model prediction or fuzzy crosswalk).
  - products_raw.jsonl  : mainImg (CDN image URL) + RU/RO title/description.

Emits eval_set.jsonl: products that have BOTH a real shop leaf and an image,
stratified by capping rows-per-leaf so the long tail is represented rather
than drowned by a few popular categories. This is the held-out real-world
set every downstream measurement (text-only vs +image) runs against.

Usage:
    python -m src.build_eval_set \
        --shop data/eval/products_shop_img.jsonl \
        --raw  data/eval/products_raw_img.jsonl \
        --out  data/eval/eval_set.jsonl \
        --per-leaf 8 --max 2000 --seed 42
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def load_raw_index(path: Path) -> dict:
    idx = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                idx[r["id"]] = r
    return idx


def eligible_rows(shop_path: Path, raw_idx: dict):
    """Yield joined eval rows: shop-labeled products that have an image."""
    for line in open(shop_path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        s = json.loads(line)
        if s.get("match") != "shop":          # only clean ground truth
            continue
        raw = raw_idx.get(s.get("product_id"))
        if not raw or not raw.get("mainImg"):  # need an image for multimodal
            continue
        yield {
            "product_id": s.get("product_id"),
            "leaf": s["leaf"], "parent": s["parent"], "super": s["super"],
            "mainImg": raw["mainImg"],
            "title_ru": raw.get("title_ru"), "title_ro": raw.get("title_ro"),
            "description_ru": raw.get("description_ru"),
            "description_ro": raw.get("description_ro"),
            "brand": raw.get("brand"), "price": raw.get("price"),
        }


def stratified_sample(rows: list, per_leaf: int, max_total: int, seed: int) -> list:
    """Cap each leaf at `per_leaf` rows (tail coverage), then shuffle and trim
    to `max_total`. Leaves with fewer than the cap are kept whole."""
    by_leaf = defaultdict(list)
    for r in rows:
        by_leaf[r["leaf"]].append(r)
    rng = random.Random(seed)
    picked = []
    for items in by_leaf.values():
        rng.shuffle(items)
        picked.extend(items[:per_leaf])
    rng.shuffle(picked)
    return picked[:max_total] if max_total else picked


def _selftest() -> None:
    rows = [{"leaf": "a"} for _ in range(20)] + [{"leaf": "b"} for _ in range(3)]
    s = stratified_sample(rows, per_leaf=5, max_total=100, seed=1)
    assert sum(r["leaf"] == "a" for r in s) == 5, "leaf over cap not trimmed"
    assert sum(r["leaf"] == "b" for r in s) == 3, "under-cap leaf not kept whole"
    assert len(s) == 8, len(s)
    print("selftest ok")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shop", default="data/eval/products_shop_img.jsonl")
    ap.add_argument("--raw", default="data/eval/products_raw_img.jsonl")
    ap.add_argument("--out", default="data/eval/eval_set.jsonl")
    ap.add_argument("--per-leaf", type=int, default=8)
    ap.add_argument("--max", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        _selftest()
        return

    raw_idx = load_raw_index(Path(args.raw))
    rows = list(eligible_rows(Path(args.shop), raw_idx))
    sample = stratified_sample(rows, args.per_leaf, args.max, args.seed)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in sample:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    stats = {
        "eligible_pool": len(rows),
        "eligible_leaves": len(Counter(r["leaf"] for r in rows)),
        "sampled": len(sample),
        "sampled_leaves": len({r["leaf"] for r in sample}),
        "sample_with_ru_title": sum(1 for r in sample if (r.get("title_ru") or "").strip()),
        "sample_with_ro_title": sum(1 for r in sample if (r.get("title_ro") or "").strip()),
        "out": str(out),
    }
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
