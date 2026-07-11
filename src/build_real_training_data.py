"""Join extracted products with the shop's 3-table taxonomy and produce
training-ready JSONL with the full super -> parent -> leaf path.

Shop schema
-----------
    super_categories  (id, name, name_ JSON{ro,ru})
    parents           (id, super_category_id, name, name_ JSON{ro,ru})
    categories        (id, parent_id, name, name_ JSON{ro,ru})

    products.category_id  ->  categories.id  (leaves)

Two modes:

  --shop-taxonomy   use shop's own 4624-leaf tree as ground truth (recommended)
  default           match shop leaves against canonical taxonomy_full.csv

Trash buckets (parents 15 "Vouchers" and 262 "Trash2", leaf 4899 "Trash3")
are dropped by default; pass --keep-trash to include them.

Usage:
    python -m src.build_real_training_data --shop-taxonomy \
        --products   data/real/products_raw.jsonl \
        --categories data/real/categories.sql \
        --parents    data/real/parents.sql \
        --supers     data/real/super_categories.sql \
        --out        data/real/products_shop.jsonl \
        --shop-grid-out taxonomy_live.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.preprocess import build_input_text, price_band

logger = logging.getLogger("build_real_training_data")


# -------------------------------------------------------------- SQL helpers

def _decode_hex_json(value: Any) -> Any:
    if isinstance(value, bytes):
        try:
            text = value.decode("utf-8", errors="replace").strip()
        except Exception:
            return None
        if not text:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text
    if isinstance(value, str) and value.strip().startswith("{"):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _load_sql_rows(path: Path, table: str) -> List[dict]:
    from src.extract_products_sql import SQLRowParser

    content = path.read_text(encoding="utf-8", errors="replace")
    pattern = re.compile(
        rf"INSERT\s+INTO\s+`{re.escape(table)}`\s*\(([^)]+)\)\s*VALUES\s*(.*?);",
        re.IGNORECASE | re.DOTALL,
    )
    out: List[dict] = []
    for m in pattern.finditer(content):
        header = [c.strip().strip("`") for c in m.group(1).split(",")]
        body = m.group(2)
        parser = SQLRowParser(body)
        while True:
            try:
                vals = parser.parse_row()
            except ValueError as e:
                logger.warning("SQL parse error in %s: %s", path.name, e)
                break
            if vals is None:
                break
            if len(vals) == len(header):
                out.append(dict(zip(header, vals)))
            parser._skip_ws()
            if parser.i < parser.n and parser.text[parser.i] == ",":
                parser.i += 1
    return out


def _names_from_row(row: dict) -> Tuple[str, Optional[str], Optional[str]]:
    plain = (row.get("name") or "").strip()
    bundle = _decode_hex_json(row.get("name_"))
    name_ru = name_ro = None
    if isinstance(bundle, dict):
        ru = bundle.get("ru")
        ro = bundle.get("ro")
        name_ru = ru.strip() if isinstance(ru, str) and ru.strip() else None
        name_ro = ro.strip() if isinstance(ro, str) and ro.strip() else None
    return plain, name_ru, name_ro


# -------------------------------------------------------------- shop taxonomy

class ShopTaxonomy:
    def __init__(
        self,
        leaves: Dict[int, dict],
        parents: Dict[int, dict],
        supers: Dict[int, dict],
    ):
        self.leaves = leaves
        self.parents = parents
        self.supers = supers

    def path_for(self, leaf_id: int) -> Optional[Tuple[dict, dict, dict]]:
        leaf = self.leaves.get(leaf_id)
        if not leaf:
            return None
        parent = self.parents.get(leaf.get("parent_id"))
        if not parent:
            return None
        super_ = self.supers.get(parent.get("super_id"))
        if not super_:
            return None
        return super_, parent, leaf

    @classmethod
    def from_sql(cls, categories_sql: Path, parents_sql: Path, supers_sql: Path) -> "ShopTaxonomy":
        supers: Dict[int, dict] = {}
        for r in _load_sql_rows(supers_sql, "super_categories"):
            plain, ru, ro = _names_from_row(r)
            supers[int(r["id"])] = {"id": int(r["id"]), "name": plain, "name_ru": ru, "name_ro": ro}

        parents: Dict[int, dict] = {}
        for r in _load_sql_rows(parents_sql, "parents"):
            plain, ru, ro = _names_from_row(r)
            sup_id = r.get("super_category_id")
            sup_id = int(sup_id) if sup_id not in (None, "", 0) else None
            parents[int(r["id"])] = {
                "id": int(r["id"]),
                "name": plain,
                "name_ru": ru,
                "name_ro": ro,
                "super_id": sup_id,
            }

        leaves: Dict[int, dict] = {}
        for r in _load_sql_rows(categories_sql, "categories"):
            plain, ru, ro = _names_from_row(r)
            par_id = r.get("parent_id")
            par_id = int(par_id) if par_id not in (None, "", 0) else None
            leaves[int(r["id"])] = {
                "id": int(r["id"]),
                "name": plain,
                "name_ru": ru,
                "name_ro": ro,
                "parent_id": par_id,
            }

        logger.info(
            "Loaded taxonomy: %d supers, %d parents, %d leaves",
            len(supers), len(parents), len(leaves),
        )
        return cls(leaves=leaves, parents=parents, supers=supers)


# -------------------------------------------------------------- canonical CSV

def _norm(s: Optional[str]) -> str:
    if not s:
        return ""
    s = s.lower()
    s = re.sub(r"[^\w\s]", " ", s, flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def load_taxonomy_csv(path: Path):
    triples_norm: Dict[Tuple[str, str, str], Tuple[str, str, str]] = {}
    leaf_to_paths: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    norm_leaf_to_paths: Dict[str, List[Tuple[str, str, str]]] = defaultdict(list)
    super_norm: Dict[str, str] = {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) < 3:
                continue
            s, p, l = row[0].strip(), row[1].strip(), row[2].strip()
            if not (s and p and l):
                continue
            triples_norm[(_norm(s), _norm(p), _norm(l))] = (s, p, l)
            if (s, p) not in leaf_to_paths[l]:
                leaf_to_paths[l].append((s, p))
            norm_leaf_to_paths[_norm(l)].append((s, p, l))
            super_norm[_norm(s)] = s
    return triples_norm, leaf_to_paths, norm_leaf_to_paths, super_norm


# -------------------------------------------------------------- matching

def match_to_taxonomy(
    shop_path: Tuple[dict, dict, dict],
    triples_norm: Dict[Tuple[str, str, str], Tuple[str, str, str]],
    norm_leaf_to_paths: Dict[str, List[Tuple[str, str, str]]],
    super_norm: Dict[str, str],
    fuzzy_threshold: float = 0.88,
) -> Tuple[Optional[Tuple[str, str, str]], str]:
    sup, par, leaf = shop_path
    sup_names = [sup.get("name_ru"), sup.get("name_ro"), sup.get("name")]
    par_names = [par.get("name_ru"), par.get("name_ro"), par.get("name")]
    leaf_names = [leaf.get("name_ru"), leaf.get("name_ro"), leaf.get("name")]

    # 1. exact triple match
    for s_name in sup_names:
        if not s_name:
            continue
        ns = _norm(s_name)
        for p_name in par_names:
            if not p_name:
                continue
            np_ = _norm(p_name)
            for l_name in leaf_names:
                if not l_name:
                    continue
                key = (ns, np_, _norm(l_name))
                if key in triples_norm:
                    return triples_norm[key], "exact_triple"

    # 2. leaf name match constrained/preferred to shop super
    shop_super_norm = set(filter(None, (_norm(x) for x in sup_names)))
    for l_name in leaf_names:
        if not l_name:
            continue
        paths = norm_leaf_to_paths.get(_norm(l_name), [])
        if not paths:
            continue
        candidates = [(s, p, l) for (s, p, l) in paths if _norm(s) in shop_super_norm]
        if len(candidates) >= 1:
            return candidates[0], "leaf_exact_in_super"
        if len(paths) == 1:
            return paths[0], "leaf_exact_unique"

    # 3. fuzzy leaf match (skip entirely when disabled)
    if fuzzy_threshold >= 1.0:
        return None, "unmatched"

    best: Optional[Tuple[str, str, str]] = None
    best_score = 0.0
    leaf_norms = [_norm(x) for x in leaf_names if x]
    if not leaf_norms:
        return None, "unmatched"
    for norm_leaf, paths in norm_leaf_to_paths.items():
        nl_len = len(norm_leaf)
        if nl_len == 0:
            continue
        for ln in leaf_norms:
            ln_len = len(ln)
            if ln_len == 0:
                continue
            min_len, max_len = (nl_len, ln_len) if nl_len < ln_len else (ln_len, nl_len)
            if 2 * min_len / (min_len + max_len) < fuzzy_threshold:
                continue
            sm = SequenceMatcher(None, ln, norm_leaf)
            if sm.quick_ratio() < fuzzy_threshold:
                continue
            score = sm.ratio()
            if score <= best_score:
                continue
            in_super = [(s, p, l) for (s, p, l) in paths if _norm(s) in shop_super_norm]
            pool = in_super or paths
            if not pool:
                continue
            best_score = score
            best = pool[0]
    if best and best_score >= fuzzy_threshold:
        return best, "fuzzy"
    return None, "unmatched"


def _resolve_shop_leaves(shop, triples_norm, norm_leaf_to_paths, super_norm, fuzzy_threshold):
    resolved: Dict[int, Tuple[Tuple[str, str, str], str]] = {}
    stats = Counter()
    for leaf_id in shop.leaves:
        path = shop.path_for(leaf_id)
        if path is None:
            continue
        triple, conf = match_to_taxonomy(
            path, triples_norm, norm_leaf_to_paths, super_norm,
            fuzzy_threshold=fuzzy_threshold,
        )
        stats[conf] += 1
        if triple is not None:
            resolved[leaf_id] = (triple, conf)
    return resolved, stats


# -------------------------------------------------------------- trash filter

# parent_id 15  = "Vouchers" (gift vouchers, not real products)
# parent_id 262 = "Trash2"        (orphaned categories from an old migration)
# leaf_id 4899  = "Trash3"        (catch-all dumping ground under parent 262)
TRASH_PARENT_IDS = {15, 262}
TRASH_LEAF_IDS = {4899}


def _is_trash(shop_path: Tuple[dict, dict, dict]) -> bool:
    sup, par, leaf = shop_path
    if par.get("id") in TRASH_PARENT_IDS:
        return True
    if leaf.get("id") in TRASH_LEAF_IDS:
        return True
    if sup.get("id") is None:
        return True
    return False


def _shop_triple_labels(shop_path: Tuple[dict, dict, dict]) -> Tuple[str, str, str]:
    """Shop's Russian names as training labels (fall back to RO / plain)."""
    sup, par, leaf = shop_path
    def pick(d: dict) -> str:
        return (d.get("name_ru") or d.get("name") or d.get("name_ro") or "").strip()
    return pick(sup), pick(par), pick(leaf)


# -------------------------------------------------------------- text builder

def build_text_for_training(product: dict) -> str:
    title = (
        product.get("title_ru")
        or product.get("title_ro")
        or product.get("title")
        or ""
    )
    brand = product.get("brand")
    keywords = product.get("keywords")
    description = (
        product.get("description_ru")
        or product.get("description_ro")
        or product.get("description")
    )
    pb = price_band(product.get("price_discount") or product.get("price"))
    extras: List[Any] = []
    if keywords:
        extras.append(keywords)
    if pb:
        extras.append(pb)
    return build_input_text(
        title=title,
        brand=brand,
        description=description,
        extra_fields=extras,
    )


# -------------------------------------------------------------- grid export

def export_shop_grid(shop: ShopTaxonomy, out_csv: Path) -> int:
    """Dump the shop's live taxonomy as a 3-column grid CSV (same format as
    taxonomy_full.csv) — useful for retraining against the production
    label space instead of the stale canonical CSV.
    """
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with open(out_csv, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["super", "parent", "leaf"])
        for leaf_id, leaf in shop.leaves.items():
            path = shop.path_for(leaf_id)
            if path is None:
                continue
            if leaf_id in TRASH_LEAF_IDS or path[1].get("id") in TRASH_PARENT_IDS:
                continue
            s_lbl, p_lbl, l_lbl = _shop_triple_labels(path)
            if not (s_lbl and p_lbl and l_lbl):
                continue
            w.writerow([s_lbl, p_lbl, l_lbl])
            rows += 1
    return rows


# -------------------------------------------------------------- main

def run(
    products_path: Path,
    categories_sql: Path,
    parents_sql: Path,
    supers_sql: Path,
    taxonomy_path: Optional[Path],
    out_path: Path,
    unmatched_path: Optional[Path] = None,
    drop_inactive: bool = False,
    drop_trash: bool = True,
    fuzzy_threshold: float = 0.88,
    shop_taxonomy: bool = False,
    shop_grid_out: Optional[Path] = None,
) -> dict:
    logger.info("Loading shop SQL dumps")
    shop = ShopTaxonomy.from_sql(categories_sql, parents_sql, supers_sql)

    if shop_taxonomy:
        logger.info("Using shop tables as ground truth (no canonical CSV match)")
        resolved: Dict[int, Tuple[Tuple[str, str, str], str]] = {}
        leaf_stats = Counter()
    else:
        if taxonomy_path is None:
            raise ValueError("taxonomy_path is required unless --shop-taxonomy")
        logger.info("Loading canonical taxonomy: %s", taxonomy_path)
        triples_norm, _, norm_leaf_to_paths, super_norm = load_taxonomy_csv(taxonomy_path)
        logger.info(
            "  %d canonical leaves, %d supers",
            len({k[2] for k in triples_norm}), len(super_norm),
        )
        logger.info("Resolving %d shop leaves (one-time)...", len(shop.leaves))
        resolved, leaf_stats = _resolve_shop_leaves(
            shop, triples_norm, norm_leaf_to_paths, super_norm, fuzzy_threshold
        )
        logger.info("  leaf-level matches: %s", dict(leaf_stats))
        logger.info("  %d / %d shop leaves mapped", len(resolved), len(shop.leaves))

    if shop_grid_out is not None:
        n = export_shop_grid(shop, shop_grid_out)
        logger.info("Wrote shop grid: %s (%d rows)", shop_grid_out, n)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    conf_counts = Counter()
    super_counts = Counter()
    leaf_counts = Counter()
    unmatched_shop_leaves = Counter()
    broken_chain_ids = Counter()
    n_in = n_emitted = 0
    n_dropped_inactive = n_dropped_trash = n_dropped_no_cat = 0

    uf = open(unmatched_path, "w", encoding="utf-8") if unmatched_path else None
    try:
        with open(products_path, encoding="utf-8") as fin, \
             open(out_path, "w", encoding="utf-8") as fout:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                n_in += 1
                p = json.loads(line)
                if drop_inactive and not p.get("active"):
                    n_dropped_inactive += 1
                    continue
                cat_id = p.get("category_id")
                if cat_id is None:
                    n_dropped_no_cat += 1
                    continue
                cat_id = int(cat_id)
                path = shop.path_for(cat_id)
                if path is None:
                    broken_chain_ids[cat_id] += 1
                    continue
                if drop_trash and _is_trash(path):
                    n_dropped_trash += 1
                    continue

                if shop_taxonomy:
                    triple = _shop_triple_labels(path)
                    conf = "shop"
                    if not all(triple):
                        n_dropped_no_cat += 1
                        continue
                else:
                    resolution = resolved.get(cat_id)
                    if resolution is None:
                        leaf = path[2]
                        key = leaf.get("name_ru") or leaf.get("name") or f"id={leaf['id']}"
                        unmatched_shop_leaves[key] += 1
                        conf_counts["unmatched"] += 1
                        if uf is not None:
                            uf.write(json.dumps({
                                "product_id": p.get("id"),
                                "category_id": cat_id,
                                "shop_super_ru": path[0].get("name_ru"),
                                "shop_parent_ru": path[1].get("name_ru"),
                                "shop_leaf_ru": path[2].get("name_ru"),
                                "shop_leaf_ro": path[2].get("name_ro"),
                            }, ensure_ascii=False) + "\n")
                        continue
                    triple, conf = resolution

                conf_counts[conf] += 1
                super_, parent, leaf = triple
                super_counts[super_] += 1
                leaf_counts[leaf] += 1
                record = {
                    "text": build_text_for_training(p),
                    "super": super_,
                    "parent": parent,
                    "leaf": leaf,
                    "match": conf,
                    "product_id": p.get("id"),
                    "shop_category_id": cat_id,
                    "shop_leaf_ru": path[2].get("name_ru"),
                    "price": p.get("price"),
                    "price_discount": p.get("price_discount"),
                    "brand": p.get("brand"),
                    "active": p.get("active"),
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                n_emitted += 1
    finally:
        if uf is not None:
            uf.close()

    return {
        "mode": "shop_taxonomy" if shop_taxonomy else "canonical_csv",
        "input_rows": n_in,
        "emitted": n_emitted,
        "dropped_inactive": n_dropped_inactive,
        "dropped_trash": n_dropped_trash,
        "dropped_no_category": n_dropped_no_cat,
        "by_match_type": dict(conf_counts),
        "supers_covered": len(super_counts),
        "leaves_covered": len(leaf_counts),
        "broken_chain_rows": sum(broken_chain_ids.values()),
        "unique_broken_chain_ids": len(broken_chain_ids),
        "top_unmatched_shop_leaves": unmatched_shop_leaves.most_common(20),
        "top_broken_chain_category_ids": broken_chain_ids.most_common(10),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--products", default="data/real/products_raw.jsonl")
    ap.add_argument("--categories", default="data/real/categories.sql")
    ap.add_argument("--parents", default="data/real/parents.sql")
    ap.add_argument("--supers", default="data/real/super_categories.sql")
    ap.add_argument("--taxonomy", default="taxonomy_full.csv")
    ap.add_argument("--out", default="data/real/products_real.jsonl")
    ap.add_argument("--unmatched", default="data/real/unmatched.jsonl")
    ap.add_argument("--drop-inactive", action="store_true")
    ap.add_argument(
        "--keep-trash", action="store_true",
        help="Keep products under parent_id 15/262 or leaf 4899. "
             "Off by default — these are shop archive buckets, not real categories.",
    )
    ap.add_argument(
        "--shop-taxonomy", action="store_true",
        help="Use shop tables as ground truth (4624 leaves). "
             "Skips canonical-CSV matching entirely — no rows become 'unmatched'.",
    )
    ap.add_argument(
        "--shop-grid-out", default=None,
        help="Also dump the shop's live taxonomy as a 3-column grid CSV "
             "(format of taxonomy_full.csv) for classifier retraining.",
    )
    ap.add_argument("--fuzzy", type=float, default=0.88)
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    stats = run(
        Path(args.products), Path(args.categories), Path(args.parents),
        Path(args.supers),
        Path(args.taxonomy) if not args.shop_taxonomy else None,
        Path(args.out),
        unmatched_path=Path(args.unmatched) if args.unmatched else None,
        drop_inactive=args.drop_inactive,
        drop_trash=not args.keep_trash,
        fuzzy_threshold=args.fuzzy,
        shop_taxonomy=args.shop_taxonomy,
        shop_grid_out=Path(args.shop_grid_out) if args.shop_grid_out else None,
    )
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
