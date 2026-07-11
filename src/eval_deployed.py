"""Measure the DEPLOYED ONNX classifier (models/onnx) on the real eval set.

Uses the production HierarchicalPredictor with the authoritative label mapping
(hierarchy.json + *_labels.json), NOT a CSV-derived taxonomy. Thresholds are
set to 0 so every product reaches the leaf stage -> we measure raw top-1
capability, comparable to the e5-retrieval baseline (63.6%).

Reports coverage: the deployed head has a fixed leaf set; a shop leaf outside
it can NEVER be predicted (a structural ceiling separate from model quality).

Usage:
    python -m src.eval_deployed                # all 2500
    python -m src.eval_deployed --limit 500
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from classifier.predictor import HierarchicalPredictor


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default="data/eval/eval_set.jsonl")
    ap.add_argument("--model-dir", default="models/onnx")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    pred = HierarchicalPredictor(
        args.model_dir, thresholds={"super": 0.0, "parent": 0.0, "leaf": 0.0})
    # leaf_labels is {int_idx: name}; we test membership by NAME, so use values()
    leaf_set = set(pred.hierarchy.leaf_labels.values())

    rows = [json.loads(l) for l in open(args.eval, encoding="utf-8")]
    if args.limit:
        rows = rows[:args.limit]

    n = leaf_ok = cov = cov_ok = par_ok = sup_ok = 0
    for r in rows:
        title = r.get("title_ru") or r.get("title_ro") or ""
        if not title:
            continue
        desc = r.get("description_ru") or r.get("description_ro")
        p = pred.predict(title=title, brand=r.get("brand"), description=desc)
        n += 1
        covered = r["leaf"] in leaf_set
        cov += covered
        leaf_ok += p.leaf_label == r["leaf"]
        cov_ok += covered and p.leaf_label == r["leaf"]
        par_ok += p.parent_label == r["parent"]
        sup_ok += p.super_label == r["super"]
        if n % 200 == 0:
            print(f"  {n}/{len(rows)} ...", flush=True)

    print("\n=== DEPLOYED ONNX classifier on real eval set ===")
    print(json.dumps({
        "n": n,
        "classifier_leaf_set_size": len(leaf_set),
        "leaf_coverage": round(cov / n, 4),            # GT leaf exists in head
        "leaf_top1_overall": round(leaf_ok / n, 4),
        "leaf_top1_among_covered": round(cov_ok / cov, 4) if cov else None,
        "parent_top1": round(par_ok / n, 4),
        "super_top1": round(sup_ok / n, 4),
    }, indent=2))


if __name__ == "__main__":
    main()
