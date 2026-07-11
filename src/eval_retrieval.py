"""Retrieval-based leaf classification eval on the REAL held-out set.

Splits products_shop_img.jsonl (ground-truth shop labels) into:
  - queries   = product_ids listed in eval_set.jsonl (held out)
  - reference = every other labeled product (the index)

Embeds both, classifies each query by its nearest-reference leaf (cosine,
1-NN aggregated to leaf), and reports top-1/top-5 leaf + parent + super
accuracy against the shop's own categorization.

This is the first real-world accuracy number for the text signal — measured
on real products with real labels, not the synthetic val set.

Stage A (this file, default): text via multilingual-e5.
Stage B plugs in image/multimodal embeddings by swapping --emb; the kNN
scoring is identical, so the delta is a clean modality comparison.

Usage:
    python -m src.eval_retrieval --emb text \
        --shop data/eval/products_shop_img.jsonl \
        --eval data/eval/eval_set.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------- data loading

def load_pool(shop_path: Path) -> dict:
    """pid -> {text, leaf, parent, super}. The 'text' field is the assembled
    training/inference string already present in products_shop_img.jsonl."""
    pool = {}
    for line in open(shop_path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        pid = r.get("product_id")
        if pid is None or not r.get("text"):
            continue
        pool[pid] = {
            "text": r["text"], "leaf": r["leaf"],
            "parent": r["parent"], "super": r["super"],
        }
    return pool


def load_query_ids(eval_path: Path) -> set:
    ids = set()
    for line in open(eval_path, encoding="utf-8"):
        line = line.strip()
        if line:
            ids.add(json.loads(line)["product_id"])
    return ids


# ---------------------------------------------------------------- embedding

def embed_e5(texts: list, batch: int = 128, max_length: int = 128) -> np.ndarray:
    """Mean-pooled, L2-normalized multilingual-e5 embeddings. e5 expects a
    prefix; we use 'query: ' on every string (symmetric similarity, so query
    and reference must share the same prefix)."""
    import torch
    from transformers import AutoModel, AutoTokenizer
    from src.model import get_device

    name = "intfloat/multilingual-e5-base"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModel.from_pretrained(name)
    device = get_device()
    model = model.to(device).eval()

    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            chunk = ["query: " + (t or "") for t in texts[i:i + batch]]
            enc = tok(chunk, max_length=max_length, padding=True,
                      truncation=True, return_tensors="pt").to(device)
            hidden = model(**enc).last_hidden_state           # (B, L, H)
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            pooled = torch.nn.functional.normalize(pooled, dim=-1)
            out.append(pooled.cpu().numpy())
            if (i // batch) % 20 == 0:
                print(f"  embedded {i + len(chunk)}/{len(texts)}", flush=True)
    return np.vstack(out).astype(np.float32)


# ---------------------------------------------------------------- classify

def classify(q_emb: np.ndarray, r_emb: np.ndarray, r_leaf: list,
             leaf_path: dict, q_meta: list, topk: int = 5,
             chunk: int = 256) -> dict:
    """1-NN aggregated to leaf. For each query, rank leaves by their single
    best-matching reference product; score top-1/top-5 leaf and top-1
    parent/super (via the predicted leaf's taxonomy path)."""
    r_leaf = np.asarray(r_leaf)
    n = q_emb.shape[0]
    top1 = top5 = par1 = sup1 = 0
    for s in range(0, n, chunk):
        sims = q_emb[s:s + chunk] @ r_emb.T          # (c, R)
        order = np.argsort(-sims, axis=1)            # ref indices, best first
        for j in range(sims.shape[0]):
            true = q_meta[s + j]
            seen, ranked = set(), []
            for ri in order[j]:
                lf = r_leaf[ri]
                if lf not in seen:
                    seen.add(lf)
                    ranked.append(lf)
                    if len(ranked) >= topk:
                        break
            pred = ranked[0]
            if pred == true["leaf"]:
                top1 += 1
            if true["leaf"] in ranked:
                top5 += 1
            pp = leaf_path.get(pred, (None, None))
            if pp[0] == true["parent"]:
                par1 += 1
            if pp[1] == true["super"]:
                sup1 += 1
    return {
        "n": n,
        "top1_leaf": round(top1 / n, 4),
        "top5_leaf": round(top5 / n, 4),
        "top1_parent": round(par1 / n, 4),
        "top1_super": round(sup1 / n, 4),
    }


def metrics_from_sims(sims: np.ndarray, r_leaf, leaf_path: dict, q_meta: list,
                      topk: int = 5) -> dict:
    """Same 1-NN-to-leaf scoring as classify(), but on a precomputed query x
    reference similarity matrix. Lets us score text, image, and a fused
    (weighted-sum) matrix identically — only the similarities differ."""
    r_leaf = np.asarray(r_leaf)
    n = sims.shape[0]
    top1 = top5 = par1 = sup1 = 0
    order = np.argsort(-sims, axis=1)
    for j in range(n):
        true = q_meta[j]
        seen, ranked = set(), []
        for ri in order[j]:
            lf = r_leaf[ri]
            if lf not in seen:
                seen.add(lf)
                ranked.append(lf)
                if len(ranked) >= topk:
                    break
        pred = ranked[0]
        top1 += pred == true["leaf"]
        top5 += true["leaf"] in ranked
        pp = leaf_path.get(pred, (None, None))
        par1 += pp[0] == true["parent"]
        sup1 += pp[1] == true["super"]
    return {
        "n": n,
        "top1_leaf": round(top1 / n, 4),
        "top5_leaf": round(top5 / n, 4),
        "top1_parent": round(par1 / n, 4),
        "top1_super": round(sup1 / n, 4),
    }


def _selftest() -> None:
    # Two leaves, clearly separable in 2-D; nearest neighbour must win.
    r_emb = np.array([[1, 0], [0, 1]], dtype=np.float32)
    r_leaf = ["a", "b"]
    leaf_path = {"a": ("pa", "sa"), "b": ("pb", "sb")}
    q_emb = np.array([[0.9, 0.1], [0.1, 0.9]], dtype=np.float32)
    q_emb /= np.linalg.norm(q_emb, axis=1, keepdims=True)
    q_meta = [{"leaf": "a", "parent": "pa", "super": "sa"},
              {"leaf": "b", "parent": "pb", "super": "sb"}]
    m = classify(q_emb, r_emb, r_leaf, leaf_path, q_meta, topk=2)
    assert m["top1_leaf"] == 1.0, m
    assert m["top1_parent"] == 1.0 and m["top1_super"] == 1.0, m
    print("selftest ok")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", choices=["text"], default="text")
    ap.add_argument("--shop", default="data/eval/products_shop_img.jsonl")
    ap.add_argument("--eval", default="data/eval/eval_set.jsonl")
    ap.add_argument("--cache", default="data/eval/emb_text.npz")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        _selftest()
        return

    pool = load_pool(Path(args.shop))
    qids = load_query_ids(Path(args.eval))
    qids &= set(pool)                      # only queries we have text for

    q_pids = [p for p in pool if p in qids]
    r_pids = [p for p in pool if p not in qids]
    leaf_path = {pool[p]["leaf"]: (pool[p]["parent"], pool[p]["super"]) for p in pool}

    print(f"queries={len(q_pids)}  reference={len(r_pids)}  "
          f"ref_leaves={len({pool[p]['leaf'] for p in r_pids})}", flush=True)

    cache = Path(args.cache)
    if cache.exists():
        print(f"loading cached embeddings {cache}", flush=True)
        z = np.load(cache, allow_pickle=True)
        q_emb, r_emb = z["q"], z["r"]
    else:
        print("embedding reference...", flush=True)
        r_emb = embed_e5([pool[p]["text"] for p in r_pids])
        print("embedding queries...", flush=True)
        q_emb = embed_e5([pool[p]["text"] for p in q_pids])
        np.savez(cache, q=q_emb, r=r_emb)
        print(f"cached -> {cache}", flush=True)

    q_meta = [pool[p] for p in q_pids]
    r_leaf = [pool[p]["leaf"] for p in r_pids]
    metrics = classify(q_emb, r_emb, r_leaf, leaf_path, q_meta)
    print("\n=== TEXT (e5) retrieval eval, real held-out set ===")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
