"""Does smarter candidate aggregation raise the retrieval recall ceiling?

The reranker is capped by recall@K of candidate generation (~78% with 1-NN).
This compares leaf-scoring strategies on the cached e5 embeddings — no LLM,
no new downloads — to see if we can lift that ceiling for free:

  max        : score(leaf) = best single reference product   (1-NN, baseline)
  mean_all   : mean similarity over all of the leaf's products
  centroid   : cosine to the leaf's normalized mean embedding (prototype)
  meantop_m  : mean of the leaf's m best-matching products    (robust 1-NN)
  blend      : 0.5*max + 0.5*centroid

Usage:
    python -m src.eval_candidates           # m=5
    python -m src.eval_candidates --m 3
"""
from __future__ import annotations

import argparse
import numpy as np

from src.eval_rerank import reconstruct_split
from pathlib import Path


def recall_at_k(scores: np.ndarray, gt_code: np.ndarray, ks=(1, 5, 10, 20)) -> dict:
    """scores: (Q, n_leaves). gt_code: (Q,), -1 if the true leaf has no
    reference (an automatic miss). rank = #leaves scoring above the truth."""
    q = scores.shape[0]
    present = gt_code >= 0
    gt_scores = np.full(q, np.inf)
    gt_scores[present] = scores[np.arange(q)[present], gt_code[present]]
    rank = (scores > gt_scores[:, None]).sum(1)
    rank[~present] = 10**9                        # missing leaf -> never recalled
    return {f"r@{k}": round(float((rank < k).mean()), 4) for k in ks}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shop", default="data/eval/products_shop_img.jsonl")
    ap.add_argument("--eval", default="data/eval/eval_set.jsonl")
    ap.add_argument("--cache", default="data/eval/emb_text.npz")
    ap.add_argument("--m", type=int, default=5)
    args = ap.parse_args()

    pool, q_pids, r_pids = reconstruct_split(Path(args.shop), Path(args.eval))
    z = np.load(args.cache)
    q_emb, r_emb = z["q"], z["r"]
    assert q_emb.shape[0] == len(q_pids) and r_emb.shape[0] == len(r_pids), \
        "cache/split mismatch — re-run src.eval_retrieval"

    leaves = sorted({pool[p]["leaf"] for p in r_pids})
    code = {lf: i for i, lf in enumerate(leaves)}
    r_code = np.array([code[pool[p]["leaf"]] for p in r_pids])
    gt_code = np.array([code.get(pool[p]["leaf"], -1) for p in q_pids])
    cols_by_leaf = [np.where(r_code == c)[0] for c in range(len(leaves))]

    sims = (q_emb @ r_emb.T).astype(np.float32)            # (Q, R)
    Q, L = sims.shape[0], len(leaves)
    print(f"queries={Q} reference={sims.shape[1]} ref_leaves={L} m={args.m}\n")

    s_max = np.full((Q, L), -1e9, np.float32)
    s_mean = np.full((Q, L), -1e9, np.float32)
    s_mtop = np.full((Q, L), -1e9, np.float32)
    cent = np.zeros((L, r_emb.shape[1]), np.float32)
    for c, cols in enumerate(cols_by_leaf):
        sub = sims[:, cols]
        s_max[:, c] = sub.max(1)
        s_mean[:, c] = sub.mean(1)
        if sub.shape[1] <= args.m:
            s_mtop[:, c] = sub.mean(1)
        else:
            s_mtop[:, c] = np.partition(sub, -args.m, axis=1)[:, -args.m:].mean(1)
        cent[c] = r_emb[cols].mean(0)
    cent /= np.clip(np.linalg.norm(cent, axis=1, keepdims=True), 1e-9, None)
    s_cent = q_emb @ cent.T
    s_blend = 0.5 * s_max + 0.5 * s_cent

    strategies = {
        "max (1-NN, baseline)": s_max,
        "mean_all": s_mean,
        f"meantop_{args.m}": s_mtop,
        "centroid": s_cent,
        "blend(max+centroid)": s_blend,
    }
    print(f"{'strategy':<22} {'r@1':>7} {'r@5':>7} {'r@10':>7} {'r@20':>7}")
    for name, sc in strategies.items():
        r = recall_at_k(sc, gt_code)
        print(f"{name:<22} {r['r@1']:>7} {r['r@5']:>7} {r['r@10']:>7} {r['r@20']:>7}")


if __name__ == "__main__":
    main()
