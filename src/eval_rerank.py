"""Stage C: retrieval -> LLM rerank, measured on the real held-out set.

Pipeline:
  1. e5 text retrieval gives a top-K leaf shortlist per product (reuses the
     cached embeddings from src.eval_retrieval -> data/eval/emb_text.npz).
  2. An LLM is shown the product + the K candidate leaves and picks one.
  3. Score the LLM's pick against the shop's ground-truth leaf.

A reranker can only be right when the true leaf is in the shortlist, so
recall@K is the ceiling. `--ceiling` computes that with NO LLM (free) and
also tells you the K to use. `--rerank` runs the actual LLM stage.

The LLM rerank uses OpenAI (gpt-4o-mini by default; override with --model),
needs OPENAI_API_KEY. Local/self-hosted providers (LM Studio, Ollama) were
removed in favor of a single hosted-API path.

Usage:
    python -m src.eval_rerank --ceiling
    python -m src.eval_rerank --rerank --model gpt-4o-mini --limit 400
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np

from src.eval_retrieval import load_pool, load_query_ids


# ---------------------------------------------------------------- candidates

def reconstruct_split(shop_path: Path, eval_path: Path):
    """Rebuild the exact query/reference split src.eval_retrieval cached, so
    the cached embedding rows line up by index."""
    pool = load_pool(shop_path)
    qids = load_query_ids(eval_path) & set(pool)
    q_pids = [p for p in pool if p in qids]
    r_pids = [p for p in pool if p not in qids]
    return pool, q_pids, r_pids


def topk_leaves(sims_row: np.ndarray, r_leaf: np.ndarray, k: int) -> list:
    """Ordered distinct leaves by best-matching reference product (1-NN agg)."""
    order = np.argsort(-sims_row)
    seen, ranked = set(), []
    for ri in order:
        lf = r_leaf[ri]
        if lf not in seen:
            seen.add(lf)
            ranked.append(lf)
            if len(ranked) >= k:
                break
    return ranked


# ---------------------------------------------------------------- LLM

OPENAI_MODEL = "gpt-4o-mini"


def make_client():
    from openai import OpenAI
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise SystemExit("OPENAI_API_KEY not set — needed for the LLM rerank.")
    return OpenAI(api_key=key)


def rerank_prompt(title, brand, price, cands, leaf_path):
    lines = []
    for i, lf in enumerate(cands, 1):
        par, sup = leaf_path.get(lf, (None, None))
        lines.append(f"{i}. {lf}  (section: {sup} > {par})")
    return (
        f"Product: {title}\nBrand: {brand or '-'}\nPrice: {price or '-'}\n\n"
        f"Candidate categories:\n" + "\n".join(lines) +
        f"\n\nPick the ONE category that best fits this product. "
        f'Reply ONLY with JSON: {{"choice": N}} where N is 1-{len(cands)}.'
    )


def parse_choice(text: str, k: int):
    m = re.search(r'"choice"\s*:\s*(\d+)', text) or re.search(r'\b(\d+)\b', text)
    if not m:
        return None
    n = int(m.group(1))
    return n if 1 <= n <= k else None


def llm_pick(client, model, title, brand, price, cands, leaf_path):
    msg = [
        {"role": "system", "content": "You are an e-commerce product "
         "categorization expert. Choose the single best category."},
        {"role": "user", "content": rerank_prompt(title, brand, price, cands, leaf_path)},
    ]
    r = client.chat.completions.create(model=model, messages=msg,
                                       temperature=0, max_tokens=30)
    choice = parse_choice(r.choices[0].message.content or "", len(cands))
    return cands[choice - 1] if choice else cands[0]   # fall back to retrieval top-1


# ---------------------------------------------------------------- runs

def load_display(eval_path: Path) -> dict:
    d = {}
    for line in open(eval_path, encoding="utf-8"):
        line = line.strip()
        if line:
            r = json.loads(line)
            d[r["product_id"]] = r
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shop", default="data/eval/products_shop_img.jsonl")
    ap.add_argument("--eval", default="data/eval/eval_set.jsonl")
    ap.add_argument("--cache", default="data/eval/emb_text.npz")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--ceiling", action="store_true", help="recall@K only, no LLM")
    ap.add_argument("--rerank", action="store_true", help="run the LLM rerank")
    ap.add_argument("--model", default=None)
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        rl = np.array(["a", "a", "b", "c"])
        sims = np.array([0.9, 0.1, 0.8, 0.7])      # nearest: a, then b, then c
        assert topk_leaves(sims, rl, 3) == ["a", "b", "c"], topk_leaves(sims, rl, 3)
        assert parse_choice('{"choice": 2}', 5) == 2
        assert parse_choice("the answer is 3.", 5) == 3
        assert parse_choice("99", 5) is None
        print("selftest ok")
        return

    pool, q_pids, r_pids = reconstruct_split(Path(args.shop), Path(args.eval))
    z = np.load(args.cache)
    q_emb, r_emb = z["q"], z["r"]
    assert q_emb.shape[0] == len(q_pids) and r_emb.shape[0] == len(r_pids), \
        "cache/split mismatch — re-run `python -m src.eval_retrieval` to refresh cache"
    r_leaf = np.array([pool[p]["leaf"] for p in r_pids])
    leaf_path = {pool[p]["leaf"]: (pool[p]["parent"], pool[p]["super"]) for p in pool}
    gt = [pool[p]["leaf"] for p in q_pids]

    sims = q_emb @ r_emb.T                          # (Q, R)

    if args.ceiling or not args.rerank:
        print("=== recall@K ceiling (max top-1 any reranker can reach) ===")
        for k in (1, 3, 5, 10, 15, 20, 30):
            hit = sum(gt[i] in topk_leaves(sims[i], r_leaf, k) for i in range(len(q_pids)))
            print(f"  recall@{k:<3}= {hit/len(q_pids):.4f}")
        if not args.rerank:
            return

    # ---- LLM rerank on a sample
    disp = load_display(Path(args.eval))
    model = args.model or OPENAI_MODEL
    client = make_client()
    n = min(args.limit, len(q_pids))
    print(f"\n=== rerank: provider=openai model={model} n={n} topk={args.topk} ===",
          flush=True)

    ret_top1 = rr_top1 = in_cands = 0
    for i in range(n):
        pid = q_pids[i]
        cands = topk_leaves(sims[i], r_leaf, args.topk)
        d = disp.get(pid, {})
        title = d.get("title_ru") or d.get("title_ro") or ""
        pick = llm_pick(client, model, title, d.get("brand"), d.get("price"),
                        cands, leaf_path)
        ret_top1 += cands[0] == gt[i]
        rr_top1 += pick == gt[i]
        in_cands += gt[i] in cands
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{n}  retrieval_top1={ret_top1/(i+1):.3f}  "
                  f"rerank_top1={rr_top1/(i+1):.3f}", flush=True)

    print("\n=== RESULT ===")
    print(json.dumps({
        "n": n, "topk": args.topk,
        "retrieval_top1": round(ret_top1 / n, 4),
        "gt_in_candidates(recall@k)": round(in_cands / n, 4),
        "pipeline_top1(retrieval+rerank)": round(rr_top1 / n, 4),
    }, indent=2))


if __name__ == "__main__":
    main()
