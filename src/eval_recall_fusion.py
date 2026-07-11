"""Does fusing the product IMAGE into candidate generation raise the recall
ceiling that caps the reranker?

Full-scale version of the Stage-B probe: builds image embeddings for the WHOLE
reference pool (not a 5/leaf cap), fuses image+text at the leaf-score level,
and reports recall@K vs text-only.

Fusion (per query q, leaf l), both cosine in [-1,1], 1-NN aggregated:
    text[q,l]  = max over leaf's reference products of  q_text . r_text
    image[q,l] = max over leaf's image-bearing reference of q_img . r_img
    fused      = a*text + (1-a)*image_filled
where image_filled = image[q,l] if defined else text[q,l] (so a missing image,
on either side, degrades gracefully to the text score — no scale break).

Image embeddings are cached to data/eval/emb_img.npz; first run downloads +
embeds (~20-30 min), later runs are instant.

Usage:
    python -m src.eval_recall_fusion
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.eval_candidates import recall_at_k
from src.eval_multimodal import download_images, embed_siglip, load_img_map
from src.eval_rerank import reconstruct_split
from src.eval_retrieval import load_pool  # noqa: F401  (re-exported convenience)


def embed_images_cached(pids, imgdir: Path, cache_path: Path) -> dict:
    cache = {}
    if cache_path.exists():
        z = np.load(cache_path, allow_pickle=True)
        cache = {int(p): e for p, e in zip(z["pids"], z["emb"])}
    missing = [p for p in pids if p not in cache]
    print(f"image embeddings: cached={len(cache)} missing={len(missing)}", flush=True)
    if missing:
        ok, emb = embed_siglip(missing, imgdir)
        for p, e in zip(ok, emb):
            cache[int(p)] = e
        allp = list(cache.keys())
        np.savez(cache_path, pids=np.array(allp),
                 emb=np.array([cache[p] for p in allp], dtype=np.float32))
        print(f"cached -> {cache_path} ({len(cache)} total)", flush=True)
    return cache


def leaf_max(sims: np.ndarray, cols_by_leaf, n_leaves: int) -> np.ndarray:
    """(Q, R) sims -> (Q, n_leaves) max similarity per leaf. Leaves with no
    columns stay -inf."""
    out = np.full((sims.shape[0], n_leaves), -np.inf, np.float32)
    for c, cols in enumerate(cols_by_leaf):
        if len(cols):
            out[:, c] = sims[:, cols].max(1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shop", default="data/eval/products_shop_img.jsonl")
    ap.add_argument("--eval", default="data/eval/eval_set.jsonl")
    ap.add_argument("--raw", default="data/eval/products_raw_img.jsonl")
    ap.add_argument("--tcache", default="data/eval/emb_text.npz")
    ap.add_argument("--icache", default="data/eval/emb_img.npz")
    ap.add_argument("--imgdir", default="data/eval/images")
    ap.add_argument("--workers", type=int, default=96,
                    help="parallel image downloads (latency-bound; raise on fast links)")
    args = ap.parse_args()

    pool, q_pids, r_pids = reconstruct_split(Path(args.shop), Path(args.eval))
    z = np.load(args.tcache)
    q_txt, r_txt = z["q"], z["r"]
    assert q_txt.shape[0] == len(q_pids) and r_txt.shape[0] == len(r_pids), \
        "text cache/split mismatch — re-run src.eval_retrieval"

    leaves = sorted({pool[p]["leaf"] for p in r_pids})
    lcode = {lf: i for i, lf in enumerate(leaves)}
    L = len(leaves)
    gt_code = np.array([lcode.get(pool[p]["leaf"], -1) for p in q_pids])
    r_code = np.array([lcode[pool[p]["leaf"]] for p in r_pids])
    cols_text = [np.where(r_code == c)[0] for c in range(L)]

    # ---- images: download + embed (cached) for everything with a mainImg
    img_map = load_img_map(Path(args.raw))
    want = [p for p in (q_pids + r_pids) if p in img_map]
    print(f"products with a mainImg: {len(want)} / {len(q_pids)+len(r_pids)}", flush=True)
    download_images(want, img_map, Path(args.imgdir), workers=args.workers)
    iemb = embed_images_cached(want, Path(args.imgdir), Path(args.icache))

    # ---- text leaf scores (full reference)
    sim_txt = (q_txt @ r_txt.T).astype(np.float32)
    text_leaf = leaf_max(sim_txt, cols_text, L)

    # ---- image leaf scores (reference products that actually have an image)
    r_img_rows = [(i, iemb[p]) for i, p in enumerate(r_pids) if p in iemb]
    r_img_idx = np.array([i for i, _ in r_img_rows])
    r_img = np.array([e for _, e in r_img_rows], dtype=np.float32)
    r_img_code = r_code[r_img_idx]
    cols_img = [np.where(r_img_code == c)[0] for c in range(L)]

    q_has = np.array([p in iemb for p in q_pids])
    q_img = np.array([iemb[p] if p in iemb else np.zeros(r_img.shape[1], np.float32)
                      for p in q_pids], dtype=np.float32)
    sim_img = (q_img @ r_img.T).astype(np.float32)
    image_leaf = leaf_max(sim_img, cols_img, L)          # -inf where no image ref
    # queries with no image: blank their image scores so fusion falls back to text
    image_leaf[~q_has] = -np.inf

    image_filled = np.where(np.isfinite(image_leaf), image_leaf, text_leaf)

    print(f"\nqueries={len(q_pids)} (with image={int(q_has.sum())})  "
          f"image-ref={len(r_img)}/{len(r_pids)}")
    print(f"{'strategy':<22} {'r@1':>7} {'r@5':>7} {'r@10':>7} {'r@20':>7}")

    def row(name, sc):
        r = recall_at_k(sc, gt_code)
        print(f"{name:<22} {r['r@1']:>7} {r['r@5']:>7} {r['r@10']:>7} {r['r@20']:>7}")

    row("text only", text_leaf)
    row("image only", np.where(np.isfinite(image_leaf), image_leaf, -1e9))
    for a in (0.8, 0.7, 0.6, 0.5, 0.4):
        row(f"fused a(text)={a:.1f}", a * text_leaf + (1 - a) * image_filled)


if __name__ == "__main__":
    main()
