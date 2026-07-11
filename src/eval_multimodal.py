"""Stage B: does adding the product image lift leaf accuracy on the real
held-out set?

Same kNN retrieval as src.eval_retrieval, on the same reference/query split,
but scored three ways on identical products:
  - text   : multilingual-e5 (handles RU/RO)
  - image  : SigLIP image embeddings
  - fused  : alpha * sim_text + (1-alpha) * sim_image  (score-level late fusion)

SigLIP's text tower is English-centric, so we deliberately fuse SigLIP-IMAGE
with e5-TEXT rather than using SigLIP text on Cyrillic/Romanian.

To keep downloads bounded we cap the reference at --ref-per-leaf products;
queries are the full eval_set. alpha=1.0 in the sweep is pure text (a sanity
check), alpha=0.0 is pure image.

Usage:
    python -m src.eval_multimodal --ref-per-leaf 5
"""
from __future__ import annotations

import argparse
import io
import json
import ssl
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from src.eval_retrieval import embed_e5, load_pool, load_query_ids, metrics_from_sims

SIGLIP = "google/siglip-base-patch16-224"
_CTX = ssl.create_default_context()
_CTX.check_hostname = False
_CTX.verify_mode = ssl.CERT_NONE


def load_img_map(raw_path: Path) -> dict:
    m = {}
    for line in open(raw_path, encoding="utf-8"):
        line = line.strip()
        if line:
            r = json.loads(line)
            if r.get("mainImg"):
                m[r["id"]] = r["mainImg"]
    return m


def _fetch(pid_url_dir):
    pid, url, outdir = pid_url_dir
    dest = outdir / str(pid)
    if dest.exists() and dest.stat().st_size > 0:
        return pid
    full = "https:" + url if url.startswith("//") else url
    try:
        req = urllib.request.Request(full, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=8, context=_CTX) as r:
            data = r.read()
        if len(data) < 256:
            return None
        dest.write_bytes(data)
        return pid
    except Exception:
        return None


def download_images(pids, img_map, outdir: Path, workers: int = 16) -> set:
    outdir.mkdir(parents=True, exist_ok=True)
    jobs = [(p, img_map[p], outdir) for p in pids if p in img_map]
    ok = set()
    with ThreadPoolExecutor(workers) as ex:
        for i, res in enumerate(ex.map(_fetch, jobs)):
            if res is not None:
                ok.add(res)
            if i % 500 == 0:
                print(f"  downloaded {i}/{len(jobs)} (ok={len(ok)})", flush=True)
    return ok


def embed_siglip(pids, imgdir: Path, batch: int = 64):
    """Return (ok_pids, normalized image embeddings) aligned by row. Unreadable
    images are skipped so the arrays stay consistent with ok_pids."""
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel
    from src.model import get_device

    proc = AutoImageProcessor.from_pretrained(SIGLIP)
    model = AutoModel.from_pretrained(SIGLIP)
    device = get_device()
    model = model.to(device).eval()

    ok_pids, vecs = [], []
    with torch.no_grad():
        for i in range(0, len(pids), batch):
            chunk = pids[i:i + batch]
            imgs, kept = [], []
            for p in chunk:
                try:
                    imgs.append(Image.open(imgdir / str(p)).convert("RGB"))
                    kept.append(p)
                except Exception:
                    continue
            if not imgs:
                continue
            pv = proc(images=imgs, return_tensors="pt").to(device)
            # vision_model.pooler_output is the image embedding; robust across
            # transformers versions (get_image_features return type varies).
            feat = model.vision_model(pixel_values=pv["pixel_values"]).pooler_output
            feat = torch.nn.functional.normalize(feat, dim=-1)
            vecs.append(feat.cpu().numpy())
            ok_pids.extend(kept)
            if (i // batch) % 10 == 0:
                print(f"  siglip {i + len(chunk)}/{len(pids)}", flush=True)
    return ok_pids, np.vstack(vecs).astype(np.float32)


def cap_reference(pool, qids, per_leaf, seed=42):
    import random
    rng = random.Random(seed)
    by_leaf = defaultdict(list)
    for p in pool:
        if p not in qids:
            by_leaf[pool[p]["leaf"]].append(p)
    ref = []
    for items in by_leaf.values():
        rng.shuffle(items)
        ref.extend(items[:per_leaf])
    return ref


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shop", default="data/eval/products_shop_img.jsonl")
    ap.add_argument("--raw", default="data/eval/products_raw_img.jsonl")
    ap.add_argument("--eval", default="data/eval/eval_set.jsonl")
    ap.add_argument("--imgdir", default="data/eval/images")
    ap.add_argument("--ref-per-leaf", type=int, default=5)
    args = ap.parse_args()

    pool = load_pool(Path(args.shop))
    qids = load_query_ids(Path(args.eval)) & set(pool)
    img_map = load_img_map(Path(args.raw))

    q_pids = [p for p in pool if p in qids]
    r_pids = cap_reference(pool, qids, args.ref_per_leaf)
    print(f"queries={len(q_pids)}  capped_reference={len(r_pids)}", flush=True)

    imgdir = Path(args.imgdir)
    print("downloading images...", flush=True)
    have = download_images(q_pids + r_pids, img_map, imgdir)
    print(f"images on disk: {len(have)}", flush=True)

    # keep only products with text AND a usable image
    q_pids = [p for p in q_pids if p in have]
    r_pids = [p for p in r_pids if p in have]

    print("embedding images (siglip)...", flush=True)
    q_ip, q_img = embed_siglip(q_pids, imgdir)
    r_ip, r_img = embed_siglip(r_pids, imgdir)
    # realign to the pids that actually embedded
    q_pids, r_pids = q_ip, r_ip

    print("embedding text (e5)...", flush=True)
    q_txt = embed_e5([pool[p]["text"] for p in q_pids])
    r_txt = embed_e5([pool[p]["text"] for p in r_pids])

    leaf_path = {pool[p]["leaf"]: (pool[p]["parent"], pool[p]["super"]) for p in pool}
    q_meta = [pool[p] for p in q_pids]
    r_leaf = [pool[p]["leaf"] for p in r_pids]

    sim_txt = q_txt @ r_txt.T
    sim_img = q_img @ r_img.T

    print(f"\n=== Multimodal eval (n_query={len(q_pids)}, ref={len(r_pids)}) ===")
    print(f"{'alpha(text)':>12} {'top1_leaf':>10} {'top5_leaf':>10} {'top1_parent':>12} {'top1_super':>11}")
    for a in (1.0, 0.7, 0.6, 0.5, 0.4, 0.3, 0.0):
        m = metrics_from_sims(a * sim_txt + (1 - a) * sim_img, r_leaf, leaf_path, q_meta)
        tag = "  (text)" if a == 1.0 else "  (image)" if a == 0.0 else ""
        print(f"{a:>12.2f} {m['top1_leaf']:>10} {m['top5_leaf']:>10} "
              f"{m['top1_parent']:>12} {m['top1_super']:>11}{tag}")


if __name__ == "__main__":
    main()
