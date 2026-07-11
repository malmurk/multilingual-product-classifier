"""
gated_predictor.py — Confidence-gated predictor: pruned-leaf ONNX classifier
with multilingual-E5 retrieval fallback over the full 4,072-leaf taxonomy.

Routing rule:
    if max softmax prob (unmasked leaf classifier) >= threshold:
        use cascaded super -> parent -> leaf classifier (hot leaves)
    else:
        embed "query: <text>" with intfloat/multilingual-e5-large,
        dot-product against the 4,072-leaf index, return top-1.

Expected layout (relative to the parent project root):
    models/tokenizer/                  — XLM-RoBERTa tokenizer
    models/onnx/{super,parent,leaf}_classifier.onnx
    taxonomy_pruned.csv        — pruned-leaf taxonomy
    data/leaf_embeddings.npy           — (4072, 1024) L2-normalised
    data/leaf_embedding_map.json       — {"0": leaf_name, ...}
"""

import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

# ---------------------------------------------------------------------------
# Paths — resolved relative to this file (src/ inside Product filter/)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent      # Product filter/

TOKENIZER_DIR   = PROJECT_ROOT / "models" / "tokenizer"
ONNX_DIR        = PROJECT_ROOT / "models" / "onnx"
PRUNED_CSV      = PROJECT_ROOT / "taxonomy_pruned.csv"
EMBEDDINGS_FILE = PROJECT_ROOT / "data" / "leaf_embeddings.npy"
MAP_FILE        = PROJECT_ROOT / "data" / "leaf_embedding_map.json"

MAX_LENGTH  = 128
NEG_INF     = -1e10
E5_MODEL_ID = "intfloat/multilingual-e5-large"

# Make sibling modules (taxonomy.py etc.) importable
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def mask_logits(logits: np.ndarray, valid_indices: list) -> np.ndarray:
    result = np.full_like(logits, NEG_INF)
    result[:, valid_indices] = logits[:, valid_indices]
    return result


# ---------------------------------------------------------------------------
# GatedPredictor
# ---------------------------------------------------------------------------
class GatedPredictor:
    """
    Confidence-gated predictor over all 4,072 taxonomy leaves.

    Hot leaves (those with training data) -> cascaded ONNX classifier.
    Anything below `threshold` -> E5 retrieval over the full 4,072-leaf index.
    """

    def __init__(self, threshold: float = 0.3):
        self.threshold = threshold
        self._loaded = False

    # ------------------------------------------------------------------
    # Lazy load (heavy: ~2 GB; only once per process)
    # ------------------------------------------------------------------
    def _ensure_loaded(self):
        if self._loaded:
            return

        print("[gated] Loading tokenizer (XLM-RoBERTa) ...", flush=True)
        from transformers import AutoTokenizer
        self._tok = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))

        print("[gated] Loading ONNX classifiers (super / parent / leaf) ...", flush=True)
        self._sess = {}
        for stage in ("super", "parent", "leaf"):
            path = ONNX_DIR / f"{stage}_classifier.onnx"
            self._sess[stage] = ort.InferenceSession(str(path))

        print("[gated] Loading pruned taxonomy ...", flush=True)
        from src.taxonomy import load_taxonomy
        self._tax = load_taxonomy(str(PRUNED_CSV))

        print("[gated] Loading E5 embedding model ...", flush=True)
        import torch
        from transformers import AutoModel
        self._e5_tok = AutoTokenizer.from_pretrained(E5_MODEL_ID)
        self._e5_model = AutoModel.from_pretrained(E5_MODEL_ID)
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._e5_model = self._e5_model.to(self._device)
        self._e5_model.eval()
        print(f"[gated] E5 model on device: {self._device}", flush=True)

        print("[gated] Loading leaf embedding index ...", flush=True)
        self._emb_matrix = np.load(str(EMBEDDINGS_FILE))   # (4072, 1024) L2-norm'd
        with open(MAP_FILE, encoding="utf-8") as f:
            self._emb_map = json.load(f)                   # {"0": leaf_name, ...}

        self._loaded = True
        print("[gated] All models loaded.", flush=True)

    # ------------------------------------------------------------------
    # Internal: tokenize for classifier (XLM-RoBERTa, batch)
    # ------------------------------------------------------------------
    def _tokenize_classifier(self, texts: list[str]) -> dict:
        enc = self._tok(
            texts,
            max_length=MAX_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="np",
        )
        return {
            "input_ids":      enc["input_ids"].astype(np.int64),
            "attention_mask": enc["attention_mask"].astype(np.int64),
        }

    # ------------------------------------------------------------------
    # Internal: cascaded super -> parent -> leaf prediction (batch=1)
    # ------------------------------------------------------------------
    def _run_classifier_single(self, feeds: dict) -> tuple[str, float]:
        tax = self._tax

        sl = self._sess["super"].run(["logits"], feeds)[0]
        sp = softmax(sl)[0]
        si = int(np.argmax(sp))
        super_name = tax["idx_to_super"].get(str(si), "")

        valid_parents = [
            tax["parent_to_idx"][p]
            for p in tax["super_to_parents"].get(super_name, [])
            if p in tax["parent_to_idx"]
        ]
        pl = self._sess["parent"].run(["logits"], feeds)[0]
        if valid_parents:
            pl = mask_logits(pl, valid_parents)
        pp = softmax(pl)[0]
        pi = int(np.argmax(pp))
        parent_name = tax["idx_to_parent"].get(str(pi), "")

        valid_leaves = [
            tax["leaf_to_idx"][lf]
            for lf in tax["parent_to_leaves"].get(parent_name, [])
            if lf in tax["leaf_to_idx"]
        ]
        ll = self._sess["leaf"].run(["logits"], feeds)[0]
        if valid_leaves:
            ll = mask_logits(ll, valid_leaves)
        lp = softmax(ll)[0]
        li = int(np.argmax(lp))
        leaf_conf = float(lp[li])
        leaf_name = tax["idx_to_leaf"].get(str(li), "")

        return leaf_name, leaf_conf

    # ------------------------------------------------------------------
    # Internal: unmasked leaf classifier (used for the routing decision)
    # ------------------------------------------------------------------
    def _run_classifier_unmasked(self, feeds: dict) -> tuple[str, float]:
        ll = self._sess["leaf"].run(["logits"], feeds)[0]
        lp = softmax(ll)[0]
        li = int(np.argmax(lp))
        leaf_conf = float(lp[li])
        leaf_name = self._tax["idx_to_leaf"].get(str(li), "")
        return leaf_name, leaf_conf

    # ------------------------------------------------------------------
    # Internal: E5 encode batch -> (B, 1024), L2-normalised
    # ------------------------------------------------------------------
    def _e5_encode(self, texts: list[str]) -> np.ndarray:
        import torch

        enc = self._e5_tok(
            texts,
            max_length=128,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        input_ids      = enc["input_ids"].to(self._device)
        attention_mask = enc["attention_mask"].to(self._device)

        with torch.no_grad():
            out = self._e5_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
        hidden = out.last_hidden_state
        mask   = attention_mask[..., None].float()
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        pooled = pooled / pooled.norm(dim=1, keepdim=True).clamp(min=1e-9)
        return pooled.cpu().float().numpy()

    # ------------------------------------------------------------------
    # Internal: retrieval top-K for a single query vector
    # ------------------------------------------------------------------
    def _retrieval_top_k(self, q_vec: np.ndarray, k: int = 5) -> list[dict]:
        sims = self._emb_matrix @ q_vec
        top_idx = np.argsort(-sims)[:k]
        return [
            {"leaf": self._emb_map[str(i)], "sim": float(sims[i])}
            for i in top_idx
        ]

    # ------------------------------------------------------------------
    # Public: predict a single title
    # ------------------------------------------------------------------
    def predict(self, title: str) -> dict:
        self._ensure_loaded()

        feeds = self._tokenize_classifier([title])
        _, clf_conf = self._run_classifier_unmasked(feeds)

        q_vec = self._e5_encode(["query: " + title])[0]
        retr_top5 = self._retrieval_top_k(q_vec, k=5)

        clf_leaf, clf_prob = self._run_classifier_single(feeds)

        if clf_conf >= self.threshold:
            return {
                "leaf":            clf_leaf,
                "confidence":      clf_conf,
                "source":          "classifier",
                "classifier_top1": {"leaf": clf_leaf, "prob": clf_prob},
                "retrieval_top5":  retr_top5,
            }
        retr_leaf = retr_top5[0]["leaf"]
        retr_sim  = retr_top5[0]["sim"]
        return {
            "leaf":            retr_leaf,
            "confidence":      retr_sim,
            "source":          "retrieval",
            "classifier_top1": {"leaf": clf_leaf, "prob": clf_prob},
            "retrieval_top5":  retr_top5,
        }

    # ------------------------------------------------------------------
    # Public: predict a batch of titles
    # ------------------------------------------------------------------
    def predict_batch(self, titles: list[str], e5_batch: int = 64) -> list[dict]:
        self._ensure_loaded()
        results = []

        clf_leaves = []
        clf_confs  = []
        clf_probs  = []
        for title in titles:
            feeds = self._tokenize_classifier([title])
            _, raw_conf = self._run_classifier_unmasked(feeds)
            leaf_m, prob_m = self._run_classifier_single(feeds)
            clf_confs.append(raw_conf)
            clf_leaves.append(leaf_m)
            clf_probs.append(prob_m)

        all_q_vecs = []
        queries = ["query: " + t for t in titles]
        for start in range(0, len(queries), e5_batch):
            batch = queries[start: start + e5_batch]
            all_q_vecs.append(self._e5_encode(batch))
        all_q_vecs = np.concatenate(all_q_vecs, axis=0)

        for i, _title in enumerate(titles):
            retr_top5 = self._retrieval_top_k(all_q_vecs[i], k=5)
            conf = clf_confs[i]
            if conf >= self.threshold:
                results.append({
                    "leaf":            clf_leaves[i],
                    "confidence":      conf,
                    "source":          "classifier",
                    "classifier_top1": {"leaf": clf_leaves[i], "prob": clf_probs[i]},
                    "retrieval_top5":  retr_top5,
                })
            else:
                retr_leaf = retr_top5[0]["leaf"]
                retr_sim  = retr_top5[0]["sim"]
                results.append({
                    "leaf":            retr_leaf,
                    "confidence":      retr_sim,
                    "source":          "retrieval",
                    "classifier_top1": {"leaf": clf_leaves[i], "prob": clf_probs[i]},
                    "retrieval_top5":  retr_top5,
                })
        return results
