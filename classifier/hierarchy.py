"""Load taxonomy + build the masks used at inference.

Two masks are pre-computed once at startup:

  super_parent_mask[super_idx]  -> bool array over all parents
      True for parents that belong to that super. ~24/601 True for "Компьютеры".

  parent_leaf_mask[parent_idx]  -> bool array over all leaves
      True for leaves that belong to that parent. ~7/3833 True on average.

Applied as: logits[~mask] = -inf  before softmax. That forces the
classifier to pick from the narrowed set, matching the 3-stage design.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


class Hierarchy:
    def __init__(
        self,
        super_labels: Dict[str, str],
        parent_labels: Dict[str, str],
        leaf_labels: Dict[str, str],
        super_to_parents: Dict[str, List[str]],
        parent_to_leaves: Dict[str, List[str]],
    ):
        # index -> label
        self.super_labels = {int(k): v for k, v in super_labels.items()}
        self.parent_labels = {int(k): v for k, v in parent_labels.items()}
        self.leaf_labels = {int(k): v for k, v in leaf_labels.items()}

        # label -> index
        self.super_to_idx = {v: k for k, v in self.super_labels.items()}
        self.parent_to_idx = {v: k for k, v in self.parent_labels.items()}
        self.leaf_to_idx = {v: k for k, v in self.leaf_labels.items()}

        self.num_super = len(self.super_labels)
        self.num_parent = len(self.parent_labels)
        self.num_leaf = len(self.leaf_labels)

        # Pre-compute boolean masks (one row per super / per parent)
        self.super_parent_mask = np.zeros(
            (self.num_super, self.num_parent), dtype=bool
        )
        for super_name, parents in super_to_parents.items():
            s_idx = self.super_to_idx.get(super_name)
            if s_idx is None:
                continue
            for p in parents:
                p_idx = self.parent_to_idx.get(p)
                if p_idx is not None:
                    self.super_parent_mask[s_idx, p_idx] = True

        self.parent_leaf_mask = np.zeros(
            (self.num_parent, self.num_leaf), dtype=bool
        )
        for parent_name, leaves in parent_to_leaves.items():
            p_idx = self.parent_to_idx.get(parent_name)
            if p_idx is None:
                continue
            for l in leaves:
                l_idx = self.leaf_to_idx.get(l)
                if l_idx is not None:
                    self.parent_leaf_mask[p_idx, l_idx] = True

    # --------- narrowing helpers (what the user asked for) ---------

    def parents_under_super(self, super_idx: int) -> np.ndarray:
        """Indices of parents allowed given the chosen super. ~24 for Компьютеры."""
        return np.where(self.super_parent_mask[super_idx])[0]

    def leaves_under_parent(self, parent_idx: int) -> np.ndarray:
        """Indices of leaves allowed given the chosen parent. ~7 on average."""
        return np.where(self.parent_leaf_mask[parent_idx])[0]

    # --------- loading ---------

    @classmethod
    def from_model_dir(cls, model_dir: Path) -> "Hierarchy":
        model_dir = Path(model_dir)
        super_labels = _load_json(model_dir / "super_labels.json")
        parent_labels = _load_json(model_dir / "parent_labels.json")
        leaf_labels = _load_json(model_dir / "leaf_labels.json")
        h = _load_json(model_dir / "hierarchy.json")
        return cls(
            super_labels=super_labels,
            parent_labels=parent_labels,
            leaf_labels=leaf_labels,
            super_to_parents=h["super_to_parents"],
            parent_to_leaves=h["parent_to_leaves"],
        )


def _load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# --------- masked softmax ---------

def masked_softmax(logits: np.ndarray, allowed_idx: np.ndarray) -> Tuple[int, float, np.ndarray]:
    """Zero out non-allowed logits, softmax, return (argmax_idx, confidence, probs).

    Operates on a single 1-D logit vector. `allowed_idx` is a 1-D array of
    the indices that survive the mask (from parents_under_super or
    leaves_under_parent). The returned index is in the full label space —
    no remapping needed on the caller side.
    """
    if allowed_idx.size == 0:
        # Degenerate (shouldn't happen with a valid hierarchy) — fall back
        # to unrestricted softmax.
        probs = _softmax(logits)
        idx = int(np.argmax(probs))
        return idx, float(probs[idx]), probs

    masked = np.full_like(logits, -1e9, dtype=np.float32)
    masked[allowed_idx] = logits[allowed_idx]
    probs = _softmax(masked)
    idx = int(np.argmax(probs))
    return idx, float(probs[idx]), probs


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)
