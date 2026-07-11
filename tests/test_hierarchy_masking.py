"""Unit tests for the hierarchical masking — the core of the 'narrow the
choice' algorithm the user asked for.

Verifies three invariants:

  1. The stage 2 mask narrows 601 parents down to the parents of the
     predicted super (e.g. 24 for "Компьютеры").
  2. The stage 3 mask narrows 3833 leaves down to the leaves of the
     predicted parent (typically <= 20).
  3. masked_softmax ignores out-of-branch logits even when they're
     much bigger than in-branch logits (proves the mask actually wins).

Also runs a stubbed end-to-end predictor — no real ONNX models needed,
we just inject fake logits — to confirm the pipeline wiring.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from classifier.hierarchy import Hierarchy, masked_softmax
from classifier.preprocessor import build_input_text, clean_text


REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = REPO_ROOT / "models" / "onnx"


# ------------------------------ preprocessor ------------------------------

def test_clean_text_preserves_cyrillic_and_latin():
    assert clean_text("Ноутбук Lenovo IdeaPad-3 15IAU7!!!") == "ноутбук lenovo ideapad-3 15iau7"


def test_build_input_text_concatenates_available_fields():
    got = build_input_text(
        title="Ноутбук Lenovo",
        brand="Lenovo",
        attributes={"RAM": "16GB", "SSD": "512GB"},
        description="15.6 inch IPS",
    )
    parts = got.split(" | ")
    # title, brand, 2 attrs, description
    assert len(parts) == 5
    assert parts[0] == "ноутбук lenovo"
    assert parts[1] == "lenovo"
    assert "16gb" in parts
    assert parts[-1].startswith("15")


def test_build_input_text_handles_missing_fields():
    got = build_input_text(title="Товар без полей")
    assert got == "товар без полей"


# ------------------------------ hierarchy masking ------------------------------

@pytest.fixture(scope="module")
def hierarchy() -> Hierarchy:
    if not (MODEL_DIR / "hierarchy.json").exists():
        pytest.skip("models/onnx/hierarchy.json not present")
    return Hierarchy.from_model_dir(MODEL_DIR)


def test_supers_count(hierarchy):
    assert hierarchy.num_super == 19


def test_computers_narrows_parent_space(hierarchy):
    """Stage 2 shouldn't face all 109 parents. Under 'Компьютеры' it
    should narrow to a small subset.
    """
    super_idx = hierarchy.super_to_idx["Компьютеры"]
    allowed = hierarchy.parents_under_super(super_idx)
    assert 1 <= allowed.size <= 30, (
        f"Expected 'Компьютеры' to narrow parents materially, got {allowed.size}"
    )
    # And it must be strictly smaller than the total
    assert allowed.size < hierarchy.num_parent


def test_every_super_narrows_parent_space(hierarchy):
    """No super should force the classifier to consider all 601 parents."""
    for s_idx in range(hierarchy.num_super):
        allowed = hierarchy.parents_under_super(s_idx)
        assert allowed.size > 0
        assert allowed.size < hierarchy.num_parent


def test_parents_narrow_leaf_space(hierarchy):
    """Stage 3 should never face 3833 options — masked to leaves under
    the chosen parent (~7 average in our taxonomy).
    """
    sizes = [
        hierarchy.leaves_under_parent(p_idx).size
        for p_idx in range(hierarchy.num_parent)
    ]
    sizes = [s for s in sizes if s > 0]
    assert sizes, "no parents have leaves — hierarchy is broken"
    assert max(sizes) < hierarchy.num_leaf  # always strictly narrower
    # Median should be tiny compared to 3833
    median = sorted(sizes)[len(sizes) // 2]
    assert median < 50


def test_masked_softmax_ignores_out_of_branch_logits():
    """Invariant: a very large logit outside the allowed set must be
    suppressed — the mask, not the raw score, decides the winner.
    """
    logits = np.zeros(10, dtype=np.float32)
    logits[0] = 100.0   # strongest overall, but OUTSIDE the allowed set
    logits[5] = 1.0     # weaker, but INSIDE

    allowed = np.array([3, 5, 7])
    idx, conf, probs = masked_softmax(logits, allowed)

    assert idx == 5, "masked_softmax picked an out-of-branch label"
    assert conf > 0.5
    # out-of-branch probabilities must be ~0
    assert probs[0] < 1e-6


def test_masked_softmax_normalizes_within_branch():
    """Probabilities over the allowed set should sum to ~1."""
    logits = np.random.randn(100).astype(np.float32)
    allowed = np.array([2, 17, 42, 99])
    _, _, probs = masked_softmax(logits, allowed)
    assert np.isclose(probs[allowed].sum(), 1.0, atol=1e-5)
    # Everything outside the allowed set must be ~0
    mask_out = np.ones(100, dtype=bool)
    mask_out[allowed] = False
    assert probs[mask_out].sum() < 1e-6


# ------------------------------ end-to-end pipeline (stubbed) ------------------------------

def _make_stubbed_predictor(hierarchy):
    """Build a HierarchicalPredictor without actually loading ONNX/transformers.
    Returns the instance with _tokenize and _infer replaced by stubs so we
    can inject deterministic logits.
    """
    from classifier.predictor import HierarchicalPredictor

    # Bypass __init__ — we'll wire the minimum attributes by hand.
    p = HierarchicalPredictor.__new__(HierarchicalPredictor)
    p.hierarchy = hierarchy
    p.thresholds = {"super": 0.5, "parent": 0.6, "leaf": 0.75}
    p.max_length = 128
    p.super_sess = SimpleNamespace(name="super")
    p.parent_sess = SimpleNamespace(name="parent")
    p.leaf_sess = SimpleNamespace(name="leaf")
    p._tokenize = lambda text: {"input_ids": np.zeros((1, 1), dtype=np.int64)}
    return p


def test_end_to_end_narrows_correctly(hierarchy):
    """Feed fake logits that say 'definitely Компьютеры' and verify the
    stages 2/3 see a narrowed decision space, not the full taxonomy.
    """
    p = _make_stubbed_predictor(hierarchy)

    # Logits that prefer "Компьютеры" at stage 1
    super_idx = hierarchy.super_to_idx["Компьютеры"]
    super_logits = np.full(hierarchy.num_super, -10.0, dtype=np.float32)
    super_logits[super_idx] = 20.0

    allowed_parents = hierarchy.parents_under_super(super_idx)
    target_parent_idx = int(allowed_parents[0])

    # Parent logits: try to trick it by putting a huge logit on an
    # out-of-branch parent. Masking must win.
    parent_logits = np.full(hierarchy.num_parent, -10.0, dtype=np.float32)
    # pick an out-of-branch parent (any that isn't in allowed_parents)
    out_of_branch = next(
        i for i in range(hierarchy.num_parent) if i not in set(allowed_parents.tolist())
    )
    parent_logits[out_of_branch] = 100.0             # trap
    parent_logits[target_parent_idx] = 5.0           # correct answer

    allowed_leaves = hierarchy.leaves_under_parent(target_parent_idx)
    if allowed_leaves.size == 0:
        pytest.skip("target parent has no leaves (pick a different one)")
    target_leaf_idx = int(allowed_leaves[0])

    leaf_logits = np.full(hierarchy.num_leaf, -10.0, dtype=np.float32)
    leaf_logits[target_leaf_idx] = 8.0

    call_log = []

    def fake_infer(session, tokens):
        call_log.append(session.name)
        if session.name == "super":
            return super_logits.reshape(1, -1)
        if session.name == "parent":
            return parent_logits.reshape(1, -1)
        return leaf_logits.reshape(1, -1)

    p._infer = fake_infer

    pred = p._run_stages("любой текст")

    assert pred.category_level == "leaf"
    assert pred.super_label == "Компьютеры"
    assert pred.parent_label == hierarchy.parent_labels[target_parent_idx]
    assert pred.leaf_label == hierarchy.leaf_labels[target_leaf_idx]
    # Stage 2 saw a narrowed set — NOT 601
    assert pred.stage2_choices == int(allowed_parents.size)
    assert pred.stage2_choices < hierarchy.num_parent
    # Stage 3 saw a narrowed set — NOT 3833
    assert pred.stage3_choices == int(allowed_leaves.size)
    assert pred.stage3_choices < hierarchy.num_leaf
    # Mask beat the trap logit
    assert pred.parent_label != hierarchy.parent_labels[out_of_branch]
    # All three models were called in order
    assert call_log == ["super", "parent", "leaf"]


def test_low_super_confidence_stops_pipeline(hierarchy):
    p = _make_stubbed_predictor(hierarchy)

    # Flat logits -> confidence ~ 1/20 = 0.05, below threshold
    flat = np.zeros(hierarchy.num_super, dtype=np.float32)

    def fake_infer(session, tokens):
        assert session.name == "super", "should stop before parent/leaf"
        return flat.reshape(1, -1)

    p._infer = fake_infer
    pred = p._run_stages("мусорный ввод")

    assert pred.category_level == "unknown"
    assert pred.needs_review is True
    assert pred.meta.get("stopped_at") == "super_below_threshold"
