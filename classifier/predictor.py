"""Three-stage masked predictor.

The key idea (what the user asked for):

    Stage 1 picks among 20 super-categories.
    Stage 2 is then ONLY allowed to choose parents inside that super
            (e.g. ~24 options under "Компьютеры", not 601).
    Stage 3 is then ONLY allowed to choose leaves inside that parent
            (~7 options on average, not 3833).

Masking is done on the raw logits before softmax, so confidence is
re-normalised within the narrowed branch.

Each stage has its own confidence threshold; below threshold the
prediction stops and the product is flagged for manual review with the
last confident label as the fallback category.
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import onnxruntime as ort

from .hierarchy import Hierarchy, masked_softmax
from .preprocessor import build_input_text

logger = logging.getLogger(__name__)


DEFAULT_THRESHOLDS = {
    "super": 0.50,
    "parent": 0.60,
    "leaf": 0.75,
}


@dataclass
class Prediction:
    # Final assignment (whatever level we stopped at)
    category: str
    category_level: str  # "leaf" | "parent" | "super" | "unknown"
    needs_review: bool

    # Full path when available
    super_label: Optional[str] = None
    parent_label: Optional[str] = None
    leaf_label: Optional[str] = None

    # Confidence per stage (None if stage didn't run)
    super_confidence: Optional[float] = None
    parent_confidence: Optional[float] = None
    leaf_confidence: Optional[float] = None

    # Narrowing stats (handy for debugging / the active-learning review queue)
    stage2_choices: Optional[int] = None  # how many parents survived the super mask
    stage3_choices: Optional[int] = None  # how many leaves survived the parent mask

    # Free-form (e.g. failure reason)
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class HierarchicalPredictor:
    """Loads three ONNX classifiers + the shared tokenizer and runs staged inference."""

    def __init__(
        self,
        model_dir: str | Path,
        tokenizer_name: str = "xlm-roberta-base",
        thresholds: Optional[Dict[str, float]] = None,
        max_length: int = 128,
        providers: Optional[List[str]] = None,
    ):
        model_dir = Path(model_dir)
        self.model_dir = model_dir
        self.thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
        self.max_length = max_length

        self.hierarchy = Hierarchy.from_model_dir(model_dir)

        # Tokenizer — must match the backbone the models were trained
        # against (``xlm-roberta-base``).  MiniLM tokenization is
        # incompatible and would silently produce wrong predictions.
        # The Dockerfile pre-downloads this name into the HF cache so
        # containers start offline.
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        providers = providers or ["CPUExecutionProvider"]
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.intra_op_num_threads = 0  # let ORT pick based on CPU

        self.super_sess = self._load_session(model_dir / "super_classifier.onnx", so, providers)
        self.parent_sess = self._load_session(model_dir / "parent_classifier.onnx", so, providers)
        self.leaf_sess = self._load_session(model_dir / "leaf_classifier.onnx", so, providers)

        logger.info(
            "HierarchicalPredictor ready: %d supers, %d parents, %d leaves",
            self.hierarchy.num_super,
            self.hierarchy.num_parent,
            self.hierarchy.num_leaf,
        )

    # --------------------------------------------------------------- public API

    def predict(
        self,
        title: str,
        attributes: Optional[Dict[str, str]] = None,
        description: Optional[str] = None,
        brand: Optional[str] = None,
    ) -> Prediction:
        """Classify a single product. See Prediction for the return shape."""
        text = build_input_text(
            title=title, brand=brand, attributes=attributes, description=description
        )
        if not text:
            return Prediction(
                category="",
                category_level="unknown",
                needs_review=True,
                meta={"error": "empty_input"},
            )
        return self._run_stages(text)

    def predict_batch(self, products: List[Dict[str, Any]]) -> List[Prediction]:
        """Classify a batch. Each item is a dict with keys: title, attributes, description, brand."""
        return [
            self.predict(
                title=p.get("title", ""),
                attributes=p.get("attributes"),
                description=p.get("description"),
                brand=p.get("brand"),
            )
            for p in products
        ]

    # --------------------------------------------------------------- internals

    def _run_stages(self, text: str) -> Prediction:
        tokens = self._tokenize(text)

        # -------- Stage 1: super (unrestricted — 20 classes) --------
        super_logits = self._infer(self.super_sess, tokens)[0]
        super_idx = int(np.argmax(super_logits))
        super_probs = _softmax_1d(super_logits)
        super_conf = float(super_probs[super_idx])
        super_label = self.hierarchy.super_labels[super_idx]

        if super_conf < self.thresholds["super"]:
            return Prediction(
                category="",
                category_level="unknown",
                needs_review=True,
                super_label=super_label,
                super_confidence=super_conf,
                meta={"stopped_at": "super_below_threshold"},
            )

        # -------- Stage 2: parent, MASKED to the chosen super branch --------
        allowed_parents = self.hierarchy.parents_under_super(super_idx)
        parent_logits = self._infer(self.parent_sess, tokens)[0]
        parent_idx, parent_conf, _ = masked_softmax(parent_logits, allowed_parents)
        parent_label = self.hierarchy.parent_labels[parent_idx]

        if parent_conf < self.thresholds["parent"]:
            return Prediction(
                category=super_label,
                category_level="super",
                needs_review=True,
                super_label=super_label,
                parent_label=parent_label,
                super_confidence=super_conf,
                parent_confidence=parent_conf,
                stage2_choices=int(allowed_parents.size),
                meta={"stopped_at": "parent_below_threshold"},
            )

        # -------- Stage 3: leaf, MASKED to the chosen parent branch --------
        allowed_leaves = self.hierarchy.leaves_under_parent(parent_idx)

        if allowed_leaves.size == 0:
            # Parent has no leaves registered — assign parent and flag.
            return Prediction(
                category=parent_label,
                category_level="parent",
                needs_review=True,
                super_label=super_label,
                parent_label=parent_label,
                super_confidence=super_conf,
                parent_confidence=parent_conf,
                stage2_choices=int(allowed_parents.size),
                stage3_choices=0,
                meta={"stopped_at": "parent_has_no_leaves"},
            )

        leaf_logits = self._infer(self.leaf_sess, tokens)[0]
        leaf_idx, leaf_conf, _ = masked_softmax(leaf_logits, allowed_leaves)
        leaf_label = self.hierarchy.leaf_labels[leaf_idx]

        if leaf_conf < self.thresholds["leaf"]:
            return Prediction(
                category=parent_label,
                category_level="parent",
                needs_review=True,
                super_label=super_label,
                parent_label=parent_label,
                leaf_label=leaf_label,
                super_confidence=super_conf,
                parent_confidence=parent_conf,
                leaf_confidence=leaf_conf,
                stage2_choices=int(allowed_parents.size),
                stage3_choices=int(allowed_leaves.size),
                meta={"stopped_at": "leaf_below_threshold"},
            )

        return Prediction(
            category=leaf_label,
            category_level="leaf",
            needs_review=False,
            super_label=super_label,
            parent_label=parent_label,
            leaf_label=leaf_label,
            super_confidence=super_conf,
            parent_confidence=parent_conf,
            leaf_confidence=leaf_conf,
            stage2_choices=int(allowed_parents.size),
            stage3_choices=int(allowed_leaves.size),
        )

    def _tokenize(self, text: str) -> Dict[str, np.ndarray]:
        enc = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="np",
        )
        # All three models share the same tokenizer / backbone inputs.
        return {
            "input_ids": enc["input_ids"].astype(np.int64),
            "attention_mask": enc["attention_mask"].astype(np.int64),
        }

    @staticmethod
    def _infer(session: ort.InferenceSession, tokens: Dict[str, np.ndarray]) -> np.ndarray:
        input_names = {i.name for i in session.get_inputs()}
        feed = {k: v for k, v in tokens.items() if k in input_names}
        outputs = session.run(None, feed)
        return outputs[0]

    @staticmethod
    def _load_session(path: Path, so: ort.SessionOptions, providers: List[str]) -> ort.InferenceSession:
        if not path.exists():
            raise FileNotFoundError(f"ONNX model not found: {path}")
        logger.info("Loading ONNX model: %s", path.name)
        return ort.InferenceSession(str(path), sess_options=so, providers=providers)


def _softmax_1d(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)
