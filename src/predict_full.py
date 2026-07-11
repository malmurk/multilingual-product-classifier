"""
predict_full.py — Production predictor covering all 4,072 taxonomy leaves.

Architecture:
  1. Cascaded ONNX classifier (super -> parent -> leaf) handles "hot" leaves.
  2. Confidence gate at threshold T (default loaded from data/retrieval_config.json).
  3. multilingual-E5 retrieval over the full 4,072-leaf index handles low-confidence inputs.

CLI:
    cd "<project-root>"
    set PYTHONIOENCODING=utf-8
    .venv312/Scripts/python.exe -m src.predict_full --title "Pylesos Samsung 2000W"
    .venv312/Scripts/python.exe -m src.predict_full --title "..." --threshold 0.40 --json

Python API:
    from src.predict_full import FullPredictor
    p = FullPredictor()                   # threshold from data/retrieval_config.json
    r = p.predict("...")
    rs = p.predict_batch(["...", "..."])

Output schema:
    {
      "leaf":            str,             # final predicted leaf
      "confidence":      float,           # classifier prob OR retrieval cosine sim
      "source":          "classifier" | "retrieval",
      "classifier_top1": {"leaf": str, "prob": float},
      "retrieval_top5":  [{"leaf": str, "sim": float}, ...],
    }
"""

import argparse
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths — resolved relative to this file (src/ inside Product filter/)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent      # Product filter/
CONFIG_FILE  = PROJECT_ROOT / "data" / "retrieval_config.json"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load_threshold_from_config() -> float:
    if not CONFIG_FILE.exists():
        raise FileNotFoundError(
            f"retrieval_config.json not found at {CONFIG_FILE}. "
            "Move the file from the data-boost project's data/ folder."
        )
    with open(CONFIG_FILE, encoding="utf-8") as f:
        cfg = json.load(f)
    return float(cfg["threshold"])


def _safe_title(title) -> str:
    if not isinstance(title, str):
        title = str(title) if title is not None else ""
    return title.strip()


class FullPredictor:
    """
    Production-ready predictor covering all 4,072 taxonomy leaves.

    Threshold is loaded from data/retrieval_config.json unless overridden.
    """

    def __init__(self, threshold: float | None = None):
        if threshold is None:
            threshold = _load_threshold_from_config()
        self._threshold = float(threshold)
        self._gated = None  # lazy

    @property
    def threshold(self) -> float:
        return self._threshold

    def _ensure_loaded(self) -> None:
        if self._gated is not None:
            return
        from src.gated_predictor import GatedPredictor
        self._gated = GatedPredictor(threshold=self._threshold)
        self._gated._ensure_loaded()

    def predict(self, title: str) -> dict:
        title = _safe_title(title)
        if not title:
            return {
                "leaf":            "",
                "confidence":      0.0,
                "source":          "retrieval",
                "classifier_top1": {"leaf": "", "prob": 0.0},
                "retrieval_top5":  [],
            }
        self._ensure_loaded()
        return self._gated.predict(title)

    def predict_batch(self, titles: list[str]) -> list[dict]:
        if not titles:
            return []

        safe_titles = [_safe_title(t) for t in titles]
        results: list[dict | None] = [None] * len(safe_titles)
        batch_indices = []
        batch_titles  = []

        for i, t in enumerate(safe_titles):
            if not t:
                results[i] = {
                    "leaf":            "",
                    "confidence":      0.0,
                    "source":          "retrieval",
                    "classifier_top1": {"leaf": "", "prob": 0.0},
                    "retrieval_top5":  [],
                }
            else:
                batch_indices.append(i)
                batch_titles.append(t)

        if batch_titles:
            self._ensure_loaded()
            batch_results = self._gated.predict_batch(batch_titles)
            for idx, res in zip(batch_indices, batch_results):
                results[idx] = res

        return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        prog="predict_full.py",
        description=(
            "Multilingual product classifier with E5 retrieval fallback.\n"
            "Covers the full taxonomy (hot leaves via the cascade, long-tail leaves via retrieval)."
        ),
    )
    parser.add_argument("--title", type=str, required=True,
                        help="Product title to classify (wrap in quotes).")
    parser.add_argument("--threshold", type=float, default=None,
                        help=("Confidence gate (0-1). Default: value in "
                              "data/retrieval_config.json"))
    parser.add_argument("--json", action="store_true", default=False,
                        help="Output raw JSON instead of a human-readable summary.")
    args = parser.parse_args()

    predictor = FullPredictor(threshold=args.threshold)
    result = predictor.predict(args.title)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"Input        : {args.title[:100]}")
        print(f"Leaf         : {result['leaf']}")
        print(f"Source       : {result['source']}")
        print(f"Confidence   : {result['confidence']:.4f}")
        print(f"Classifier#1 : {result['classifier_top1']['leaf']} "
              f"(prob={result['classifier_top1']['prob']:.4f})")
        print("Retrieval#5  :")
        for i, hit in enumerate(result["retrieval_top5"], 1):
            print(f"  {i}. {hit['leaf']}  (sim={hit['sim']:.4f})")


if __name__ == "__main__":
    main()
