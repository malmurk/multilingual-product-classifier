import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
from tqdm import tqdm

from src.taxonomy import load_taxonomy

BACKBONE_DIR = Path("models/tokenizer")
NEG_INF = -1e10


def mask_logits(logits: np.ndarray, valid_indices: List[int]) -> np.ndarray:
    if not valid_indices:
        raise ValueError("valid_indices must not be empty")
    result = np.full_like(logits, NEG_INF)
    result[:, valid_indices] = logits[:, valid_indices]
    return result


def compute_accuracy(predictions: List[int], labels: List[int]) -> float:
    if not predictions:
        return 0.0
    return sum(p == l for p, l in zip(predictions, labels)) / len(predictions)


def evaluate_stage(
    jsonl_file: Path,
    onnx_path: Path,
    stage: str,
    taxonomy: dict,
    max_length: int = 128,
    batch_size: int = 64,
) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(str(BACKBONE_DIR))
    session = ort.InferenceSession(str(onnx_path))
    label_key = stage
    label_to_idx = taxonomy[f"{stage}_to_idx"]

    texts, labels = [], []
    with open(jsonl_file, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line.strip())
            if r.get(label_key) in label_to_idx:
                texts.append(r["text"])
                labels.append(label_to_idx[r[label_key]])

    predictions = []
    for i in tqdm(range(0, len(texts), batch_size), desc=f"Evaluating {stage}"):
        batch_texts = texts[i: i + batch_size]
        enc = tokenizer(batch_texts, max_length=max_length, padding="max_length",
                        truncation=True, return_tensors="np")
        logits = session.run(["logits"], {
            "input_ids": enc["input_ids"].astype(np.int64),
            "attention_mask": enc["attention_mask"].astype(np.int64),
        })[0]
        predictions.extend(np.argmax(logits, axis=1).tolist())

    accuracy = compute_accuracy(predictions, labels)
    return {"stage": stage, "accuracy": accuracy, "n_samples": len(labels)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-file", default="data/processed/test.jsonl")
    args = parser.parse_args()

    taxonomy = load_taxonomy(Path("taxonomy_full.csv"))
    test_file = Path(args.test_file)

    print("\n=== Evaluation Results ===")
    for stage in ["super", "parent", "leaf"]:
        onnx_path = Path(f"models/onnx/{stage}_classifier.onnx")
        if not onnx_path.exists():
            print(f"  {stage}: model not found, skipping")
            continue
        result = evaluate_stage(test_file, onnx_path, stage, taxonomy)
        print(f"  {stage:8s}: {result['accuracy']*100:.1f}%  ({result['n_samples']} samples)")
