import json
import random
from pathlib import Path
from typing import Dict, List


def shuffle_pipes(text: str) -> str:
    parts = text.split(" | ")
    if len(parts) <= 1:
        return text
    first, rest = parts[0], parts[1:]
    random.shuffle(rest)
    return " | ".join([first] + rest)


def drop_pipes(text: str, drop_prob: float = 0.3) -> str:
    parts = text.split(" | ")
    if len(parts) <= 1:
        return text
    first = parts[0]
    # Keep each attribute part with probability (1 - drop_prob)
    rest = [p for p in parts[1:] if random.random() < (1.0 - drop_prob)]
    kept = [first] + rest
    return " | ".join(kept) if kept else first


def augment_record(record: Dict, n: int = 3) -> List[Dict]:
    results = []
    for _ in range(n):
        text = record["text"]
        apply_shuffle = random.random() > 0.5
        apply_drop = random.random() > 0.5
        # Always apply at least one transformation to avoid identical copies
        if not apply_shuffle and not apply_drop:
            apply_shuffle = True
        if apply_shuffle:
            text = shuffle_pipes(text)
        if apply_drop:
            text = drop_pipes(text)
        results.append({**record, "text": text})
    return results


def augment_file(input_file: Path, output_file: Path, n: int = 3) -> None:
    """Write original records + n augmented copies each to output_file."""
    with open(input_file, encoding="utf-8") as fin, \
         open(output_file, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")  # original
            for aug in augment_record(record, n=n):
                fout.write(json.dumps(aug, ensure_ascii=False) + "\n")
