import json
import random
from pathlib import Path
from typing import List

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

BACKBONE = "intfloat/multilingual-e5-base"
_tokenizer = None


def _get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(BACKBONE)
    return _tokenizer


def merge_jsonl_files(input_files: List[Path], output_file: Path) -> None:
    with open(output_file, "w", encoding="utf-8") as fout:
        for f in input_files:
            with open(f, encoding="utf-8") as fin:
                for line in fin:
                    if line.strip():
                        fout.write(line)


def split_dataset(
    input_file: Path,
    train_file: Path,
    val_file: Path,
    test_file: Path,
    train: float = 0.8,
    val: float = 0.1,
    test: float = 0.1,
    seed: int = 42,
) -> None:
    assert abs(train + val + test - 1.0) < 1e-6, "Splits must sum to 1.0"
    with open(input_file, encoding="utf-8") as f:
        records = [line for line in f if line.strip()]

    rng = random.Random(seed)
    rng.shuffle(records)
    n = len(records)
    n_train = int(n * train)
    n_val = int(n * val)

    splits = [
        (train_file, records[:n_train]),
        (val_file, records[n_train: n_train + n_val]),
        (test_file, records[n_train + n_val:]),
    ]
    for path, lines in splits:
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(lines)


class ProductDataset(Dataset):
    LABEL_KEYS = {"super": "super", "parent": "parent", "leaf": "leaf"}
    IDX_KEYS = {"super": "super_to_idx", "parent": "parent_to_idx", "leaf": "leaf_to_idx"}

    def __init__(self, jsonl_file: Path, taxonomy: dict, stage: str, max_length: int = 128,
                 device: torch.device | None = None):
        assert stage in self.LABEL_KEYS, f"stage must be one of {list(self.LABEL_KEYS)}"
        self.label_key = self.LABEL_KEYS[stage]
        self.label_to_idx = taxonomy[self.IDX_KEYS[stage]]
        tokenizer = _get_tokenizer()

        texts, labels = [], []
        with open(jsonl_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    if r.get(self.label_key) in self.label_to_idx:
                        texts.append(r["text"])
                        labels.append(self.label_to_idx[r[self.label_key]])

        print(f"  Tokenizing {len(texts)} samples...")
        enc = tokenizer(
            texts,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        # Pin tensors to GPU to eliminate per-batch CPU→GPU transfers
        if device is not None:
            self.input_ids = enc["input_ids"].to(device)
            self.attention_mask = enc["attention_mask"].to(device)
            self.labels = torch.tensor(labels, dtype=torch.long).to(device)
        else:
            self.input_ids = enc["input_ids"]
            self.attention_mask = enc["attention_mask"]
            self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "label": self.labels[idx],
        }
