import json
from pathlib import Path
import pytest
from src.dataset import merge_jsonl_files, split_dataset, ProductDataset


@pytest.fixture
def sample_jsonl(tmp_path):
    records = [
        {"text": f"product {i}", "leaf": "Автокомпрессоры",
         "parent": "Аварийное оборудование", "super": "Автотовары"}
        for i in range(100)
    ]
    p = tmp_path / "sample.jsonl"
    with open(p, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return p


def test_merge_jsonl_files(tmp_path, sample_jsonl):
    out = tmp_path / "merged.jsonl"
    merge_jsonl_files([sample_jsonl, sample_jsonl], out)
    with open(out, encoding="utf-8") as f:
        lines = f.readlines()
    assert len(lines) == 200


def test_split_dataset(tmp_path, sample_jsonl):
    train_f = tmp_path / "train.jsonl"
    val_f = tmp_path / "val.jsonl"
    test_f = tmp_path / "test.jsonl"
    split_dataset(sample_jsonl, train_f, val_f, test_f, train=0.8, val=0.1, test=0.1)

    def count(f): return sum(1 for _ in open(f, encoding="utf-8"))
    assert count(train_f) == 80
    assert count(val_f) == 10
    assert count(test_f) == 10


def test_product_dataset_len(sample_jsonl):
    from src.taxonomy import load_taxonomy
    taxonomy = load_taxonomy(Path("taxonomy_full.csv"))
    ds = ProductDataset(sample_jsonl, taxonomy, stage="super", max_length=128)
    assert len(ds) == 100


def test_product_dataset_item_shape(sample_jsonl):
    from src.taxonomy import load_taxonomy
    taxonomy = load_taxonomy(Path("taxonomy_full.csv"))
    ds = ProductDataset(sample_jsonl, taxonomy, stage="super", max_length=128)
    item = ds[0]
    assert "input_ids" in item
    assert "attention_mask" in item
    assert "label" in item
    assert item["input_ids"].shape == (128,)
    assert 0 <= item["label"].item() < taxonomy["num_super"]
