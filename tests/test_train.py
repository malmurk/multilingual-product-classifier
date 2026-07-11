import json
from pathlib import Path
import pytest
import torch
from src.train import train_stage, TrainingConfig


@pytest.fixture
def tiny_dataset(tmp_path):
    """Write 20 records for one leaf category — enough to run 1 epoch."""
    from src.taxonomy import load_taxonomy
    taxonomy = load_taxonomy(Path("taxonomy_full.csv"))
    leaf = list(taxonomy["leaf_to_idx"].keys())[0]
    parent = taxonomy["leaf_to_parent"][leaf]
    super_cat = taxonomy["parent_to_super"][parent]

    records = [
        {"text": f"product {i}", "leaf": leaf, "parent": parent, "super": super_cat}
        for i in range(20)
    ]
    data_file = tmp_path / "tiny.jsonl"
    with open(data_file, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return data_file, tmp_path


def test_train_stage_produces_checkpoint(tiny_dataset):
    data_file, tmp_dir = tiny_dataset
    from src.taxonomy import load_taxonomy
    taxonomy = load_taxonomy(Path("taxonomy_full.csv"))

    cfg = TrainingConfig(
        stage="super",
        train_file=data_file,
        val_file=data_file,  # reuse for speed
        output_dir=tmp_dir / "checkpoints",
        max_epochs=1,
        batch_size=4,
        learning_rate=2e-5,
        max_length=32,  # shorter for speed
        early_stopping_patience=1,
    )
    best_val_loss = train_stage(cfg, taxonomy)
    assert isinstance(best_val_loss, float)
    assert (tmp_dir / "checkpoints" / "best_model.pt").exists()
