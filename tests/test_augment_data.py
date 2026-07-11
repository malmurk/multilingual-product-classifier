import json
from pathlib import Path
from src.augment_data import shuffle_pipes, drop_pipes, augment_record, augment_file


def test_shuffle_pipes_returns_same_parts():
    text = "samsung a20 | black | 64gb | 6.4 inch"
    result = shuffle_pipes(text)
    assert sorted(result.split(" | ")) == sorted(text.split(" | "))


def test_shuffle_pipes_single_part_unchanged():
    text = "samsung a20"
    assert shuffle_pipes(text) == "samsung a20"


def test_drop_pipes_removes_at_least_one():
    text = "samsung | black | 64gb | 6.4 inch"
    result = drop_pipes(text, drop_prob=1.0)
    assert result.count("|") < text.count("|")


def test_drop_pipes_always_keeps_first_part():
    text = "samsung | black | 64gb"
    result = drop_pipes(text, drop_prob=1.0)
    assert result.startswith("samsung")


def test_augment_record_returns_list():
    record = {"text": "samsung | black | 64gb", "leaf": "Смартфоны", "parent": "P", "super": "S"}
    results = augment_record(record, n=3)
    assert len(results) == 3
    for r in results:
        assert r["leaf"] == "Смартфоны"
        assert r["parent"] == "P"
        assert r["super"] == "S"


def test_augment_file_produces_more_records(tmp_path):
    input_file = tmp_path / "input.jsonl"
    records = [
        {"text": "samsung | black | 64gb", "leaf": "Смартфоны", "parent": "P", "super": "S"},
        {"text": "nokia | white | 32gb", "leaf": "Смартфоны", "parent": "P", "super": "S"},
    ]
    with open(input_file, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    output_file = tmp_path / "output.jsonl"
    augment_file(input_file, output_file, n=3)

    with open(output_file, encoding="utf-8") as f:
        output_records = [json.loads(line) for line in f]

    # 2 original + 2*3 augmented = 8
    assert len(output_records) == 8
