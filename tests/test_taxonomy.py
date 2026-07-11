import pytest
from pathlib import Path
from src.taxonomy import load_taxonomy, save_hierarchy_json

TAXONOMY_CSV = Path("taxonomy_full.csv")


def test_load_taxonomy_returns_expected_keys():
    t = load_taxonomy(TAXONOMY_CSV)
    for key in ["super_to_parents", "parent_to_leaves", "leaf_to_parent",
                "parent_to_super", "super_to_idx", "parent_to_idx",
                "leaf_to_idx", "idx_to_super", "idx_to_parent", "idx_to_leaf",
                "num_super", "num_parent", "num_leaf"]:
        assert key in t, f"Missing key: {key}"


def test_load_taxonomy_counts():
    t = load_taxonomy(TAXONOMY_CSV)
    assert t["num_super"] == 20
    assert t["num_parent"] == 601
    assert t["num_leaf"] == 3833


def test_hierarchy_consistency():
    t = load_taxonomy(TAXONOMY_CSV)
    for parent, super_cat in t["parent_to_super"].items():
        assert parent in t["super_to_parents"][super_cat]
    for leaf, parent in t["leaf_to_parent"].items():
        assert leaf in t["parent_to_leaves"][parent]


def test_idx_mappings_are_inverses():
    t = load_taxonomy(TAXONOMY_CSV)
    for idx, name in t["idx_to_super"].items():
        assert t["super_to_idx"][name] == int(idx)
    for idx, name in t["idx_to_parent"].items():
        assert t["parent_to_idx"][name] == int(idx)
    for idx, name in t["idx_to_leaf"].items():
        assert t["leaf_to_idx"][name] == int(idx)


def test_save_hierarchy_json(tmp_path):
    t = load_taxonomy(TAXONOMY_CSV)
    out = tmp_path / "hierarchy.json"
    save_hierarchy_json(t, out)
    assert out.exists()
    import json
    data = json.loads(out.read_text(encoding="utf-8"))
    assert "super_to_parents" in data
    assert "num_leaf" in data
