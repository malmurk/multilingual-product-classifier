import csv
import json
from pathlib import Path
from typing import Dict


def load_taxonomy(csv_path: Path) -> Dict:
    super_to_parents: Dict = {}
    parent_to_leaves: Dict = {}
    leaf_to_parent: Dict = {}
    parent_to_super: Dict = {}

    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            if len(row) < 3:
                continue
            super_cat = row[0].strip().strip('"')
            parent = row[1].strip().strip('"')
            leaf = row[2].strip().strip('"')
            if not super_cat or not parent or not leaf:
                continue

            super_to_parents.setdefault(super_cat, [])
            if parent not in super_to_parents[super_cat]:
                super_to_parents[super_cat].append(parent)

            parent_to_leaves.setdefault(parent, [])
            if leaf not in parent_to_leaves[parent]:
                parent_to_leaves[parent].append(leaf)

            leaf_to_parent[leaf] = parent
            parent_to_super[parent] = super_cat

    all_supers = sorted(super_to_parents.keys())
    all_parents = sorted(parent_to_leaves.keys())
    all_leaves = sorted({l for leaves in parent_to_leaves.values() for l in leaves})

    return {
        "super_to_parents": super_to_parents,
        "parent_to_leaves": parent_to_leaves,
        "leaf_to_parent": leaf_to_parent,
        "parent_to_super": parent_to_super,
        "super_to_idx": {s: i for i, s in enumerate(all_supers)},
        "parent_to_idx": {p: i for i, p in enumerate(all_parents)},
        "leaf_to_idx": {l: i for i, l in enumerate(all_leaves)},
        "idx_to_super": {str(i): s for i, s in enumerate(all_supers)},
        "idx_to_parent": {str(i): p for i, p in enumerate(all_parents)},
        "idx_to_leaf": {str(i): l for i, l in enumerate(all_leaves)},
        "num_super": len(all_supers),
        "num_parent": len(all_parents),
        "num_leaf": len(all_leaves),
    }


def save_hierarchy_json(taxonomy: Dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(taxonomy, f, ensure_ascii=False, indent=2)
