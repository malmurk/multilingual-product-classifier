"""Unit tests for the worker — resolver, auto-assign vs. manual routing,
sweep, and stale-queue cleanup. The DB is mocked; no MariaDB needed.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from classifier.worker import CategoryResolver, Worker


# ---------------- CategoryResolver ----------------

def test_resolver_returns_id_for_known_name():
    rows = [(1, "Ноутбуки"), (2, "Смартфоны"), (3, "Холодильники")]
    resolver = CategoryResolver(loader=lambda: rows, refresh_seconds=3600)
    resolver.refresh()

    assert resolver.resolve("Ноутбуки") == 1
    assert resolver.resolve("Смартфоны") == 2
    assert len(resolver) == 3


def test_resolver_returns_none_for_unknown_name():
    rows = [(1, "Ноутбуки")]
    resolver = CategoryResolver(loader=lambda: rows, refresh_seconds=3600)
    resolver.refresh()

    assert resolver.resolve("Не существует") is None
    assert resolver.resolve("") is None


def test_resolver_skips_null_names():
    rows = [(1, "Ноутбуки"), (2, None), (3, "Смартфоны")]
    resolver = CategoryResolver(loader=lambda: rows, refresh_seconds=3600)
    resolver.refresh()

    assert len(resolver) == 2
    assert resolver.resolve("Ноутбуки") == 1
    assert resolver.resolve("Смартфоны") == 3


def test_resolver_lazy_initial_load():
    """Calling resolve() before refresh() should trigger a load."""
    calls = {"n": 0}

    def loader():
        calls["n"] += 1
        return [(1, "Ноутбуки")]

    resolver = CategoryResolver(loader=loader, refresh_seconds=3600)
    assert resolver.resolve("Ноутбуки") == 1
    assert calls["n"] == 1


def test_resolver_refreshes_after_interval():
    state = {"rows": [(1, "Old")]}
    fake_time = {"t": 0.0}

    def loader():
        return list(state["rows"])

    def clock():
        return fake_time["t"]

    resolver = CategoryResolver(loader=loader, refresh_seconds=10.0, clock=clock)
    resolver.refresh()
    assert resolver.resolve("Old") == 1

    state["rows"] = [(2, "New")]
    fake_time["t"] = 100.0

    assert resolver.resolve("New") == 2
    assert resolver.resolve("Old") is None


# ---------------- Worker: routing ----------------

def _make_pred(
    category,
    level="leaf",
    leaf_conf=0.95,
    super_label="Компьютеры",
    parent_label="Ноутбуки и аксессуары",
    leaf_label="Ноутбуки",
):
    return SimpleNamespace(
        category=category,
        category_level=level,
        leaf_confidence=leaf_conf,
        parent_confidence=0.8,
        super_confidence=0.7,
        super_label=super_label,
        parent_label=parent_label,
        leaf_label=leaf_label,
        to_dict=lambda: {
            "category": category,
            "category_level": level,
            "leaf_confidence": leaf_conf,
        },
    )


def _row(unsorted_id, product_id, *, title="Notebook", category_id=None):
    return {
        "unsorted_id": unsorted_id,
        "product_id": product_id,
        "title": title,
        "brand": None,
        "description": None,
        "category_id": category_id,
    }


def _build_worker(tmp_path: Path, db_mock, resolver, auto_threshold=0.93):
    return Worker(
        predictor=MagicMock(),
        db=db_mock,
        resolver=resolver,
        corrections_path=tmp_path / "corrections.jsonl",
        batch_size=10,
        poll_interval=1.0,
        auto_threshold=auto_threshold,
        install_signal_handlers=False,
    )


def test_high_confidence_leaf_auto_assigns(tmp_path):
    db = MagicMock()
    db.fetch_batch.return_value = [_row(1, 7)]
    resolver = CategoryResolver(loader=lambda: [(11, "Ноутбуки")], refresh_seconds=3600)
    resolver.refresh()

    worker = _build_worker(tmp_path, db, resolver)
    worker.predictor.predict_batch.return_value = [
        _make_pred("Ноутбуки", leaf_conf=0.96)
    ]

    worker._process_one_batch()

    db.assign_category.assert_called_once_with(7, 11, 1)
    db.route_to_manual.assert_not_called()

    entry = json.loads((tmp_path / "corrections.jsonl").read_text(encoding="utf-8").strip())
    assert entry["reason"] == "assigned"


def test_low_confidence_routes_to_manual(tmp_path):
    db = MagicMock()
    db.fetch_batch.return_value = [_row(2, 8)]
    resolver = CategoryResolver(loader=lambda: [(11, "Ноутбуки")], refresh_seconds=3600)
    resolver.refresh()

    worker = _build_worker(tmp_path, db, resolver)
    worker.predictor.predict_batch.return_value = [
        _make_pred("Ноутбуки", leaf_conf=0.85)
    ]

    worker._process_one_batch()

    db.assign_category.assert_not_called()
    db.route_to_manual.assert_called_once()
    kwargs = db.route_to_manual.call_args.kwargs
    assert kwargs["product_id"] == 8
    assert kwargs["unsorted_id"] == 2
    assert kwargs["reason"] == "low_confidence"
    assert kwargs["predicted_category"] == "Ноутбуки"
    assert kwargs["category_level"] == "leaf"
    assert abs(kwargs["confidence"] - 0.85) < 1e-9
    assert kwargs["super_label"] == "Компьютеры"

    entry = json.loads((tmp_path / "corrections.jsonl").read_text(encoding="utf-8").strip())
    assert entry["reason"] == "low_confidence"


def test_non_leaf_routes_to_manual(tmp_path):
    db = MagicMock()
    db.fetch_batch.return_value = [_row(3, 5)]
    resolver = CategoryResolver(loader=lambda: [(11, "Ноутбуки")], refresh_seconds=3600)
    resolver.refresh()

    worker = _build_worker(tmp_path, db, resolver)
    worker.predictor.predict_batch.return_value = [
        _make_pred("Компьютеры", level="parent", leaf_conf=0.99)
    ]

    worker._process_one_batch()

    db.assign_category.assert_not_called()
    db.route_to_manual.assert_called_once()
    assert db.route_to_manual.call_args.kwargs["reason"] == "non_leaf"


def test_leaf_not_in_db_routes_to_manual(tmp_path):
    db = MagicMock()
    db.fetch_batch.return_value = [_row(4, 42)]
    resolver = CategoryResolver(loader=lambda: [(1, "Ноутбуки")], refresh_seconds=3600)
    resolver.refresh()

    worker = _build_worker(tmp_path, db, resolver)
    worker.predictor.predict_batch.return_value = [
        _make_pred("ИмяКоторогоНетВБазе", leaf_conf=0.96)
    ]

    worker._process_one_batch()

    db.assign_category.assert_not_called()
    db.route_to_manual.assert_called_once()
    assert db.route_to_manual.call_args.kwargs["reason"] == "leaf_not_in_db"


def test_already_categorized_row_skipped_and_dequeued(tmp_path):
    """If the marketplace dashboard categorized the product before we got to it,
    the unsorted row is dropped without calling the predictor."""
    db = MagicMock()
    db.fetch_batch.return_value = [_row(5, 9, category_id=42)]
    resolver = CategoryResolver(loader=lambda: [(11, "Ноутбуки")], refresh_seconds=3600)
    resolver.refresh()

    worker = _build_worker(tmp_path, db, resolver)

    n = worker._process_one_batch()

    assert n == 1
    db.drop_unsorted.assert_called_once_with(5)
    db.assign_category.assert_not_called()
    db.route_to_manual.assert_not_called()
    worker.predictor.predict_batch.assert_not_called()


def test_threshold_boundary_exact_match_auto_assigns(tmp_path):
    """Confidence == threshold should auto-assign (>=, not >)."""
    db = MagicMock()
    db.fetch_batch.return_value = [_row(6, 12)]
    resolver = CategoryResolver(loader=lambda: [(99, "Ноутбуки")], refresh_seconds=3600)
    resolver.refresh()

    worker = _build_worker(tmp_path, db, resolver, auto_threshold=0.93)
    worker.predictor.predict_batch.return_value = [
        _make_pred("Ноутбуки", leaf_conf=0.93)
    ]

    worker._process_one_batch()
    db.assign_category.assert_called_once_with(12, 99, 6)
