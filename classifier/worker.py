"""Background worker — polls the ``unsorted`` queue, classifies products,
auto-assigns ``products.category_id`` for high-confidence predictions,
and routes uncertain ones to the ``manual_sorting`` queue for a human.

Flow per tick (every POLL_INTERVAL seconds):

  1. Sweep: delete rows from ``manual_sorting`` whose product has since
     received a category_id (a human fixed it from the dashboard).

  2. Fetch up to BATCH_SIZE rows from ``unsorted`` joined with
     ``products``.  Skip rows whose product already has a category_id
     (the human got there first) — just remove them from the queue.

  3. Classify the batch.

  4. For each prediction:
       - leaf-level AND confidence >= THRESHOLD_LEAF_AUTO AND the
         predicted leaf name resolves to a categories.id
            -> UPDATE products.category_id, DELETE from unsorted
       - otherwise (low confidence, non-leaf stop, or unknown name)
            -> INSERT into manual_sorting, DELETE from unsorted

  5. If the batch was empty, sleep POLL_INTERVAL seconds.

Confidence is NOT persisted to ``products`` — the live schema has no
column for it.  It goes into ``manual_sorting.confidence`` for rows
sent to manual review and into ``corrections.jsonl`` for every row
(useful for the active-learning retrain loop).

Schema dependencies: ``products``, ``categories``, ``unsorted``,
``manual_sorting`` (the last two are created by
a queue-tables migration against your own schema — not included in this repository).
"""
from __future__ import annotations

import json
import logging
import os
import signal
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .predictor import HierarchicalPredictor

logger = logging.getLogger("classifier.worker")


# ---------------- name -> id resolver ----------------

class CategoryResolver:
    """In-memory cache mapping ``categories.name`` -> ``categories.id``
    for ``active = 1`` rows. Refreshes lazily on access after
    ``refresh_seconds`` have elapsed.

    ``loader`` is a zero-arg callable returning a list of ``(id, name)``
    tuples — injected so tests can mock the DB.
    """

    def __init__(
        self,
        loader: Callable[[], List[tuple]],
        refresh_seconds: float = 3600.0,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._loader = loader
        self._refresh_seconds = float(refresh_seconds)
        self._clock = clock
        self._map: Dict[str, int] = {}
        self._loaded_at: float = 0.0

    def refresh(self) -> None:
        rows = self._loader()
        new_map: Dict[str, int] = {}
        for cid, name in rows:
            if name is None:
                continue
            new_map[str(name)] = int(cid)
        self._map = new_map
        self._loaded_at = self._clock()
        logger.info("CategoryResolver refreshed: %d active leaf names", len(new_map))

    def _maybe_refresh(self) -> None:
        if not self._map or (self._clock() - self._loaded_at) >= self._refresh_seconds:
            self.refresh()

    def resolve(self, name: str) -> Optional[int]:
        self._maybe_refresh()
        return self._map.get(name)

    def __len__(self) -> int:
        return len(self._map)


# ---------------- DB adapter ----------------

class DBAdapter:
    """Thin DB-API 2.0 wrapper around the live marketplace MariaDB.

    Targets your production schema (``products``, ``categories``) plus the two
    queue tables added by your migration (see module docstring).
    """

    FETCH_SQL = """
        SELECT u.id          AS unsorted_id,
               p.id          AS product_id,
               p.title       AS title,
               p.brand       AS brand,
               p.description AS description,
               p.category_id AS category_id
        FROM unsorted u
        JOIN products p ON p.id = u.product_id
        ORDER BY u.id
        LIMIT %s
    """

    UPDATE_SQL = """
        UPDATE products
        SET category_id = %s
        WHERE id = %s
    """

    DELETE_UNSORTED_SQL = "DELETE FROM unsorted WHERE id = %s"

    INSERT_MANUAL_SQL = """
        INSERT INTO manual_sorting
            (product_id, predicted_category, category_level, confidence,
             super_label, parent_label, leaf_label, reason)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            predicted_category = VALUES(predicted_category),
            category_level     = VALUES(category_level),
            confidence         = VALUES(confidence),
            super_label        = VALUES(super_label),
            parent_label       = VALUES(parent_label),
            leaf_label         = VALUES(leaf_label),
            reason             = VALUES(reason),
            created_at         = CURRENT_TIMESTAMP
    """

    SWEEP_MANUAL_SQL = """
        DELETE ms FROM manual_sorting ms
        JOIN products p ON p.id = ms.product_id
        WHERE p.category_id IS NOT NULL
    """

    LOAD_CATEGORIES_SQL = """
        SELECT id, name
        FROM categories
        WHERE active = 1
    """

    def __init__(self, conn):
        self.conn = conn

    def fetch_batch(self, batch_size: int) -> List[Dict[str, Any]]:
        with self.conn.cursor() as cur:
            cur.execute(self.FETCH_SQL, (batch_size,))
            rows = cur.fetchall()
            cols = [c[0] for c in cur.description]
        return [dict(zip(cols, row)) for row in rows]

    def load_categories(self) -> List[tuple]:
        with self.conn.cursor() as cur:
            cur.execute(self.LOAD_CATEGORIES_SQL)
            return list(cur.fetchall())

    def assign_category(self, product_id: int, category_id: int, unsorted_id: int) -> None:
        """Atomic auto-classification: set category_id and remove from queue."""
        with self.conn.cursor() as cur:
            cur.execute(self.UPDATE_SQL, (category_id, product_id))
            cur.execute(self.DELETE_UNSORTED_SQL, (unsorted_id,))
        self.conn.commit()

    def route_to_manual(
        self,
        product_id: int,
        unsorted_id: int,
        predicted_category: Optional[str],
        category_level: Optional[str],
        confidence: Optional[float],
        super_label: Optional[str],
        parent_label: Optional[str],
        leaf_label: Optional[str],
        reason: str,
    ) -> None:
        with self.conn.cursor() as cur:
            cur.execute(
                self.INSERT_MANUAL_SQL,
                (
                    product_id,
                    predicted_category,
                    category_level,
                    confidence,
                    super_label,
                    parent_label,
                    leaf_label,
                    reason,
                ),
            )
            cur.execute(self.DELETE_UNSORTED_SQL, (unsorted_id,))
        self.conn.commit()

    def drop_unsorted(self, unsorted_id: int) -> None:
        """Remove a stale queue entry without classifying (e.g. product
        was already categorized by a human before we got to it)."""
        with self.conn.cursor() as cur:
            cur.execute(self.DELETE_UNSORTED_SQL, (unsorted_id,))
        self.conn.commit()

    def sweep_manual(self) -> int:
        """Delete manual_sorting rows whose product is now categorized."""
        with self.conn.cursor() as cur:
            cur.execute(self.SWEEP_MANUAL_SQL)
            n = cur.rowcount
        self.conn.commit()
        return int(n or 0)


# ---------------- worker loop ----------------

class Worker:
    def __init__(
        self,
        predictor: HierarchicalPredictor,
        db: DBAdapter,
        resolver: CategoryResolver,
        corrections_path: Path,
        batch_size: int = 100,
        poll_interval: float = 1800.0,
        auto_threshold: float = 0.93,
        install_signal_handlers: bool = True,
    ):
        self.predictor = predictor
        self.db = db
        self.resolver = resolver
        self.corrections_path = corrections_path
        self.batch_size = batch_size
        self.poll_interval = poll_interval
        self.auto_threshold = float(auto_threshold)
        self._stop = False
        if install_signal_handlers:
            signal.signal(signal.SIGTERM, self._handle_signal)
            signal.signal(signal.SIGINT, self._handle_signal)

    def _handle_signal(self, *_):
        logger.info("Shutdown requested")
        self._stop = True

    def run(self) -> None:
        logger.info(
            "Worker starting (batch=%d, interval=%.1fs, auto_threshold=%.2f)",
            self.batch_size,
            self.poll_interval,
            self.auto_threshold,
        )
        while not self._stop:
            try:
                swept = self.db.sweep_manual()
                if swept:
                    logger.info("Swept %d resolved rows from manual_sorting", swept)
                n = self._process_one_batch()
                if n == 0:
                    time.sleep(self.poll_interval)
            except Exception:
                logger.exception("Tick failed, backing off")
                time.sleep(self.poll_interval)

    def _log_correction(self, fh, product_id: int, pred, reason: str) -> None:
        fh.write(
            json.dumps(
                {
                    "product_id": product_id,
                    "reason": reason,
                    "prediction": pred.to_dict(),
                },
                ensure_ascii=False,
            )
            + "\n"
        )

    def _process_one_batch(self) -> int:
        rows = self.db.fetch_batch(self.batch_size)
        if not rows:
            return 0

        # Skip rows whose product already has a category — a human got
        # there first.  Just clean them out of the queue.
        live_rows: List[Dict[str, Any]] = []
        for r in rows:
            if r.get("category_id") is not None:
                self.db.drop_unsorted(r["unsorted_id"])
                continue
            live_rows.append(r)

        if not live_rows:
            return len(rows)  # we still did work (cleaned the queue)

        products = [
            {
                "title": r.get("title") or "",
                "brand": r.get("brand"),
                "description": r.get("description"),
                "attributes": None,
            }
            for r in live_rows
        ]
        preds = self.predictor.predict_batch(products)

        with open(self.corrections_path, "a", encoding="utf-8") as fh:
            for row, pred in zip(live_rows, preds):
                product_id = row["product_id"]
                unsorted_id = row["unsorted_id"]
                self._handle_one(fh, product_id, unsorted_id, pred)

        logger.info(
            "Processed %d products (%d skipped as already-categorized)",
            len(live_rows),
            len(rows) - len(live_rows),
        )
        return len(rows)

    def _handle_one(self, fh, product_id: int, unsorted_id: int, pred) -> None:
        """Route a single prediction to either auto-assign or manual queue."""
        leaf_conf = float(pred.leaf_confidence or 0.0)
        is_leaf = pred.category_level == "leaf"

        # Reasons that force manual review, in priority order
        if not is_leaf:
            self._send_to_manual(
                fh, product_id, unsorted_id, pred, reason="non_leaf"
            )
            return

        if leaf_conf < self.auto_threshold:
            self._send_to_manual(
                fh, product_id, unsorted_id, pred, reason="low_confidence"
            )
            return

        category_id = self.resolver.resolve(pred.category)
        if category_id is None:
            logger.warning(
                "Predicted leaf %r for product %s not in categories table",
                pred.category,
                product_id,
            )
            self._send_to_manual(
                fh, product_id, unsorted_id, pred, reason="leaf_not_in_db"
            )
            return

        # Auto-assign
        try:
            self.db.assign_category(product_id, category_id, unsorted_id)
        except Exception:
            logger.exception("Failed to assign product %s", product_id)
            self._log_correction(fh, product_id, pred, reason="db_update_failed")
            return

        self._log_correction(fh, product_id, pred, reason="assigned")

    def _send_to_manual(
        self, fh, product_id: int, unsorted_id: int, pred, reason: str
    ) -> None:
        try:
            self.db.route_to_manual(
                product_id=product_id,
                unsorted_id=unsorted_id,
                predicted_category=pred.category,
                category_level=pred.category_level,
                confidence=float(pred.leaf_confidence or 0.0)
                if pred.category_level == "leaf"
                else float(
                    pred.parent_confidence
                    if pred.category_level == "parent"
                    else (pred.super_confidence or 0.0)
                ),
                super_label=pred.super_label,
                parent_label=pred.parent_label,
                leaf_label=pred.leaf_label,
                reason=reason,
            )
        except Exception:
            logger.exception(
                "Failed to route product %s to manual_sorting", product_id
            )
            self._log_correction(fh, product_id, pred, reason="db_update_failed")
            return

        self._log_correction(fh, product_id, pred, reason=reason)


# ---------------- entrypoint ----------------

def _connect_from_env():
    """Connect to the live marketplace MariaDB via pymysql.

    DB_URL form: ``mysql+pymysql://user:pass@host:3306/catalog_db?charset=utf8mb4``
    Plain ``mysql://`` is also accepted.
    """
    url = os.getenv("DB_URL")
    if not url:
        raise RuntimeError(
            "DB_URL is required (e.g. mysql+pymysql://user:pw@host:3306/catalog_db)"
        )
    if not (url.startswith("mysql") or url.startswith("mariadb")):
        raise RuntimeError(f"Unsupported DB_URL scheme: {url}")

    import pymysql
    from urllib.parse import urlparse, parse_qs

    cleaned = url.replace("mysql+pymysql://", "mysql://", 1)
    cleaned = cleaned.replace("mariadb+pymysql://", "mysql://", 1)
    u = urlparse(cleaned)
    qs = parse_qs(u.query)
    charset = (qs.get("charset", ["utf8mb4"]) or ["utf8mb4"])[0]
    return pymysql.connect(
        host=u.hostname,
        port=u.port or 3306,
        user=u.username,
        password=u.password,
        database=(u.path or "/").lstrip("/"),
        charset=charset,
        autocommit=False,
    )


def main() -> None:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    model_dir = Path(os.getenv("MODEL_DIR", "/app/models"))
    corrections = Path(os.getenv("CORRECTIONS_PATH", "/app/data/corrections.jsonl"))
    corrections.parent.mkdir(parents=True, exist_ok=True)

    # Predictor's needs_review is no longer load-bearing — the worker
    # makes the auto/manual decision via THRESHOLD_LEAF_AUTO.  Predictor
    # thresholds stay low so we always get a leaf-level guess to show
    # the human reviewer.
    thresholds = {
        "super": float(os.getenv("THRESHOLD_SUPER", "0.50")),
        "parent": float(os.getenv("THRESHOLD_PARENT", "0.50")),
        "leaf": float(os.getenv("THRESHOLD_LEAF", "0.50")),
    }

    predictor = HierarchicalPredictor(
        model_dir=model_dir,
        tokenizer_name=os.getenv("TOKENIZER", "xlm-roberta-base"),
        thresholds=thresholds,
    )

    conn = _connect_from_env()
    db = DBAdapter(conn)

    resolver = CategoryResolver(
        loader=db.load_categories,
        refresh_seconds=float(os.getenv("RESOLVER_REFRESH_SECONDS", "3600")),
    )
    resolver.refresh()  # fail fast if categories table is unreachable

    worker = Worker(
        predictor=predictor,
        db=db,
        resolver=resolver,
        corrections_path=corrections,
        batch_size=int(os.getenv("BATCH_SIZE", "100")),
        poll_interval=float(os.getenv("POLL_INTERVAL", "1800")),
        auto_threshold=float(os.getenv("THRESHOLD_LEAF_AUTO", "0.93")),
    )
    worker.run()


if __name__ == "__main__":
    main()
