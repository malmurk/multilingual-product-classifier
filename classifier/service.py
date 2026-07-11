"""FastAPI HTTP service.

This is the drop-in endpoint the online shop server calls. Two ways to
consume it:

  POST /classify       -> one product in, one Prediction out  (synchronous,
                          low-latency — use this from the product import
                          pipeline itself)

  POST /classify/batch -> up to N products in, N predictions out (use this
                          for bulk imports; the background worker also
                          uses this internally)

  GET  /health         -> 200 once models are loaded
  GET  /hierarchy      -> returns the taxonomy tree (handy for the admin
                          panel's manual-review dropdowns)

Run locally:
    uvicorn classifier.service:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .predictor import HierarchicalPredictor

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger("classifier.service")


# ---------------- request/response schemas ----------------

class ProductIn(BaseModel):
    title: str = Field(..., description="Product title (Russian or Romanian)")
    brand: Optional[str] = None
    attributes: Optional[Dict[str, str]] = None
    description: Optional[str] = None


class BatchIn(BaseModel):
    products: List[ProductIn]


class PredictionOut(BaseModel):
    category: str
    category_level: str
    needs_review: bool
    super_label: Optional[str] = None
    parent_label: Optional[str] = None
    leaf_label: Optional[str] = None
    super_confidence: Optional[float] = None
    parent_confidence: Optional[float] = None
    leaf_confidence: Optional[float] = None
    stage2_choices: Optional[int] = None
    stage3_choices: Optional[int] = None
    meta: Dict[str, Any] = Field(default_factory=dict)


# ---------------- app + lifespan ----------------

app = FastAPI(title="Multilingual product classifier", version="1.0.0")
_predictor: Optional[HierarchicalPredictor] = None


@app.on_event("startup")
def _load():
    global _predictor
    model_dir = Path(os.getenv("MODEL_DIR", "/app/models"))
    tokenizer = os.getenv(
        "TOKENIZER",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    thresholds = {
        "super": float(os.getenv("THRESHOLD_SUPER", "0.50")),
        "parent": float(os.getenv("THRESHOLD_PARENT", "0.60")),
        "leaf": float(os.getenv("THRESHOLD_LEAF", "0.75")),
    }
    logger.info("Loading predictor from %s (thresholds=%s)", model_dir, thresholds)
    _predictor = HierarchicalPredictor(
        model_dir=model_dir,
        tokenizer_name=tokenizer,
        thresholds=thresholds,
    )
    logger.info("Predictor ready.")


def _get_predictor() -> HierarchicalPredictor:
    if _predictor is None:
        raise HTTPException(status_code=503, detail="Predictor not loaded yet")
    return _predictor


# ---------------- endpoints ----------------

@app.get("/health")
def health():
    return {"status": "ok" if _predictor is not None else "loading"}


@app.get("/hierarchy")
def hierarchy():
    p = _get_predictor()
    return {
        "num_super": p.hierarchy.num_super,
        "num_parent": p.hierarchy.num_parent,
        "num_leaf": p.hierarchy.num_leaf,
        "super_labels": p.hierarchy.super_labels,
        "parent_labels": p.hierarchy.parent_labels,
        "leaf_labels": p.hierarchy.leaf_labels,
    }


@app.post("/classify", response_model=PredictionOut)
def classify(body: ProductIn):
    p = _get_predictor()
    pred = p.predict(
        title=body.title,
        brand=body.brand,
        attributes=body.attributes,
        description=body.description,
    )
    return PredictionOut(**pred.to_dict())


@app.post("/classify/batch", response_model=List[PredictionOut])
def classify_batch(body: BatchIn):
    p = _get_predictor()
    if len(body.products) > 500:
        raise HTTPException(status_code=413, detail="Batch too large (max 500)")
    preds = p.predict_batch([prod.model_dump() for prod in body.products])
    return [PredictionOut(**pred.to_dict()) for pred in preds]
