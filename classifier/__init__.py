"""Hierarchical product categorization inference package.

Drop-in service for an online shop server. Loads three ONNX classifiers
(super -> parent -> leaf) and runs them in sequence with logit masking,
so each stage only chooses among descendants of the previous prediction.
"""
from .predictor import HierarchicalPredictor, Prediction
from .preprocessor import build_input_text, clean_text

__all__ = ["HierarchicalPredictor", "Prediction", "build_input_text", "clean_text"]
