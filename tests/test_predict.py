import numpy as np
import pytest


def test_softmax_sums_to_one():
    from src.predict import _softmax
    logits = np.array([[1.0, 2.0, 3.0]])
    probs = _softmax(logits)
    assert abs(probs.sum() - 1.0) < 1e-6


def test_softmax_max_wins():
    from src.predict import _softmax
    logits = np.array([[0.0, 10.0, 0.0]])
    probs = _softmax(logits)
    assert np.argmax(probs) == 1


def test_mask_logits_blocks_invalid():
    from src.predict import _mask_logits
    logits = np.array([[1.0, 2.0, 3.0]])
    masked = _mask_logits(logits, valid_indices=[0, 2])
    assert masked[0, 1] < -1e9   # blocked
    assert masked[0, 0] == 1.0   # kept
    assert masked[0, 2] == 3.0   # kept


def test_mask_logits_valid_wins():
    from src.predict import _mask_logits
    # Even though index 1 has highest raw logit, it's masked — index 0 should win
    logits = np.array([[1.0, 100.0, 0.5]])
    masked = _mask_logits(logits, valid_indices=[0, 2])
    assert np.argmax(masked) == 0
