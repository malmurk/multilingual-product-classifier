import json
import numpy as np
from pathlib import Path
import pytest
from src.evaluate import compute_accuracy, mask_logits


def test_mask_logits_zeros_invalid():
    logits = np.array([[1.0, 2.0, 3.0, 4.0]])
    valid_indices = [1, 3]
    result = mask_logits(logits, valid_indices)
    assert result[0][0] == -np.inf or result[0][0] < -1e9
    assert result[0][2] == -np.inf or result[0][2] < -1e9
    assert result[0][1] == 2.0
    assert result[0][3] == 4.0


def test_mask_logits_argmax_respects_mask():
    logits = np.array([[10.0, 1.0, 1.0, 1.0]])
    valid_indices = [1, 2, 3]
    result = mask_logits(logits, valid_indices)
    assert np.argmax(result) != 0  # index 0 should be masked out


def test_compute_accuracy_perfect():
    predictions = [0, 1, 2]
    labels = [0, 1, 2]
    assert compute_accuracy(predictions, labels) == 1.0


def test_compute_accuracy_half():
    predictions = [0, 1, 2, 3]
    labels = [0, 1, 9, 9]
    assert compute_accuracy(predictions, labels) == 0.5


def test_compute_accuracy_empty():
    assert compute_accuracy([], []) == 0.0


def test_mask_logits_raises_on_empty_indices():
    logits = np.array([[1.0, 2.0, 3.0]])
    with pytest.raises(ValueError):
        mask_logits(logits, [])
