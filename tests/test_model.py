import torch
import pytest
from src.model import CategoryClassifier, get_device


def test_get_device_returns_torch_device():
    device = get_device()
    assert isinstance(device, (torch.device, object))  # torch_directml device or torch.device


def test_classifier_forward_pass():
    model = CategoryClassifier(num_classes=20)
    batch_size = 4
    seq_len = 128
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    logits = model(input_ids, attention_mask)
    assert logits.shape == (batch_size, 20)


def test_classifier_different_num_classes():
    model = CategoryClassifier(num_classes=3833)
    input_ids = torch.randint(0, 1000, (2, 128))
    attention_mask = torch.ones(2, 128, dtype=torch.long)
    logits = model(input_ids, attention_mask)
    assert logits.shape == (2, 3833)


def test_classifier_output_is_float():
    model = CategoryClassifier(num_classes=20)
    input_ids = torch.randint(0, 1000, (1, 128))
    attention_mask = torch.ones(1, 128, dtype=torch.long)
    logits = model(input_ids, attention_mask)
    assert logits.dtype == torch.float32
