import json
import torch
import numpy as np
from pathlib import Path
import pytest
from src.export_onnx import export_to_onnx, quantize_onnx, verify_onnx_output


@pytest.fixture
def trained_checkpoint(tmp_path):
    """Save a minimal untrained checkpoint for testing export."""
    from src.model import CategoryClassifier
    model = CategoryClassifier(num_classes=20)
    ckpt_path = tmp_path / "best_model.pt"
    torch.save({"model_state_dict": model.state_dict(), "config": {"stage": "super"}}, ckpt_path)
    return ckpt_path


def test_export_to_onnx_creates_file(trained_checkpoint, tmp_path):
    onnx_path = tmp_path / "model.onnx"
    export_to_onnx(trained_checkpoint, onnx_path, num_classes=20, max_length=128)
    assert onnx_path.exists()
    assert onnx_path.stat().st_size > 0


def test_quantize_onnx_creates_smaller_file(trained_checkpoint, tmp_path):
    onnx_path = tmp_path / "model.onnx"
    quant_path = tmp_path / "model_int8.onnx"
    export_to_onnx(trained_checkpoint, onnx_path, num_classes=20, max_length=128)
    quantize_onnx(onnx_path, quant_path)
    assert quant_path.exists()
    assert quant_path.stat().st_size < onnx_path.stat().st_size


def test_verify_onnx_output_shape(trained_checkpoint, tmp_path):
    onnx_path = tmp_path / "model.onnx"
    export_to_onnx(trained_checkpoint, onnx_path, num_classes=20, max_length=128)
    logits = verify_onnx_output(onnx_path, max_length=128)
    assert logits.shape == (1, 20)
