import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoTokenizer

from src.model import CategoryClassifier, BACKBONE


class _ExportWrapper(nn.Module):
    """Equivalent forward to CategoryClassifier, but bypasses
    XLMRobertaModel.forward so we dodge transformers' new masking helpers
    (create_bidirectional_mask -> sdpa_mask) which fail under torch.jit.trace
    on transformers >= 4.46 with:
        IndexError: tuple index out of range
        at q_length.shape[0], q_length[0].to(device)

    We call encoder.embeddings + encoder.encoder (the transformer layer stack)
    directly, and pre-build a legacy additive 4-D attention mask ourselves.
    Numerically identical to the training-time forward; weights untouched.
    """

    def __init__(self, cc: CategoryClassifier):
        super().__init__()
        # Peel the XLMRobertaModel apart so we never call its .forward.
        self.embeddings = cc.encoder.embeddings
        self.layers = cc.encoder.encoder  # XLMRobertaEncoder (stack of layers)
        self.dropout = cc.dropout
        self.classifier = cc.classifier

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        embeds = self.embeddings(input_ids=input_ids)

        # Build the same (batch, 1, 1, seq_len) additive mask that XLM-R's
        # eager attention path expects: 0.0 where we keep, large-negative where
        # we mask.  finfo.min is safer than -1e4 for fp32 export.
        dtype = embeds.dtype
        ext = attention_mask[:, None, None, :].to(dtype)
        ext = (1.0 - ext) * torch.finfo(dtype).min

        encoder_out = self.layers(embeds, attention_mask=ext)
        # encoder_out can be a BaseModelOutput or a tuple depending on version
        hidden = encoder_out[0] if isinstance(encoder_out, tuple) else encoder_out.last_hidden_state

        # Mean-pool over non-pad tokens (same as CategoryClassifier.forward)
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden.size()).float()
        pooled = torch.sum(hidden * mask_expanded, dim=1)
        pooled = pooled / torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        pooled = self.dropout(pooled)
        return self.classifier(pooled)


def export_to_onnx(
    checkpoint_path: Path,
    output_path: Path,
    num_classes: int = None,
    max_length: int = 128,
) -> None:
    # weights_only=False because our checkpoints contain the dataclass-asdict
    # of TrainingConfig, which includes pathlib.WindowsPath objects that
    # PyTorch 2.6's default safe unpickler refuses. These files come from
    # our own training runs so the security risk of weights_only=False is nil.
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if num_classes is None:
        # Auto-detect from the classifier layer in the checkpoint
        num_classes = ckpt["model_state_dict"]["classifier.weight"].shape[0]

    # Force eager attention on the encoder: the SDPA kernel uses fused ops
    # that don't trace cleanly either.
    model = CategoryClassifier(num_classes=num_classes, attn_implementation="eager")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    export_model = _ExportWrapper(model).eval()

    dummy_ids = torch.zeros(1, max_length, dtype=torch.long)
    dummy_mask = torch.ones(1, max_length, dtype=torch.long)

    # Sanity check: make sure the wrapper runs without error before tracing.
    with torch.no_grad():
        _ = export_model(dummy_ids, dummy_mask)

    torch.onnx.export(
        export_model,
        (dummy_ids, dummy_mask),
        str(output_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size"},
            "attention_mask": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
        opset_version=14,
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"Exported: {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)")


def quantize_onnx(input_path: Path, output_path: Path) -> None:
    quantize_dynamic(str(input_path), str(output_path), weight_type=QuantType.QInt8)
    print(f"Quantized: {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)")


def verify_onnx_output(onnx_path: Path, max_length: int = 128) -> np.ndarray:
    session = ort.InferenceSession(str(onnx_path))
    dummy_ids = np.zeros((1, max_length), dtype=np.int64)
    dummy_mask = np.ones((1, max_length), dtype=np.int64)
    logits = session.run(["logits"], {"input_ids": dummy_ids, "attention_mask": dummy_mask})[0]
    return logits


def save_tokenizer(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(BACKBONE)
    tokenizer.save_pretrained(str(output_dir))
    print(f"Tokenizer saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=["super", "parent", "leaf"])
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint or f"models/checkpoints/{args.stage}/best_model.pt")
    onnx_dir = Path("models/onnx")
    onnx_dir.mkdir(parents=True, exist_ok=True)

    full_path = onnx_dir / f"{args.stage}_classifier_full.onnx"
    quant_path = onnx_dir / f"{args.stage}_classifier.onnx"

    export_to_onnx(ckpt_path, full_path, num_classes=args.num_classes)
    quantize_onnx(full_path, quant_path)
    full_path.unlink()  # remove unquantized file

    logits = verify_onnx_output(quant_path)
    print(f"Verified output shape: {logits.shape}")

    save_tokenizer(Path("models/tokenizer"))
