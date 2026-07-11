import argparse
import dataclasses
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import ProductDataset
from src.model import CategoryClassifier, get_device
from src.taxonomy import load_taxonomy

BACKBONE = "intfloat/multilingual-e5-base"


@dataclass
class TrainingConfig:
    stage: str                  # "super" | "parent" | "leaf"
    train_file: Path
    val_file: Path
    output_dir: Path
    max_epochs: int = 20
    batch_size: int = 256
    learning_rate: float = 2e-5     # encoder LR
    head_lr_mult: float = 50.0      # classifier head LR = learning_rate * head_lr_mult
    weight_decay: float = 0.01
    max_length: int = 128
    dropout: float = 0.1
    early_stopping_patience: int = 3
    grad_clip: float = 1.0
    label_smoothing: float = 0.0
    warmup_steps: int = 500
    init_encoder_from: Optional[Path] = None


def _warmup_rocm(model: nn.Module, batch_size: int, max_length: int, device: torch.device) -> None:
    """Run one dummy forward+backward to trigger MIOpen kernel compilation before training starts."""
    print("  Warming up ROCm kernels (first run compiles HIP kernels, may take a minute)...")
    dummy_ids = torch.zeros(batch_size, max_length, dtype=torch.long, device=device)
    dummy_mask = torch.ones(batch_size, max_length, dtype=torch.long, device=device)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out = model(dummy_ids, dummy_mask)
        loss = out.sum()
    loss.backward()
    model.zero_grad()
    torch.cuda.synchronize()
    print("  Kernels ready.")


def _load_encoder_weights(model: nn.Module, ckpt_path: Path, device: torch.device) -> None:
    """Copy just the encoder submodule from another stage's checkpoint.
    Used to warm-start the leaf stage from the parent's trained encoder
    so it doesn't have to re-learn XLM-R fine-tuning from scratch.
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"]
    encoder_state = {
        k[len("encoder."):]: v for k, v in state.items() if k.startswith("encoder.")
    }
    missing, unexpected = model.encoder.load_state_dict(encoder_state, strict=False)
    print(
        f"  Warm-started encoder from {ckpt_path} "
        f"({len(encoder_state)} tensors, missing={len(missing)}, unexpected={len(unexpected)})"
    )


def _build_param_groups(model: nn.Module, encoder_lr: float, head_lr: float, weight_decay: float):
    """Split params so the classifier head gets a much higher LR than the
    pretrained encoder. With 3833 leaf classes the head has ~3M fresh params
    that need orders of magnitude more gradient updates than the encoder.
    """
    encoder_params, head_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (head_params if n.startswith("classifier") else encoder_params).append(p)
    return [
        {"params": encoder_params, "lr": encoder_lr, "weight_decay": weight_decay},
        {"params": head_params, "lr": head_lr, "weight_decay": weight_decay},
    ]


def _lr_lambda(step: int, warmup_steps: int, total_steps: int) -> float:
    """Linear warmup then cosine decay. Step-level schedule — critical so the
    first few hundred updates don't diverge on a fresh 3833-way head.
    """
    if step < warmup_steps:
        return (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    return 0.5 * (1 + math.cos(math.pi * progress))


def train_stage(cfg: TrainingConfig, taxonomy: dict, resume: bool = False) -> float:
    os.environ.setdefault("MIOPEN_FIND_MODE", "FAST")
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    device = get_device()
    num_classes = taxonomy[f"num_{cfg.stage}"]

    model = CategoryClassifier(num_classes=num_classes, dropout=cfg.dropout)
    model.to(device)

    train_ds = ProductDataset(cfg.train_file, taxonomy, stage=cfg.stage, max_length=cfg.max_length, device=device)
    val_ds = ProductDataset(cfg.val_file, taxonomy, stage=cfg.stage, max_length=cfg.max_length, device=device)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    encoder_lr = cfg.learning_rate
    head_lr = cfg.learning_rate * cfg.head_lr_mult
    print(f"  LR split: encoder={encoder_lr:.2e}  head={head_lr:.2e}  (head_lr_mult={cfg.head_lr_mult})")
    param_groups = _build_param_groups(model, encoder_lr, head_lr, cfg.weight_decay)
    optimizer = torch.optim.AdamW(param_groups)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

    total_steps = max(1, len(train_loader) * cfg.max_epochs)
    warmup_steps = min(cfg.warmup_steps, total_steps // 10)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: _lr_lambda(step, warmup_steps, total_steps),
    )

    best_val_loss = float("inf")
    patience_counter = 0
    start_epoch = 0
    global_step = 0

    checkpoint_path = cfg.output_dir / "best_model.pt"
    if resume and checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            except ValueError as e:
                print(f"  (optimizer state mismatch, restarting optimizer: {e})")
        best_val_loss = ckpt.get("val_loss", float("inf"))
        start_epoch = ckpt.get("epoch", 0)
        global_step = start_epoch * len(train_loader)
        for _ in range(global_step):
            scheduler.step()
        print(f"  Resumed from epoch {start_epoch}, best val_loss={best_val_loss:.4f}")
    elif cfg.init_encoder_from is not None and Path(cfg.init_encoder_from).exists():
        _load_encoder_weights(model, Path(cfg.init_encoder_from), device)
    elif resume:
        print(f"  No checkpoint found at {checkpoint_path}, starting fresh.")

    if device.type == "cuda":
        _warmup_rocm(model, cfg.batch_size, cfg.max_length, device)

    use_amp = device.type == "cuda"

    for epoch in range(start_epoch, start_epoch + cfg.max_epochs):
        # Train
        model.train()
        train_loss_sum = 0.0
        train_n = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.max_epochs} [train]"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16) if use_amp else torch.amp.autocast("cpu"):
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                print(f"  WARNING: non-finite loss at step {global_step}, skipping batch")
                optimizer.zero_grad()
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            scheduler.step()
            global_step += 1
            train_loss_sum += loss.item() * labels.size(0)
            train_n += labels.size(0)

        # Validate
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_samples = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16) if use_amp else torch.amp.autocast("cpu"):
                    logits = model(input_ids, attention_mask)
                    val_loss += criterion(logits, labels).item() * labels.size(0)
                val_correct += (logits.argmax(dim=-1) == labels).sum().item()
                val_samples += labels.size(0)

        val_loss = val_loss / val_samples if val_samples > 0 else float("inf")
        val_acc = val_correct / val_samples if val_samples > 0 else 0.0
        train_loss = train_loss_sum / train_n if train_n > 0 else float("inf")
        print(
            f"  Epoch {epoch+1} train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
            f"lr_enc={optimizer.param_groups[0]['lr']:.2e} "
            f"lr_head={optimizer.param_groups[1]['lr']:.2e}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch + 1,
                "val_loss": val_loss,
                "config": dataclasses.asdict(cfg),
            }, cfg.output_dir / "best_model.pt")
            (cfg.output_dir / "checkpoint_info.json").write_text(
                json.dumps({"epoch": epoch + 1, "val_loss": round(val_loss, 4), "val_acc": round(val_acc, 4)}),
                encoding="utf-8",
            )
        else:
            patience_counter += 1
            if patience_counter >= cfg.early_stopping_patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    return best_val_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=["super", "parent", "leaf"])
    parser.add_argument("--train-file", default="data/processed/train.jsonl")
    parser.add_argument("--val-file", default="data/processed/val.jsonl")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5, help="Encoder learning rate")
    parser.add_argument(
        "--head-lr-mult", type=float, default=50.0,
        help="Multiplier for classifier-head LR (head_lr = lr * head_lr_mult). "
             "Big heads (e.g. 3833-way leaf) need this to train at all.",
    )
    parser.add_argument(
        "--warmup-steps", type=int, default=500,
        help="Linear-warmup steps before cosine decay starts.",
    )
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument(
        "--init-encoder-from", type=str, default=None,
        help="Path to a trained checkpoint whose encoder weights should be "
             "copied in before training starts. Use this to warm-start the leaf "
             "stage from the parent stage's encoder.",
    )
    parser.add_argument("--resume", action="store_true", help="Continue from saved checkpoint")
    args = parser.parse_args()

    taxonomy = load_taxonomy(Path(os.getenv("TAXONOMY_CSV", "taxonomy_full.csv")))
    output_dir = Path(args.output_dir) if args.output_dir else Path(f"models/checkpoints/{args.stage}")

    cfg = TrainingConfig(
        stage=args.stage,
        train_file=Path(args.train_file),
        val_file=Path(args.val_file),
        output_dir=output_dir,
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        head_lr_mult=args.head_lr_mult,
        warmup_steps=args.warmup_steps,
        label_smoothing=args.label_smoothing,
        init_encoder_from=Path(args.init_encoder_from) if args.init_encoder_from else None,
    )
    best_loss = train_stage(cfg, taxonomy, resume=args.resume)
    print(f"\nTraining complete. Best val loss: {best_loss:.4f}")
    print(f"Checkpoint saved to: {output_dir}/best_model.pt")
