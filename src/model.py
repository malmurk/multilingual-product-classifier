import torch
import torch.nn as nn
from transformers import AutoModel

BACKBONE = "intfloat/multilingual-e5-base"


def get_device():
    """Return CUDA (ROCm) device if available, else CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class CategoryClassifier(nn.Module):
    def __init__(self, num_classes: int, dropout: float = 0.1, backbone: str = BACKBONE,
                 attn_implementation: str | None = None):
        """
        attn_implementation: forwarded to AutoModel.from_pretrained.
            None     -> transformers picks its default (SDPA on modern versions).
            "eager"  -> plain attention; required for torch.jit.trace / legacy
                        ONNX export because SDPA's mask helpers break under
                        tracing on transformers >= 4.46 (IndexError in
                        sdpa_mask's q_length.shape[0]).
        Weights are identical between implementations, so an encoder trained
        with SDPA can be exported with eager.
        """
        super().__init__()
        # None is from_pretrained's own default (auto-select), so pass it straight through.
        self.encoder = AutoModel.from_pretrained(backbone, attn_implementation=attn_implementation)
        hidden_size = self.encoder.config.hidden_size  # derive from actual model, not constant
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Mean pooling over token embeddings
        token_embeddings = outputs.last_hidden_state  # (batch, seq_len, hidden)
        mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        pooled = torch.sum(token_embeddings * mask_expanded, dim=1)
        pooled = pooled / torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        pooled = self.dropout(pooled)
        return self.classifier(pooled)
