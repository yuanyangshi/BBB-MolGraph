"""
MolFormer molecular language model feature extractor and standalone sequence classifier.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn

from bbb_molgraph.core.constants import DEFAULT_MOLFORMER_MODEL_NAME, MOLFORMER_EMBEDDING_DIM
from bbb_molgraph.core.registry import MODELS
from bbb_molgraph.utils.logging import logger


class SimpleCharTokenizer:
    """Fallback character-level tokenizer for SMILES strings when offline."""

    def __init__(self, vocab_size: int = 1000) -> None:
        self.vocab_size = vocab_size

    def __call__(
        self,
        texts: Union[str, List[str]],
        return_tensors: str = "pt",
        padding: bool = True,
        truncation: bool = True,
        max_length: int = 256,
    ) -> Dict[str, Any]:
        if isinstance(texts, str):
            texts = [texts]
        batch_ids = []
        for text in texts:
            chars = text[:max_length] if truncation else text
            ids = [(ord(c) % (self.vocab_size - 1)) + 1 for c in chars]
            if not ids:
                ids = [1]
            batch_ids.append(ids)

        max_len = max(len(ids) for ids in batch_ids) if batch_ids else 0
        if padding and max_len > 0:
            padded_ids = []
            padded_mask = []
            for ids in batch_ids:
                pad_len = max_len - len(ids)
                padded_ids.append(ids + [0] * pad_len)
                padded_mask.append([1] * len(ids) + [0] * pad_len)
            batch_ids = padded_ids
            batch_mask = padded_mask
        else:
            batch_mask = [[1] * len(ids) for ids in batch_ids]

        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor(batch_ids, dtype=torch.long),
                "attention_mask": torch.tensor(batch_mask, dtype=torch.long),
            }
        return {"input_ids": batch_ids, "attention_mask": batch_mask}


class LightweightSequenceEncoder(nn.Module):
    """
    Lightweight transformer encoder used as offline/fast fallback when HuggingFace
    hub is unreachable or during unit testing.
    """

    def __init__(self, vocab_size: int = 1000, embed_dim: int = 128, num_layers: int = 2) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=4,
            dim_feedforward=embed_dim * 2,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out_dim = embed_dim

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x = self.embedding(input_ids)
        key_padding_mask = (attention_mask == 0) if attention_mask is not None else None
        h = self.transformer(x, src_key_padding_mask=key_padding_mask)

        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).float()
            pooled = (h * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1.0)
        else:
            pooled = h.mean(dim=1)

        # Uniform dummy attention weights
        seq_attn = attention_mask.float() if attention_mask is not None else torch.ones_like(input_ids).float()
        return pooled, seq_attn


@MODELS.register("molformer_encoder")
class MolFormerEncoder(nn.Module):
    """
    Wrapper for pretrained MolFormer molecular sequence Transformer.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MOLFORMER_MODEL_NAME,
        out_dim: int = 64,
        freeze: bool = True,
        use_fallback: bool = False,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.out_dim = out_dim
        self.freeze = freeze
        self.use_fallback = use_fallback

        if not use_fallback:
            try:
                from transformers import AutoModel, AutoTokenizer
                self.backbone = AutoModel.from_pretrained(model_name, trust_remote_code=True)
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                except Exception:
                    self.tokenizer = SimpleCharTokenizer()
                raw_dim = getattr(self.backbone.config, "hidden_size", MOLFORMER_EMBEDDING_DIM)
            except Exception as e:
                logger.warning(
                    f"Could not load HuggingFace model '{model_name}' ({e}). "
                    "Falling back to LightweightSequenceEncoder."
                )
                self.backbone = LightweightSequenceEncoder(embed_dim=128)
                self.tokenizer = SimpleCharTokenizer()
                raw_dim = 128
                self.use_fallback = True
        else:
            self.backbone = LightweightSequenceEncoder(embed_dim=128)
            self.tokenizer = SimpleCharTokenizer()
            raw_dim = 128

        if self.freeze and not self.use_fallback:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.seq_proj = nn.Sequential(
            nn.Linear(raw_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)

        Returns:
            seq_embeddings: (batch_size, out_dim)
            seq_attentions: (batch_size, seq_len) or attention tensor
        """
        if self.use_fallback:
            pooled, seq_attn = self.backbone(input_ids, attention_mask)
            projected = self.seq_proj(pooled)
            return projected, seq_attn

        if self.freeze:
            self.backbone.eval()

        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
        )

        last_hidden_state = outputs.last_hidden_state  # (batch_size, seq_len, hidden_dim)

        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).float()
            pooled = (last_hidden_state * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1.0)
        else:
            pooled = last_hidden_state.mean(dim=1)

        projected = self.seq_proj(pooled)

        # Extract attention weights if available
        seq_attn = None
        if hasattr(outputs, "attentions") and outputs.attentions is not None:
            # Average over heads in the final layer
            last_layer_attn = outputs.attentions[-1]  # (batch_size, heads, seq_len, seq_len)
            seq_attn = last_layer_attn.mean(dim=1).mean(dim=1)  # (batch_size, seq_len)

        return projected, seq_attn


@MODELS.register("molformer_classifier")
class MolFormerClassifier(nn.Module):
    """
    Standalone sequence classification baseline based on MolFormer.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MOLFORMER_MODEL_NAME,
        out_dim: int = 64,
        freeze: bool = True,
        dropout: float = 0.2,
        use_fallback: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = MolFormerEncoder(
            model_name=model_name,
            out_dim=out_dim,
            freeze=freeze,
            use_fallback=use_fallback,
        )
        self.classifier = nn.Sequential(
            nn.Linear(out_dim, out_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim // 2, 1),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        embeddings, seq_attn = self.encoder(input_ids, attention_mask)
        logits = self.classifier(embeddings)
        return logits, seq_attn
