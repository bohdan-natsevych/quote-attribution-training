"""
Maximum Performance Speaker Attribution Model

DeBERTa-v3-large based model with:
- Special tokens: [QUOTE], [ALTQUOTE], [PAR]
- Multi-layer BiLSTM for context encoding
- Multi-head cross-attention between quote and candidates
- Transformer encoder for candidate interaction
- Deep classifier with residual connections

Expected accuracy: 80-85% on PDNC test set
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DebertaV2Model, DebertaV2Tokenizer, DebertaV2Config, PreTrainedTokenizerBase
from typing import Optional, Tuple, Dict, List


class MaxPerformanceSpeakerModel(nn.Module):
    """
    Maximum performance speaker attribution model using DeBERTa-v3-large.

    Architecture:
    1. DeBERTa-large encoder (1024 hidden size)
    2. Special tokens for structure: [QUOTE], [ALTQUOTE], [PAR]
    3. Multi-layer BiLSTM for context encoding
    4. Multi-head cross-attention (quote attends to candidates)
    5. Transformer encoder for candidate interaction
    6. Deep classifier with dropout
    """

    # CURSOR: Special tokens for better structural understanding
    SPECIAL_TOKENS = ["[QUOTE]", "[ALTQUOTE]", "[PAR]"]

    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-large",
        num_lstm_layers: int = 3,
        num_attention_heads: int = 16,
        num_transformer_layers: int = 2,
        dropout: float = 0.2,
        hidden_dropout: float = 0.1,
        freeze_encoder_layers: int = 0,
        device: str = "cuda:0"  # CURSOR: Target device for the full module (moved at end of __init__).
    ):
        """
        Initialize the model.

        Args:
            model_name: HuggingFace model name (default: deberta-v3-large)
            num_lstm_layers: Number of BiLSTM layers
            num_attention_heads: Number of attention heads
            num_transformer_layers: Number of transformer encoder layers
            dropout: Dropout rate for classifier
            hidden_dropout: Dropout rate for hidden layers
            freeze_encoder_layers: Number of encoder layers to freeze (0 = none)
            device: Device to initialize model on (default: cuda:0)
        """
        super().__init__()

        self.model_name = model_name
        self.device = device

        # CURSOR: Initialize tokenizer first (CPU only - just vocabulary)
        self.tokenizer = DebertaV2Tokenizer.from_pretrained(model_name, use_fast=True)
        num_added = self.tokenizer.add_tokens(self.SPECIAL_TOKENS)
        
        # CURSOR: Store special token IDs for later use
        self.quote_token_id = self.tokenizer.convert_tokens_to_ids("[QUOTE]")
        self.altquote_token_id = self.tokenizer.convert_tokens_to_ids("[ALTQUOTE]")
        self.par_token_id = self.tokenizer.convert_tokens_to_ids("[PAR]")

        # CURSOR: Load encoder on CPU first; we move the FULL module to the requested device at the end.
        self.encoder = DebertaV2Model.from_pretrained(model_name)
        if num_added > 0:
            self.encoder.resize_token_embeddings(len(self.tokenizer))
        
        self.hidden_size = self.encoder.config.hidden_size  # 1024 for large

        # CURSOR: Freeze encoder layers if specified (for fine-tuning efficiency)
        if freeze_encoder_layers > 0:
            for i, layer in enumerate(self.encoder.encoder.layer[:freeze_encoder_layers]):
                for param in layer.parameters():
                    param.requires_grad = False

        # CURSOR: Multi-layer BiLSTM for context encoding
        self.context_lstm = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size // 2,
            num_layers=num_lstm_layers,
            bidirectional=True,
            dropout=hidden_dropout if num_lstm_layers > 1 else 0,
            batch_first=True
        )

        # CURSOR: Multi-head cross-attention (quote attends to candidates)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=num_attention_heads,
            dropout=hidden_dropout,
            batch_first=True
        )

        # CURSOR: Self-attention for candidates
        self.candidate_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=num_attention_heads,
            dropout=hidden_dropout,
            batch_first=True
        )

        # CURSOR: Transformer encoder for candidate interaction
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=num_attention_heads,
            dim_feedforward=self.hidden_size * 4,
            dropout=hidden_dropout,
            activation='gelu',
            batch_first=True
        )
        self.candidate_transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_transformer_layers
        )

        # CURSOR: Layer normalization for stability
        self.layer_norm = nn.LayerNorm(self.hidden_size)

        # CURSOR: Deep classifier with residual-style connections
        # Input: concat of [candidate_emb, quote_emb, cross_attended, self_attended, element_wise_mult]
        classifier_input_size = self.hidden_size * 5

        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_size, 2048),
            nn.LayerNorm(2048),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2048, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),  # Less dropout in final layers
            nn.Linear(512, 1)
        )

        # CURSOR: Initialize weights
        self._init_weights()

        # CURSOR: Explicitly move the FULL module (all params + buffers) to the requested device.
        target_device = torch.device(device)
        if target_device.type == "cuda" and not torch.cuda.is_available():
            # CURSOR: Safe fallback for CPU-only environments (unit tests, local runs without GPU).
            target_device = torch.device("cpu")
        self.to(target_device)
        self.device = str(target_device)

    def _init_weights(self):
        """Initialize classifier weights using Xavier initialization."""
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Enable gradient checkpointing for the encoder."""
        # CURSOR: Prefer non-reentrant checkpointing; it is more compatible with DataParallel + AMP and avoids
        # CURSOR: "Trying to backward through the graph a second time" errors seen with reentrant checkpointing.
        if gradient_checkpointing_kwargs is None:
            gradient_checkpointing_kwargs = {"use_reentrant": False}
        try:
            self.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
        except TypeError:
            # CURSOR: Backwards compatibility for older Transformers versions.
            self.encoder.gradient_checkpointing_enable()

    def get_tokenizer(self) -> PreTrainedTokenizerBase:
        """Return the tokenizer with special tokens."""
        return self.tokenizer

    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode input text using DeBERTa.

        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]

        Returns:
            Hidden states [batch, seq_len, hidden_size]
        """
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        return outputs.last_hidden_state

    def extract_span_embedding(
        self,
        hidden_states: torch.Tensor,
        span_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Extract embedding for a span using masked mean pooling.

        Args:
            hidden_states: [batch, seq_len, hidden_size]
            span_mask: [batch, seq_len] binary mask for the span

        Returns:
            Span embedding [batch, hidden_size]
        """
        # Expand mask for broadcasting
        mask_expanded = span_mask.unsqueeze(-1).float()

        # Masked sum
        sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)

        # Count non-zero elements (avoid division by zero)
        sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)

        # Mean pooling
        return sum_embeddings / sum_mask

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        quote_mask: torch.Tensor,
        candidate_masks: torch.Tensor,
        candidate_attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for speaker attribution.

        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            quote_mask: Binary mask for quote tokens [batch, seq_len]
            candidate_masks: Binary masks for each candidate [batch, num_candidates, seq_len]
            candidate_attention_mask: Optional mask for valid candidates [batch, num_candidates]

        Returns:
            logits: Speaker scores [batch, num_candidates]
            attention_weights: Cross-attention weights for interpretability
        """
        batch_size = input_ids.size(0)
        num_candidates = candidate_masks.size(1)

        # CURSOR: Encode full context with DeBERTa
        hidden_states = self.encode_text(input_ids, attention_mask)

        # CURSOR: Apply BiLSTM for sequential context
        lstm_output, _ = self.context_lstm(hidden_states)

        # CURSOR: Combine DeBERTa and LSTM outputs (residual connection)
        context_encoded = self.layer_norm(hidden_states + lstm_output)

        # CURSOR: Extract quote embedding
        quote_emb = self.extract_span_embedding(context_encoded, quote_mask)

        # CURSOR: Extract candidate embeddings
        candidate_embs = []
        for i in range(num_candidates):
            cand_mask = candidate_masks[:, i, :]  # [batch, seq_len]
            cand_emb = self.extract_span_embedding(context_encoded, cand_mask)
            candidate_embs.append(cand_emb)

        # Stack candidates: [batch, num_candidates, hidden_size]
        candidate_embs = torch.stack(candidate_embs, dim=1)

        # CURSOR: Apply transformer to model candidate interactions
        if candidate_attention_mask is not None:
            # Create proper attention mask for transformer (True = masked out)
            transformer_mask = ~candidate_attention_mask.bool()
        else:
            transformer_mask = None

        candidate_transformed = self.candidate_transformer(
            candidate_embs,
            src_key_padding_mask=transformer_mask
        )

        # CURSOR: Cross-attention: quote attends to candidates
        quote_expanded = quote_emb.unsqueeze(1).expand(-1, num_candidates, -1)

        cross_attended, attn_weights = self.cross_attention(
            query=quote_expanded,
            key=candidate_transformed,
            value=candidate_transformed,
            key_padding_mask=transformer_mask
        )

        # CURSOR: Self-attention among candidates
        self_attended, _ = self.candidate_attention(
            query=candidate_transformed,
            key=candidate_transformed,
            value=candidate_transformed,
            key_padding_mask=transformer_mask
        )

        # CURSOR: Element-wise multiplication for interaction
        element_mult = quote_expanded * candidate_transformed

        # CURSOR: Concatenate all representations
        combined = torch.cat([
            candidate_transformed,      # Candidate embeddings
            quote_expanded,             # Quote embedding (repeated)
            cross_attended,             # Quote-to-candidate attention
            self_attended,              # Candidate self-attention
            element_mult                # Element-wise interaction
        ], dim=-1)

        # CURSOR: Classify each candidate
        logits = self.classifier(combined).squeeze(-1)  # [batch, num_candidates]

        # CURSOR: Mask out invalid candidates
        if candidate_attention_mask is not None:
            logits = logits.masked_fill(~candidate_attention_mask.bool(), float('-inf'))

        return logits, attn_weights

    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        quote_mask: torch.Tensor,
        candidate_masks: torch.Tensor,
        candidate_attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict speaker for a quote.

        Returns:
            predictions: Predicted speaker indices [batch]
            probabilities: Softmax probabilities [batch, num_candidates]
            attention_weights: Cross-attention weights
        """
        self.eval()
        with torch.no_grad():
            logits, attn_weights = self.forward(
                input_ids, attention_mask, quote_mask,
                candidate_masks, candidate_attention_mask
            )

            probabilities = F.softmax(logits, dim=-1)
            predictions = torch.argmax(probabilities, dim=-1)

        return predictions, probabilities, attn_weights

    def save_pretrained(self, save_path: str):
        """Save model and tokenizer."""
        import os
        os.makedirs(save_path, exist_ok=True)

        # Save model state
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': {
                'model_name': self.model_name,
                'hidden_size': self.hidden_size,
            }
        }, os.path.join(save_path, 'model.pt'))

        # Save tokenizer
        self.tokenizer.save_pretrained(save_path)

    @classmethod
    def from_pretrained(cls, load_path: str, device: str = 'cpu'):
        """Load model from saved checkpoint."""
        import os

        checkpoint = torch.load(
            os.path.join(load_path, 'model.pt'),
            map_location=device
        )

        model = cls(model_name=checkpoint['config']['model_name'], device=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        # CURSOR: Model is already initialized on requested device; keep an explicit .to() for safety.
        model.to(device)

        return model


def create_model(
    model_name: str = "microsoft/deberta-v3-large",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    **kwargs
) -> MaxPerformanceSpeakerModel:
    """
    Factory function to create and initialize the model.

    Args:
        model_name: HuggingFace model name
        device: Device to place model on
        **kwargs: Additional arguments passed to model constructor

    Returns:
        Initialized model on specified device
    """
    # CURSOR: Allow callers to pass device either as an explicit arg or inside **kwargs without crashing.
    if "device" in kwargs:
        device = kwargs.pop("device")

    # CURSOR: Pass device into constructor to avoid unnecessary CPU->GPU->CPU moves.
    model = MaxPerformanceSpeakerModel(model_name=model_name, device=device, **kwargs)
    return model


# CURSOR: Export public API
__all__ = [
    'MaxPerformanceSpeakerModel',
    'create_model'
]
