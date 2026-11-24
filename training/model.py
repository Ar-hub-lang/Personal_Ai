import torch
from torch import nn

from training.tokenizer import ByteTokenizer


TEXT_CONTEXT_LENGTH = 128
DOCUMENT_CONTEXT_LENGTH = 640


class TinyGPT(nn.Module):
    """A decoder-only Transformer that predicts the next token at each position."""

    def __init__(
        self,
        *,
        vocab_size: int = ByteTokenizer.vocab_size,
        context_length: int = TEXT_CONTEXT_LENGTH,
        width: int = 128,
        heads: int = 4,
        layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.context_length = context_length
        self.token_embedding = nn.Embedding(vocab_size, width)
        self.position_embedding = nn.Embedding(context_length, width)
        self.blocks = nn.ModuleList(
            nn.TransformerEncoderLayer(
                d_model=width,
                nhead=heads,
                dim_feedforward=4 * width,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(layers)
        )
        self.final_norm = nn.LayerNorm(width)
        self.output = nn.Linear(width, vocab_size, bias=False)
        self.register_buffer(
            "causal_mask",
            torch.ones(context_length, context_length, dtype=torch.bool).triu(diagonal=1),
            persistent=False,
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 2 or not 0 < tokens.shape[1] <= self.context_length:
            raise ValueError(f"Expected token IDs shaped [batch, sequence] with 1–{self.context_length} positions.")

        length = tokens.shape[1]
        positions = torch.arange(length, device=tokens.device)
        hidden = self.token_embedding(tokens) + self.position_embedding(positions)
        mask = self.causal_mask[:length, :length]
        for block in self.blocks:
            hidden = block(hidden, src_mask=mask, is_causal=True)
        return self.output(self.final_norm(hidden))
