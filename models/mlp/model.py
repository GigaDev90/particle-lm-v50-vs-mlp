"""Small, editable decoder-only language model for architecture experiments.

The baseline is deliberately fixed at roughly 30 million parameters. Edit this
file for later architecture experiments; increasing model size is not required.
"""

from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class ModelConfig:
    vocab_size: int = 8192
    context_length: int = 512
    n_layers: int = 8
    width: int = 512
    n_heads: int = 8
    ff_width: int = 2048
    dropout: float = 0.0
    bias: bool = True


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        assert config.width % config.n_heads == 0
        self.n_heads = config.n_heads
        self.head_width = config.width // config.n_heads
        self.dropout = config.dropout
        self.qkv = nn.Linear(config.width, 3 * config.width, bias=config.bias)
        self.projection = nn.Linear(config.width, config.width, bias=config.bias)
        self.residual_dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        batch, length, width = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = (
            t.view(batch, length, self.n_heads, self.head_width).transpose(1, 2)
            for t in (q, k, v)
        )
        attended = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        attended = attended.transpose(1, 2).contiguous().view(batch, length, width)
        return self.residual_dropout(self.projection(attended))


class MLP(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.up = nn.Linear(config.width, config.ff_width, bias=config.bias)
        self.down = nn.Linear(config.ff_width, config.width, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.down(F.gelu(self.up(x))))


class Block(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.width, bias=config.bias)
        self.attention = CausalSelfAttention(config)
        self.mlp_norm = nn.LayerNorm(config.width, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attention(self.attention_norm(x))
        return x + self.mlp(self.mlp_norm(x))


class GPT(nn.Module):
    def __init__(self, config: ModelConfig | None = None):
        super().__init__()
        self.config = config or ModelConfig()
        c = self.config
        self.token_embedding = nn.Embedding(c.vocab_size, c.width)
        self.position_embedding = nn.Embedding(c.context_length, c.width)
        self.dropout = nn.Dropout(c.dropout)
        self.blocks = nn.ModuleList([Block(c) for _ in range(c.n_layers)])
        self.final_norm = nn.LayerNorm(c.width, bias=c.bias)
        self.lm_head = nn.Linear(c.width, c.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.apply(self._initialize)
        # Scale the two residual branch outputs as in common GPT baselines.
        for name, parameter in self.named_parameters():
            if name.endswith(("attention.projection.weight", "mlp.down.weight")):
                nn.init.normal_(parameter, mean=0.0, std=0.02 / math.sqrt(2 * c.n_layers))

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, tokens, targets=None):
        length = tokens.size(1)
        if length > self.config.context_length:
            raise ValueError(f"Sequence length {length} exceeds the configured context.")
        positions = torch.arange(length, device=tokens.device)
        x = self.dropout(self.token_embedding(tokens) + self.position_embedding(positions))
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        if targets is None:
            # Generation only needs the final position.
            return self.lm_head(x[:, [-1], :]), None
        logits = self.lm_head(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    def parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters())

    def config_dict(self):
        return asdict(self.config)
