"""A decoder-only Transformer, including explicit causal multi-head attention."""
import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from data import VOCAB_SIZE


@dataclass
class Config:
    block_size: int = 256
    width: int = 96
    heads: int = 3
    layers: int = 2
    dropout: float = 0.1

    def __post_init__(self):
        if min(self.block_size, self.width, self.heads, self.layers) < 1:
            raise ValueError("Model dimensions must be positive.")
        if self.width % self.heads:
            raise ValueError("Width must be divisible by the number of attention heads.")
        if not 0 <= self.dropout < 1:
            raise ValueError("Dropout must be between 0 and 1 (exclusive).")


class CausalAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = config.heads
        self.qkv = nn.Linear(config.width, 3 * config.width)
        self.projection = nn.Linear(config.width, config.width)
        self.attention_dropout = nn.Dropout(config.dropout)
        self.output_dropout = nn.Dropout(config.dropout)
        self.register_buffer(
            "mask", torch.tril(torch.ones(config.block_size, config.block_size, dtype=torch.bool)),
            persistent=False,
        )

    def forward(self, x):
        batch, length, width = x.shape
        head_width = width // self.heads
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [part.view(batch, length, self.heads, head_width).transpose(1, 2)
                   for part in (q, k, v)]
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(head_width)
        scores = scores.masked_fill(~self.mask[:length, :length], float("-inf"))
        weights = self.attention_dropout(F.softmax(scores, dim=-1))
        attended = (weights @ v).transpose(1, 2).contiguous().view(batch, length, width)
        return self.output_dropout(self.projection(attended))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.width)
        self.attention = CausalAttention(config)
        self.norm2 = nn.LayerNorm(config.width)
        self.mlp = nn.Sequential(
            nn.Linear(config.width, 4 * config.width), nn.GELU(),
            nn.Linear(4 * config.width, config.width), nn.Dropout(config.dropout),
        )

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class TransformerChatbot(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(VOCAB_SIZE, config.width)
        self.position_embedding = nn.Embedding(config.block_size, config.width)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.layers)])
        self.norm = nn.LayerNorm(config.width)
        self.output = nn.Linear(config.width, VOCAB_SIZE, bias=False)
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, tokens):
        length = tokens.shape[1]
        if not 1 <= length <= self.config.block_size:
            raise ValueError("Input sequence exceeds the model context window or is empty.")
        positions = torch.arange(length, device=tokens.device)
        x = self.dropout(self.token_embedding(tokens) + self.position_embedding(positions))
        return self.output(self.norm(self.blocks(x)))
