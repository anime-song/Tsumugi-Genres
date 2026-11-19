"""The genre head: one query per genre attends over a song's per-second encoder features.

Input ``[B, layers, seconds, 1024]``: SheetSage2's encoder states of a few layers, averaged per second.  A learnt
softmax mix of the layers is projected to ``d``; each genre's query attends over the seconds (one attention layer
and a feed-forward), and its own weight vector reads its logit.
"""
from __future__ import annotations

import torch
from torch import nn


class GenreHead(nn.Module):
    def __init__(self, n_layers: int, n_genres: int, dim: int = 1024, d: int = 256, heads: int = 4):
        super().__init__()
        self.mix = nn.Parameter(torch.zeros(n_layers))
        self.inp = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, d))
        self.queries = nn.Parameter(torch.zeros(n_genres, d))
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Dropout(0.0), nn.Linear(2 * d, d))
        self.norm2 = nn.LayerNorm(d)
        self.cls_weight = nn.Parameter(torch.zeros(n_genres, d))
        self.cls_bias = nn.Parameter(torch.zeros(n_genres))

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """``x`` ``[B, layers, seconds, dim]``, ``mask`` ``[B, seconds]`` (True: a real second) -> logits ``[B, genres]``."""
        h = self.inp((x * torch.softmax(self.mix, 0)[None, :, None, None]).sum(1))
        q = self.queries[None].expand(h.shape[0], -1, -1)
        a, _ = self.attn(q, h, h, key_padding_mask=None if mask is None else ~mask)
        e = self.norm1(q + a)
        e = self.norm2(e + self.ffn(e))
        return (e * self.cls_weight).sum(-1) + self.cls_bias
