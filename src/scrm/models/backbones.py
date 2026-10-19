# 活动词表、时间缩放与本框架的 Transformer 骨干。
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import numpy as np
import torch
import torch.nn as nn

class ActivityVocab:
    PAD = '<PAD>'
    UNK = '<UNK>'

    def __init__(self):
        self.token_to_id = {self.PAD: 0, self.UNK: 1}

    def fit(self, samples):
        tokens = sorted({t for s in samples for t in s.activity_names})
        for t in tokens:
            if t not in self.token_to_id:
                self.token_to_id[t] = len(self.token_to_id)
        return self

    def encode(self, names: Sequence[str]) -> list[int]:
        unk = self.token_to_id[self.UNK]
        return [self.token_to_id.get(str(x), unk) for x in names]

    def __len__(self):
        return len(self.token_to_id)

@dataclass
class TimeScaler:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, samples):
        rows = [s.time_features for s in samples if len(s.time_features)]
        if not rows:
            return cls(np.zeros(6, dtype=np.float32), np.ones(6, dtype=np.float32))
        x = np.concatenate(rows, axis=0).astype(np.float32)
        mean = x.mean(axis=0)
        std = x.std(axis=0)
        std = np.where(std < 1e-06, 1.0, std)
        return cls(mean.astype(np.float32), std.astype(np.float32))

    def transform(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.std).astype(np.float32)

class TransformerRegressor(nn.Module):

    def __init__(self, vocab_size: int, time_dim: int=6, embedding_dim: int=32, hidden_dim: int=64, n_layers: int=2, n_heads: int=4, dropout: float=0.1, max_len: int=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.input_proj = nn.Linear(embedding_dim + time_dim, hidden_dim)
        self.pos_embedding = nn.Embedding(max_len, hidden_dim)
        layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim * 2, dropout=dropout, activation='gelu', batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(), nn.Linear(hidden_dim // 2, 1))

    def encode(self, activity_ids, time_features, lengths):
        bsz, seq_len = activity_ids.shape
        emb = self.embedding(activity_ids)
        x = torch.cat([emb, time_features], dim=-1)
        x = self.input_proj(x)
        pos = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(bsz, seq_len)
        x = x + self.pos_embedding(pos)
        padding_mask = activity_ids.eq(0)
        x = self.encoder(x, src_key_padding_mask=padding_mask)
        idx = (lengths - 1).clamp_min(0)
        last = x[torch.arange(bsz, device=x.device), idx]
        return self.norm(last)

    def forward(self, activity_ids, time_features, lengths):
        return self.head(self.encode(activity_ids, time_features, lengths)).squeeze(-1)
