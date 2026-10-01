"""Warehouse demand model: first-principles attention + a linear prediction head."""
import torch
import torch.nn as nn
from src.attention import SelfAttention


class WarehouseModel(nn.Module):
    def __init__(self, d_in=5, d_k=16, scale=True, init_std=None, uniform=False):
        super().__init__()
        self.attn = SelfAttention(d_in, d_k, scale=scale, init_std=init_std)
        self.head = nn.Linear(d_k + 4, 1)      # context (d_k) + time features of the target hour (4)
        self.uniform = uniform                 # control: replace learned weights by 1/T everywhere

    def forward(self, X, next_tf):
        """X: (B, 24, 5), next_tf: (B, 4). Returns normalised prediction (B,) and weights A (B, 24, 24)."""
        if self.uniform:
            B, T, _ = X.shape
            V = X @ self.attn.W_V
            Y = V.mean(dim=1, keepdim=True).expand(-1, T, -1)     # A = 1/T: plain average of reports
            A = torch.full((B, T, T), 1.0 / T)
        else:
            Y, A = self.attn(X)                # every hour mixes the reports of all 24 hours
        context = Y[:, -1, :]                  # last hour's summary
        out = self.head(torch.cat([context, next_tf], dim=-1)).squeeze(-1)
        return out, A