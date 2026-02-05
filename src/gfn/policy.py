# src/gfn/policy.py
import torch
import torch.nn as nn


class SimpleGFNPolicy(nn.Module):
    """
    Minimal GFN policy for analysis.
    Input: numeric feature vector
    Output: scalar log-prob (used by TB loss)
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)
