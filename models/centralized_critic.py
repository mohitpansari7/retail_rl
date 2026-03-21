"""
models/centralized_critic.py
──────────────────────────────
Phase 6 — Centralized critic for CTDE / MAPPO.

Maps: global_state (all stores concatenated) → scalar value V(s_global)

Used ONLY during training. Discarded at deployment.
Each actor keeps its own local policy — this critic is shared across all.

WHY separate from PolicyNetwork?
  Different input size (global_state vs local_obs), different width,
  different purpose. Mixing them would force actors to carry
  the global-state processing overhead they never use at runtime.

Architecture:
  Input  : global_state shape (n_stores × obs_dim,)
  Trunk  : Linear(global_dim → 512) → LN → Tanh  ×2 layers
  Output : scalar V(s_global) — no activation, unbounded
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional


class CentralizedCritic(nn.Module):
    """
    Shared critic that evaluates the joint state of all stores.

    Args:
        global_state_dim : n_stores × obs_dim_per_store.
        hidden_dim       : Hidden layer width. 512 — wider than actors
                           because the input is much larger.
        n_layers         : Depth. 2 layers sufficient — output is just
                           a scalar, doesn't need deep feature extraction.
    """

    def __init__(
        self,
        global_state_dim: int,
        hidden_dim: int = 512,
        n_layers: int = 2,
    ) -> None:
        super().__init__()
        self.global_state_dim = global_state_dim

        # Trunk: compresses global_state into a rich feature vector
        layers = []
        in_dim = global_state_dim
        for _ in range(n_layers):
            layers += [
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.Tanh(),
            ]
            in_dim = hidden_dim
        self.trunk = nn.Sequential(*layers)

        # Output: single scalar per input
        # No activation — V(s) is unbounded (can be any real number)
        self.value_head = nn.Linear(hidden_dim, 1)

        # Orthogonal init — same reason as PolicyNetwork
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                nn.init.zeros_(module.bias)

    def forward(self, global_state: torch.Tensor) -> torch.Tensor:
        """
        Estimate joint state value.

        Args:
            global_state : shape (batch, global_state_dim)
                           Concatenation of all stores' local observations.

        Returns:
            value : shape (batch, 1) — V(s_global)
        """
        features = self.trunk(global_state)
        return self.value_head(features)

    def get_value(self, global_state: torch.Tensor) -> torch.Tensor:
        """
        Convenience wrapper — returns value with no gradient tracking.
        Used during rollout collection for GAE bootstrapping.
        """
        with torch.no_grad():
            return self.forward(global_state)